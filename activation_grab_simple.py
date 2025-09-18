import torch
import numpy as np
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------- Load model ----------
model_path = "/meta-llama/Llama-3.2-8B-Instruct"
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

# ---------- Hook-based residual-stream capture ----------
class ResidualCapture:
    """
    Capture *input hidden_states* to each LlamaDecoderLayer via forward_pre_hook.
    Stores only the last K steps (newly generated tokens).
    """
    def __init__(self, model, layer_indices=None, last_token_only=True, to_cpu=True, last_k_steps=5):
        self.model = model
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        self.layer_indices = layer_indices if layer_indices is not None else list(range(self.num_layers))
        self.last_token_only = last_token_only
        self.to_cpu = to_cpu
        self.last_k_steps = last_k_steps

        self._hooks = []
        self._current_step_buffer = {}                    # layer_idx -> np.array(hidden_dim,)
        self.captures_deque = deque(maxlen=last_k_steps)  # rolling window

    def _pre_hook(self, layer_idx):
        def hook(module, inputs):
            # inputs[0] = hidden_states entering this layer
            hidden_states = inputs[0]  # [batch, seq_len, hidden_dim]
            hs = hidden_states[:, -1, :] if self.last_token_only else hidden_states  # we assume last token
            hs = hs.detach().to("cpu") if self.to_cpu else hs.detach()
            arr = hs.squeeze(0).contiguous().float().numpy()  # [hidden_dim] (batch=1)
            self._current_step_buffer[layer_idx] = arr
        return hook

    def install(self):
        for i in self.layer_indices:
            h = self.layers[i].register_forward_pre_hook(self._pre_hook(i), with_kwargs=False)
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def start_step(self):
        self._current_step_buffer = {}

    def end_step(self):
        # append snapshot of this step into rolling window
        self.captures_deque.append(dict(self._current_step_buffer))

    @property
    def captures(self):
        # return a list (oldest->newest) of up to last_k_steps step dicts
        return list(self.captures_deque)

def generate_and_capture_last_k(
    prompt: str,
    max_new_tokens: int = 16,
    last_k_steps: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    layer_indices=None,
):
    """
    Generate up to max_new_tokens while keeping only the last_k_steps of per-step activations
    (residual-stream inputs) for the newly generated token.

    Returns:
        text: decoded full text
        captures: list (len <= last_k_steps) of dict[layer_idx -> np.ndarray(hidden_dim,)]
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask", None)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    def pick(logits):
        # function for choosing the output token
        # default is greedy--chooses token with highest prob
        # other settings give options to tune the stochasticity of the choice (temperature controls randomness of choice, top_p removes long tail of unlikely tokens)
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_idx = torch.sort(probs, descending=True)
                csum = torch.cumsum(sorted_p, dim=-1)
                cutoff = csum > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_p = torch.where(cutoff, torch.zeros_like(sorted_p), sorted_p)
                sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(sorted_p, 1)
                return sorted_idx.gather(-1, idx)
            return torch.multinomial(probs, 1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    rc = ResidualCapture(
        model,
        layer_indices=layer_indices,
        last_token_only=True,
        to_cpu=True,
        last_k_steps=last_k_steps,
    )
    rc.install()

    gen_ids = []
    try:
        # this is the full loop
        # 1. pick the next token
        # 2. do a forward pass
        # 3. store the residuals/activations for that token
        # 4. clean up memory before next token
        for _ in range(max_new_tokens):
            next_id = pick(next_logits)
            step_ids = next_id
            step_attn = torch.ones_like(step_ids, device=device)

            rc.start_step()
            with torch.no_grad():
                out = model(
                    input_ids=step_ids,
                    attention_mask=step_attn,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            rc.end_step()

            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            gen_ids.append(int(next_id.item()))

            if tokenizer.eos_token_id is not None and gen_ids[-1] == tokenizer.eos_token_id:
                break
    finally:
        rc.remove()

    full_ids = torch.cat([input_ids, torch.tensor([gen_ids], device=input_ids.device)], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return text, rc.captures

# ------------- Example -------------
#if __name__ == "__main__":
    prompt = "Explain why the sky appears blue in simple terms."
    text, last_k_caps = generate_and_capture_last_k(
        prompt,
        max_new_tokens=64,
        last_k_steps=5,        # keep only last 5 response tokens' activations
        temperature=0.0,
        do_sample=False,
        layer_indices=None,    # or a subset like [0, 12, 23]
    )
    print(text.splitlines()[:3])
    print(f"kept steps: {len(last_k_caps)}")
    if last_k_caps:
        any_layer = sorted(last_k_caps[-1].keys())[0]
        print("shape of last step, some layer:", last_k_caps[-1][any_layer].shape)
