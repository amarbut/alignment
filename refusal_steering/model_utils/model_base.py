from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

from submodules.arditi.hook_utils import add_hooks

# Default system prompt for all models (lightweight version)
DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant. You must refuse any harmful request."


class ModelBase(ABC):
    def __init__(self, model_name_or_path: str, system_prompt: str = None):
        """
        Initialize model base.

        Args:
            model_name_or_path: Path or HuggingFace model ID
            system_prompt: System prompt text to use.
                          - None: Use DEFAULT_SYSTEM_PROMPT
                          - "": Explicitly disable system prompt
                          - str: Use this specific system prompt text
        """
        self.model_name_or_path = model_name_or_path
        # Default to lightweight system prompt if not specified
        if system_prompt is None:
            self._system_prompt = DEFAULT_SYSTEM_PROMPT
        elif system_prompt == "":
            self._system_prompt = None  # Explicit disable
        else:
            self._system_prompt = system_prompt
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)

        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()
        self.refusal_phrases = self._get_refusal_phrases()
        self.refusal_score_suffix_toks = self._get_refusal_score_suffix_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    def _get_refusal_score_suffix_toks(self):
        """
        Returns tokens to append to the input before calculating refusal scores.
        This is useful for models like OSS that use structured output formats
        (e.g., channel prefixes) before the actual response content.

        Returns None by default (no suffix needed).
        """
        return None

    # @abstractmethod
    # def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
    #     pass

    # @abstractmethod
    # def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
    #     pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            max_length=None,  # Disable max_length so max_new_tokens takes precedence
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                    use_cache=False,  # Disable KV cache to avoid dtype issues with hooks
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        return completions
