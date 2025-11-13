import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks, get_all_subspace_ablation_hooks, get_activation_addition_subspace_input_pre_hook, add_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.generate_activations import get_activations
from pipeline.submodules.cpca_functions import choose_best_cpca
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--skip_generate', type=bool, default=False)
    parser.add_argument('--skip_select', type=bool, default=False)
    parser.add_argument('--align', type=bool, default=False)
    parser.add_argument('--baseline', type=bool, default=True)
    parser.add_argument('--ablate', type=bool, default=True)
    parser.add_argument('--actadd', type=bool, default=True)
    parser.add_argument('--method', type=str, default="arditi", help="direction/subspace pipeline to use", choices=["arditi", "cpca", "pls", "nonlinear", "arditi_auc"])
    parser.add_argument('--topk', type=int, default=1, help="Number of components to include in subspace; topk=1 is a single vector")
    parser.add_argument('--coeff', type=float, default=1.0, help="Scaling for actadd intervention")
    parser.add_argument('--tau', type=float, default=1.0, help="Scaling for ablation intervention")
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, tokenizer = model_base.tokenizer)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, tokenizer = model_base.tokenizer)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y) #REMEMBER TO CHANGE BACK

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, tokenizer = model_base.tokenizer)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, tokenizer = model_base.tokenizer)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y) # REMEMBER TO CHANGE BACK
    
    return harmful_train, harmless_train, harmful_val, harmless_val

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return [mean_diffs]

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, topk, align):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions[0],
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    mu_b = torch.zeros(basis.shape[0], device=basis.device, dtype=basis.dtype)
    
    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')
    
    torch.save(mu_b, f'{cfg.artifact_path()}/mu_b.pt')

    

    return pos, layer, direction, mu_b

def generate_and_save_activations(cfg, model_base, harmless_train, harmful_train, batch_size: int = 32):
    
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_acts')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_acts'))

    hf_acts = get_activations(
        model_base,
        harmful_train,
        batch_size=batch_size,
    )
    out_path = os.path.join(cfg.artifact_path(), "generate_acts/hf_activations.pt")
    torch.save(hf_acts, out_path)

    hl_acts = get_activations(
        model_base,
        harmless_train,
        batch_size=batch_size,
    )
    out_path = os.path.join(cfg.artifact_path(), "generate_acts/hl_activations.pt")
    torch.save(hl_acts, out_path)
    return hl_acts, hf_acts


def select_and_save_cpca(cfg, model_base, harmful_val, harmless_val, cands, topk, align, keep_var = 0.999, eig_floor_frac = 1e-3):
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_cpca')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_cpca'))

    layer, pos, res = choose_best_cpca(cands[1], cands[0], topk, keep_var, eig_floor_frac, align)
    pos = pos-5

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(res, f'{cfg.artifact_path()}/select_cpca/subspace_res.pt')
    torch.save(res["components_norm"][:,:topk], f'{cfg.artifact_path()}/direction.pt')
    torch.save(res["mu_b"], f'{cfg.artifact_path()}/mu_b.pt')

    return layer, pos, res["components_norm"][:,:topk], res["mu_b"]


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w", encoding = "utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r', encoding="utf-8") as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

method_dict = {"arditi": {"candidate_loc": ["generate_directions/mean_diffs.pt"],
                          "generate_cands": generate_and_save_candidate_directions,
                          "select_dirs": select_and_save_direction
                         },
               "cpca": {"candidate_loc": ["generate_acts/hl_activations.pt", "generate_acts/hf_activations.pt"],
                        "generate_cands": generate_and_save_activations,
                        "select_dirs": select_and_save_cpca
                       }
              }



def run_pipeline(model_path, skip_generate, skip_select, method, topk, coeff, tau, align, baseline, ablate, actadd):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)+f"/{method}"
    cfg = Config(model_alias=model_alias, model_path=model_path)
    method_args = method_dict[method]

    model_base = construct_model_base(cfg.model_path)
    model_base.model.config.pad_token_id = model_base.tokenizer.pad_token_id

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    print("Raw data:", len(harmful_train), len(harmless_train))
    
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)
    print("Filtered_data:", len(harmful_train), len(harmless_train))

    # 1. Generate or load candidate refusal directions
    if skip_generate:
        cands = [torch.load(os.path.join(cfg.artifact_path(), p), map_location = "cpu") for p in method_args["candidate_loc"]]
    else:
        cands = method_args["generate_cands"](cfg, model_base, harmless_train, harmful_train)
    
    # 2. Select or load the most effective refusal direction/subspace
    if skip_select:
        meta = json.load(open(f'{cfg.artifact_path()}/direction_metadata.json', "r"))
        layer = meta["layer"]
        pos = meta["pos"]
        mu_b = meta["mu_b"]
        direction = torch.load(f'{cfg.artifact_path()}/direction.pt', map_location = "cpu")
    else:
        layer, pos, direction, mu_b = method_args["select_dirs"](cfg, model_base, harmful_val, harmless_val, cands, topk = topk, align=align) 

    print(f"best dir @ layer {layer} and pos {pos}")

    direction = direction.to(model_base.model.device, dtype=getattr(model_base.model, "dtype", torch.float16))

    if direction.dim() == 1:
        direction = direction.unsqueeze(-1)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_subspace_ablation_hooks(model_base, direction, mu_b, tau) 
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_subspace_input_pre_hook(direction, coeffs=torch.tensor(-coeff)))], []
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_subspace_input_pre_hook(direction, coeffs=torch.tensor(+coeff)))], []
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    if baseline:
        print("Running baseline completions")
        for dataset_name in cfg.evaluation_datasets:
            print("harmful evaluation datasets")
            generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
            evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            print("harmless test set")
            generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
            evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    if ablate:
        print("Running ablation completions")
        for dataset_name in cfg.evaluation_datasets:
            generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
            evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

    if actadd:
        print("Running actadd completions")
        for dataset_name in cfg.evaluation_datasets:
            print("harmful evaluation datasets")
            generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)
            evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
            print("harmless test set")
            generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)
            evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    
    # print("Evaluate loss on harmless datasets")
    # evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    # evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    # evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, skip_generate=args.skip_generate, skip_select=args.skip_select, method=args.method, topk=args.topk, coeff=args.coeff, tau=args.tau, align=args.align, baseline=args.baseline, ablate=args.ablate, actadd=args.actadd)
