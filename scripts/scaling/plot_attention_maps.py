#!/usr/bin/env python
"""Plot attention maps for a single epitope-TCR pair across all 5 SFT models.

Runs on wolf (needs GPU + checkpoints). Saves figure to results/figures/.

Usage:
    python scripts/scaling/plot_attention_maps.py \
        --epitope GILGFVFTL \
        --tcr CASSIRSSYEQYF \
        --saveto results/figures/attention_maps.png
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("PROJECT_ROOT", os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch

from byprot import utils
from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
from byprot.utils.config import load_yaml_config

SEP_TOKEN = "<sep>"

CHECKPOINTS = {
    "0.1M": "logs/tcr_sft_0.1m/checkpoints/best.ckpt",
    "1M":   "logs/tcr_sft_1m/checkpoints/best.ckpt",
    "5M":   "logs/tcr_sft_5m/checkpoints/best.ckpt",
    "15M":  "logs/tcr_sft_15m/checkpoints/best.ckpt",
    "50M":  "logs/tcr_sft_50m/checkpoints/best.ckpt",
}


def load_sft_model(checkpoint_path):
    cfg_path = Path(checkpoint_path).parents[1] / ".hydra" / "config.yaml"
    cfg = load_yaml_config(str(cfg_path)).model
    cfg.net.pretrain = False
    cfg.net.sft_from_checkpoint = None
    cfg.pop("_target_")
    model = DiffusionProteinLanguageModel(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    new_state = OrderedDict((k[6:], v) for k, v in ckpt.items())
    model.load_state_dict(new_state, strict=False)
    return model.eval()


def get_attention(model, tokenizer, sep_token_id, epitope, tcr, device):
    """Run one forward pass, return averaged last-layer attention [seq_len, seq_len].

    Uses forward hooks to capture Q*K^T attention weights since the custom
    EsmSelfAttention uses F.scaled_dot_product_attention which discards weights.
    """
    from byprot.models.dplm.modules.dplm_modeling_esm import EsmSelfAttention

    # Collect all EsmSelfAttention layers in order
    attn_layers = [(name, mod) for name, mod in model.named_modules()
                   if isinstance(mod, EsmSelfAttention)]

    layer_attentions = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            hidden_states = inputs[0]
            with torch.no_grad():
                q = module.transpose_for_scores(module.query(hidden_states))
                k = module.transpose_for_scores(module.key(hidden_states))
                q = q * module.attention_head_size ** -0.5
                if module.position_embedding_type == "rotary":
                    q, k = module.rotary_embeddings(q, k)
                attn_w = torch.matmul(q.contiguous(), k.contiguous().transpose(-2, -1))
                attn_w = torch.softmax(attn_w, dim=-1)
                layer_attentions[layer_name] = attn_w[0].detach().cpu()  # [heads, L, L]
        return hook

    for name, mod in attn_layers:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    # Build input tokens
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    epi_ids = tokenizer.encode(epitope, add_special_tokens=False)
    tcr_ids = tokenizer.encode(tcr,     add_special_tokens=False)
    ids = [cls_id] + epi_ids + [sep_token_id] + tcr_ids + [eos_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        _ = model(input_ids)

    for h in hooks:
        h.remove()

    # Use last layer, averaged over heads
    last_key = sorted(layer_attentions.keys())[-1]
    avg_attn = layer_attentions[last_key].mean(0).numpy()  # [L, L]

    tokens = ["<cls>"] + list(epitope) + ["<sep>"] + list(tcr) + ["<eos>"]
    return avg_attn, tokens


def plot_all(attentions_list, model_names, epitope, tcr, outpath):
    n = len(attentions_list)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    fig.suptitle(
        f"TCR → Epitope Attention (last layer, avg heads)\n"
        f"Epitope: {epitope}  |  TCR: {tcr}",
        fontsize=13, fontweight="bold",
    )

    epi_tokens = list(epitope)
    tcr_tokens = list(tcr)

    # Extract submatrix: rows=TCR positions, cols=epitope positions
    # Full token order: <cls>, *epitope, <sep>, *tcr, <eos>
    # epitope cols: indices 1 .. len(epitope)
    # tcr rows:     indices len(epitope)+2 .. len(epitope)+2+len(tcr)-1
    epi_start = 1
    epi_end   = 1 + len(epitope)          # exclusive
    tcr_start = epi_end + 1               # skip <sep>
    tcr_end   = tcr_start + len(tcr)      # exclusive

    submats = []
    for attn, _ in attentions_list:
        sub = attn[tcr_start:tcr_end, epi_start:epi_end]  # [len_tcr, len_epi]
        submats.append(sub)

    vmax = max(s.max() for s in submats)

    for ax, sub, name in zip(axes, submats, model_names):
        im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(epi_tokens)))
        ax.set_xticklabels(epi_tokens, rotation=45, fontsize=8, ha="right")
        ax.set_yticks(range(len(tcr_tokens)))
        ax.set_yticklabels(tcr_tokens, fontsize=8)
        ax.set_xlabel("Epitope", fontsize=9)
        ax.set_ylabel("TCR", fontsize=9)

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Attention weight")
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epitope", type=str, default="GILGFVFTL")
    parser.add_argument("--tcr",     type=str, default="CASSIRSSYEQYF")
    parser.add_argument("--saveto",  type=str, default="results/figures/attention_maps.png")
    parser.add_argument("--cpu",     action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    attentions_list = []
    model_names = []

    for name, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"Checkpoint not found: {ckpt_path}, skipping {name}")
            continue
        print(f"Loading {name} from {ckpt_path} ...")
        model = load_sft_model(ckpt_path).to(device)
        tokenizer = model.tokenizer
        tokenizer.add_tokens([SEP_TOKEN])
        sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

        attn, tokens = get_attention(model, tokenizer, sep_token_id, args.epitope, args.tcr, device)
        attentions_list.append((attn, tokens))
        model_names.append(name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not attentions_list:
        print("No models loaded — check checkpoint paths.")
        sys.exit(1)

    plot_all(attentions_list, model_names, args.epitope, args.tcr, args.saveto)


if __name__ == "__main__":
    main()
