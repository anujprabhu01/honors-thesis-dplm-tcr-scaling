#!/usr/bin/env python
"""Conditional epitope -> TCR generation using a fine-tuned DPLM.

Given an epitope sequence, generates TCR CDR3 beta chains by running
discrete diffusion on the masked TCR region while keeping the epitope fixed.

Usage:
    python scripts/scaling/generate_conditional.py \
        --checkpoint logs/tcr_sft_0.1m/checkpoints/last.ckpt \
        --epitopes GILGFVFTL NLVPMVATV \
        --tcr_len 15 \
        --num_seqs 10 \
        --max_iter 500 \
        --saveto results/conditional_gen
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

# Set PROJECT_ROOT for Hydra config resolution (normally set by train.py)
os.environ.setdefault("PROJECT_ROOT", os.getcwd())

import torch
from transformers import EsmTokenizer

from byprot import utils
from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
from byprot.utils.config import load_yaml_config


SEP_TOKEN = "<sep>"


def load_sft_model(checkpoint_path):
    """Load an SFT model from a training checkpoint."""
    # Load config from the training run
    cfg_path = Path(checkpoint_path).parents[1] / ".hydra" / "config.yaml"
    cfg = load_yaml_config(str(cfg_path)).model
    cfg.net.pretrain = False
    cfg.net.sft_from_checkpoint = None  # Don't re-load; we load the full SFT checkpoint
    cfg.pop("_target_")

    model = DiffusionProteinLanguageModel(cfg)

    # Load state dict
    ckpt = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    new_state = OrderedDict()
    for k, v in ckpt.items():
        new_state[k[6:]] = v  # strip "model." prefix (loading into DiffusionProteinLanguageModel)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def build_input(epitope, tcr_len, tokenizer, sep_token_id, num_seqs, device):
    """Build input tokens: <cls> EPITOPE <sep> <mask>*tcr_len <eos>."""
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    epi_ids = tokenizer.encode(epitope, add_special_tokens=False)
    input_ids = [cls_id] + epi_ids + [sep_token_id] + [mask_id] * tcr_len + [eos_id]
    input_ids = torch.tensor([input_ids] * num_seqs, dtype=torch.long, device=device)

    # partial_masks: True for positions that should NOT change during generation
    # (cls, epitope, sep are fixed; TCR masks + eos can change)
    n_fixed = 1 + len(epi_ids) + 1  # cls + epitope + sep
    partial_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    partial_mask[:, :n_fixed] = True  # fix the epitope portion

    return input_ids, partial_mask


def main():
    parser = argparse.ArgumentParser(description="Conditional epitope -> TCR generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--epitopes", nargs="+", type=str, required=True, help="Epitope sequences")
    parser.add_argument("--tcr_len", type=int, default=15, help="Length of TCR to generate")
    parser.add_argument("--num_seqs", type=int, default=10, help="Number of sequences per epitope")
    parser.add_argument("--max_iter", type=int, default=500, help="Diffusion decoding iterations")
    parser.add_argument("--sampling_strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--saveto", type=str, default="results/conditional_gen")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_sft_model(args.checkpoint)
    tokenizer = model.tokenizer

    # Add <sep> token to tokenizer (must match training)
    tokenizer.add_tokens([SEP_TOKEN])
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f"<sep> token ID: {sep_token_id}")

    model = model.eval().cuda()
    device = next(model.parameters()).device

    os.makedirs(args.saveto, exist_ok=True)

    for epitope in args.epitopes:
        print(f"\nGenerating TCRs for epitope: {epitope}")
        input_tokens, partial_masks = build_input(
            epitope, args.tcr_len, tokenizer, sep_token_id, args.num_seqs, device
        )

        with torch.cuda.amp.autocast():
            output_tokens = model.generate(
                input_tokens=input_tokens,
                tokenizer=tokenizer,
                max_iter=args.max_iter,
                sampling_strategy=args.sampling_strategy,
                partial_masks=partial_masks,
            )

        # Decode and extract TCR portion
        results = []
        for seq_tokens in output_tokens:
            decoded = tokenizer.decode(seq_tokens, skip_special_tokens=False)
            tokens = decoded.split()
            # Find <sep> position and extract TCR after it
            if SEP_TOKEN in tokens:
                sep_idx = tokens.index(SEP_TOKEN)
                tcr_tokens = tokens[sep_idx + 1:]
                # Remove special tokens from TCR portion
                tcr_tokens = [t for t in tcr_tokens if t not in
                              [tokenizer.cls_token, tokenizer.eos_token,
                               tokenizer.pad_token, tokenizer.mask_token]]
                tcr_seq = "".join(tcr_tokens)
            else:
                # Fallback: decode whole thing without special tokens
                tcr_seq = "".join(tokenizer.decode(seq_tokens, skip_special_tokens=True).split())
            results.append(tcr_seq)

        print(f"Generated {len(results)} TCR sequences:")
        for i, seq in enumerate(results):
            print(f"  {i}: {seq}")

        # Save to FASTA
        saveto_name = os.path.join(args.saveto, f"epitope_{epitope}.fasta")
        with open(saveto_name, "w") as f:
            for idx, seq in enumerate(results):
                f.write(f">EPITOPE={epitope}|SEQ_{idx}|L={len(seq)}\n")
                f.write(f"{seq}\n")
        print(f"Saved to {saveto_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
