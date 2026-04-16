import math
import os

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import EsmTokenizer

from byprot import utils
from byprot.datamodules.dataset.uniref_hf import (
    ApproxBatchSampler,
    SortishSampler,
)

log = utils.get_logger(__name__)


class TCRConditionalDataset(Dataset):
    """Dataset for epitope-TCR conditional generation.

    Loads HuggingFace dataset with columns: epitope, tcr, length.
    """

    def __init__(self, data_dir: str, split: str, max_len=64):
        self.data_dir = data_dir
        self.split = split
        self.max_len = max_len
        self.data = load_from_disk(data_dir)[split]
        log.info(f"TCRConditionalDataset [{split}]: {len(self.data)} pairs")

    def __len__(self):
        return len(self.data)

    def get_metadata_lens(self):
        return self.data["length"]

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        return {"epitope": item["epitope"], "tcr": item["tcr"]}


class DPLMCollaterConditional:
    """Collater for epitope-TCR conditional training.

    Builds sequences: <cls> EPITOPE <sep> TCR <eos>
    and a condition_mask marking epitope positions (never masked during diffusion).
    """

    SEP_TOKEN = "<sep>"

    def __init__(self, tokenizer_path=None):
        if tokenizer_path is None:
            self.tokenizer = EsmTokenizer.from_pretrained(
                "facebook/esm2_t30_150M_UR50D"
            )
        else:
            self.tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)

        # Add separator token
        num_added = self.tokenizer.add_tokens([self.SEP_TOKEN])
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.SEP_TOKEN)
        log.info(f"Added {num_added} token(s). <sep> ID = {self.sep_token_id}")

    def __call__(self, samples):
        # samples is a list of {"epitope": str, "tcr": str}
        all_input_ids = []
        all_condition_masks = []

        for sample in samples:
            epitope = sample["epitope"]
            tcr = sample["tcr"]

            # Tokenize epitope and TCR separately (no special tokens)
            epi_ids = self.tokenizer.encode(epitope, add_special_tokens=False)
            tcr_ids = self.tokenizer.encode(tcr, add_special_tokens=False)

            # Build: <cls> EPITOPE <sep> TCR <eos>
            cls_id = self.tokenizer.cls_token_id  # 2
            eos_id = self.tokenizer.eos_token_id  # 3
            input_ids = [cls_id] + epi_ids + [self.sep_token_id] + tcr_ids + [eos_id]

            # condition_mask: True for positions that should NOT be masked
            # (epitope + sep + cls are the condition; TCR + eos are the target)
            n_condition = 1 + len(epi_ids) + 1  # cls + epitope + sep
            n_target = len(tcr_ids) + 1  # tcr + eos
            condition_mask = [True] * n_condition + [False] * n_target

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_condition_masks.append(torch.tensor(condition_mask, dtype=torch.bool))

        # Pad to longest in batch
        max_len = max(ids.size(0) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id  # 1

        padded_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(samples), max_len, dtype=torch.bool)
        padded_condition = torch.zeros(len(samples), max_len, dtype=torch.bool)

        for i, (ids, cmask) in enumerate(zip(all_input_ids, all_condition_masks)):
            length = ids.size(0)
            padded_ids[i, :length] = ids
            attention_mask[i, :length] = True
            padded_condition[i, :length] = cmask

        return {
            "input_ids": padded_ids,
            "input_mask": attention_mask,
            "targets": padded_ids.clone(),
            "condition_mask": padded_condition,
        }


def setup_conditional_dataloader(
    ds: TCRConditionalDataset,
    max_tokens=4096,
    bucket_size=1000,
    max_batch_size=800,
    num_workers=4,
    rank=0,
    world_size=1,
    max_len=64,
) -> DataLoader:
    collater = DPLMCollaterConditional()
    lens = ds.get_metadata_lens()
    train_sortish_sampler = SortishSampler(
        lens, bucket_size, num_replicas=world_size, rank=rank
    )
    train_sampler = ApproxBatchSampler(
        train_sortish_sampler,
        max_tokens,
        max_batch_size,
        lens,
        max_len=max_len,
    )
    dl = DataLoader(
        dataset=ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collater,
    )
    return dl
