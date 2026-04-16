from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from byprot import utils
from byprot.datamodules import register_datamodule
from byprot.datamodules.dataset.tcr_conditional import (
    TCRConditionalDataset,
    setup_conditional_dataloader,
)

log = utils.get_logger(__name__)


@register_datamodule("tcr_conditional")
class TCRConditionalDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data-bin/tcr_sft",
        max_tokens: int = 4096,
        max_len: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = TCRConditionalDataset(
                data_dir=self.hparams.data_dir,
                split="train",
                max_len=self.hparams.max_len,
            )
            self.valid_dataset = TCRConditionalDataset(
                data_dir=self.hparams.data_dir,
                split="valid",
                max_len=self.hparams.max_len,
            )
        elif stage == "test" or stage == "predict":
            self.test_dataset = TCRConditionalDataset(
                data_dir=self.hparams.data_dir,
                split="test",
                max_len=self.hparams.max_len,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}.")
        self.stage = stage

    def train_dataloader(self):
        return setup_conditional_dataloader(
            self.train_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
        )

    def val_dataloader(self):
        return setup_conditional_dataloader(
            self.valid_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
        )

    def test_dataloader(self):
        return setup_conditional_dataloader(
            self.test_dataset,
            max_tokens=self.hparams.max_tokens,
            num_workers=self.hparams.num_workers,
            max_len=self.hparams.max_len,
        )
