from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import glob
import os
import numpy as np
from rich.progress import Progress
from multiprocessing import Pool, Manager
from itertools import repeat
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from miditok import MIDITokenizer
from miditoolkit import MidiFile
from src.data_preprocess.data_preprocess import guitarset_preprocess
from shutil import rmtree
from src.datamodules.components import  CustomDataset, CustomPadCollate


class GuitarsetDataModule(LightningDataModule):
    """Custom datamodule for GuitarSet dataset.

    Args:
        data_preprocess_cfg: config file object for data preprocessing
        tokenizer: MIDITokenizer class
        vocab_size: vocabulary size of tokenizer
        data_dir: base data directory
        train_val_split_ratio: ratio for splitting train/valid in dev data
        test_player_n: test player number for testing
        batch_size: batch size
        normalize_cqt: whether to normalize the cqt 
        cache_dataset: whether to cache the whole dataset during first epoch
            might not work well with multiple dataloader workers
        num_workers: number of workers for parallel data preprocessing
        dataloader_workers: number of workers for dataloader
        pin_memory: whether to pin memory or not
        persistent_workers: whether to make the workers persistent
        preprocess_on_training_start: if set to False, it will bypass the preprocessing and use the existing preprocessed data
    """
    def __init__(
        self,
        data_preprocess_cfg: any,
        tokenizer: MIDITokenizer,
        vocab_size: int,
        data_dir: str = "data/",
        train_val_split_ratio: float = 0.9,
        test_player_n: int = 5,
        batch_size: int = 16,
        normalize_cqt: bool = False,
        cache_dataset: bool = False,
        num_workers: int = 10,
        dataloader_workers: int = 5,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        preprocess_on_training_start: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.audio_filename_list = glob.glob(
            os.path.join(
                data_preprocess_cfg.data_dir, "guitarset", "audio_mono-mic", "*.wav"
            )
        )

    def prepare_data(self):
        """Make dirs, preprocess and split the dataset
        """
        output_dir = os.path.join(self.hparams.data_preprocess_cfg.output_dir, "guitarset")
        cqt_dir = os.path.join(
            self.hparams.data_preprocess_cfg.output_dir, "guitarset", "cqt"
        )
        midi_dir = os.path.join(
            self.hparams.data_preprocess_cfg.output_dir, "guitarset", "midi"
        )

        if self.hparams.preprocess_on_training_start:
            # make dirs for data preprocessing
            if os.path.exists(output_dir):
                rmtree(output_dir)
                os.makedirs(cqt_dir)
                os.makedirs(midi_dir)
            else:
                os.makedirs(cqt_dir)
                os.makedirs(midi_dir)

            # preprocess the data using multiprocessing for parallel computation
            pool = Pool(processes=self.hparams.num_workers)
            with Progress() as p:
                task = p.add_task(
                    "Preprocessing...", total=len(self.audio_filename_list)
                )
                for _ in pool.imap_unordered(
                    guitarset_preprocess,
                    zip(
                        self.audio_filename_list,
                        repeat(self.hparams.data_preprocess_cfg),
                    ),
                ):
                    p.update(task, advance=1)

            pool.close()

        # split the dataset to train/valid/test set
        split_trackname_list = glob.glob(os.path.join(cqt_dir, "*.npy"))
        split_trackname_list = [
            os.path.split(path)[1][:-4] for path in split_trackname_list
        ]
        dev_data_list = [
            dataname
            for dataname in split_trackname_list
            if not dataname.startswith(f"0{self.hparams.test_player_n}_")
        ]
        self.train_data_list = dev_data_list[
            : int(round(len(dev_data_list) * self.hparams.train_val_split_ratio))
        ]
        self.val_data_list = dev_data_list[
            int(round(len(dev_data_list) * self.hparams.train_val_split_ratio)) :
        ]
        self.test_data_list = [
            dataname
            for dataname in split_trackname_list
            if dataname.startswith(f"0{self.hparams.test_player_n}_")
        ]

    def setup(self):
        """Initialize train, validation and test dataset
        """
        self.data_train = CustomDataset(
            self.train_data_list,
            self.hparams.data_preprocess_cfg.output_dir + "guitarset",
            self.hparams.tokenizer,
            self.hparams.mode,
            self.hparams.cache_dataset,
        )
        self.data_val = CustomDataset(
            self.val_data_list,
            self.hparams.data_preprocess_cfg.output_dir + "guitarset",
            self.hparams.tokenizer,
            self.hparams.mode,
            self.hparams.cache_dataset,
        )
        self.data_test = CustomDataset(
            self.test_data_list,
            self.hparams.data_preprocess_cfg.output_dir + "guitarset",
            self.hparams.tokenizer,
            self.hparams.mode,
            self.hparams.cache_dataset,
        )

    def train_dataloader(self):
        """Initialize the train dataloader
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomPadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Initialize the validataion dataloader
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomPadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Initialize the test dataloader
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomPadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or tes."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint"""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint"""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "guitarset.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
