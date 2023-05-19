from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import random
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
from src.data_preprocess.data_preprocess import (
    classic_guitar_preprocess,
    check_midi_validity,
)
from shutil import rmtree
from src.datamodules.components import CustomDataset, CustomPadCollate


class ClassicGuitarDataModule(LightningDataModule):
    """Custom datamodule for classic guitar dataset.

    Args:
        data_preprocess_cfg: config file object for data preprocessing
        tokenizer: MIDITokenizer class
        vocab_size: vocabulary size of tokenizer
        data_dir: base data directory
        train_ratio: ratio of training data from the whole dataset
        valid_ratio: ratio of validataion data from the whole dataset
        test_player_n: dummy argument. it should always be "classic_guitar"
        batch_size: batch size
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
        train_ratio: float = 0.9,
        valid_ratio: float = 0.05,
        test_player_n="classic_guitar",
        batch_size: int = 16,
        cache_dataset: bool = False,
        num_workers: int = 10,
        dataloader_workers: int = 5,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        preprocess_on_training_start: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        classclef_filename_list = glob.glob(
            os.path.join(
                data_preprocess_cfg.data_dir,
                "classic_guitar_midi",
                "guitar_classclef",
                "*.mid",
            )
        )
        midifiles_filename_list = glob.glob(
            os.path.join(
                data_preprocess_cfg.data_dir,
                "classic_guitar_midi",
                "midifiles",
                "*.mid",
            )
        )
        self.midi_filename_list = midifiles_filename_list + classclef_filename_list

    def prepare_data(self):
        """Make dirs, preprocess and split the dataset"""
        output_dir = os.path.join(
            self.hparams.data_preprocess_cfg.output_dir, "classic_guitar"
        )
        wav_dir = os.path.join(output_dir, "wav")
        cqt_dir = os.path.join(output_dir, "cqt")
        midi_dir = os.path.join(output_dir, "midi")

        if self.hparams.preprocess_on_training_start:
            # make dirs for data preprocessing
            if os.path.exists(output_dir):
                rmtree(output_dir)
                os.makedirs(wav_dir)
                os.makedirs(cqt_dir)
                os.makedirs(midi_dir)
            else:
                os.makedirs(wav_dir)
                os.makedirs(cqt_dir)
                os.makedirs(midi_dir)

            self.valid_midi_filename_list = []
            self.total_dataset_time = 0

            # check validity of MIDI file
            pool = Pool(processes=self.hparams.num_workers)
            with Progress() as p:
                task = p.add_task(
                    "Checking validity...", total=len(self.midi_filename_list)
                )
                for valid_midi_filename, validity, time in pool.imap_unordered(
                    check_midi_validity,
                    self.midi_filename_list,
                ):
                    if validity:
                        self.valid_midi_filename_list.append(valid_midi_filename)
                        self.total_dataset_time += time
                    p.update(task, advance=1)
            pool.close()
            print(
                f"Using {len(self.valid_midi_filename_list)} valid MIDI files out of {len(self.midi_filename_list)} total MIDI files."
            )
            print(f"{int(self.total_dataset_time)} mins in total.")

            # preprocess the data using multiprocessing for parallel computation
            pool = Pool(processes=self.hparams.num_workers)
            with Progress() as p:
                task = p.add_task(
                    "Preprocessing...", total=len(self.valid_midi_filename_list)
                )
                for _ in pool.imap_unordered(
                    classic_guitar_preprocess,
                    zip(
                        self.valid_midi_filename_list,
                        repeat(self.hparams.data_preprocess_cfg),
                    ),
                ):
                    p.update(task, advance=1)

            pool.close()

        # split the dataset to train/valid/test set
        split_trackname_list = glob.glob(os.path.join(cqt_dir, "*.npy"))
        glob_in = os.path.join(cqt_dir, "*.npy")
        split_trackname_list = [
            os.path.split(path)[1][:-4] for path in split_trackname_list
        ]
        random.shuffle(split_trackname_list)
        train_valid_test_ratio = [self.hparams.train_ratio, self.hparams.valid_ratio]
        cumsum_ratio = np.cumsum(train_valid_test_ratio)
        self.train_data_list = split_trackname_list[
            : int(round(len(split_trackname_list) * cumsum_ratio[0]))
        ]
        self.val_data_list = split_trackname_list[
            int(round(len(split_trackname_list) * cumsum_ratio[0])) : int(
                round(len(split_trackname_list) * cumsum_ratio[1])
            )
        ]
        self.test_data_list = split_trackname_list[
            int(round(len(split_trackname_list) * cumsum_ratio[1])) :
        ]

    def setup(self, stage=None):
        """Initialize train, validation and test dataset"""
        self.data_train = CustomDataset(
            self.train_data_list,
            self.hparams.data_preprocess_cfg.output_dir + "classic_guitar",
            self.hparams.tokenizer,
            self.hparams.cache_dataset,
        )
        self.data_val = CustomDataset(
            self.val_data_list,
            self.hparams.data_preprocess_cfg.output_dir + "classic_guitar",
            self.hparams.tokenizer,
            self.hparams.cache_dataset,
        )
        self.data_test = CustomDataset(
            self.test_data_list,
            self.hparams.data_preprocess_cfg.output_dir + "classic_guitar",
            self.hparams.tokenizer,
            self.hparams.cache_dataset,
        )

    def train_dataloader(self):
        """Initialize the train dataloader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomPadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Initialize the validataion dataloader"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomPadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Initialize the test dataloader"""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=CustomPadCollate(self.hparams.vocab_size),
            num_workers=self.hparams.dataloader_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test"""
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
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "datamodule" / "classic_guitar.yaml"
    )
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
