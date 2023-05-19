import torch
import os
import numpy as np
from rich.progress import Progress
from torch.utils.data import Dataset
from miditoolkit import MidiFile
import sys


class CustomPadCollate:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, batch):
        cqt_max_len = 0
        token_max_len = 0
        for _, cqt, token, _ in batch:
            if cqt_max_len < cqt.shape[0]:
                cqt_max_len = cqt.shape[0]
            if token_max_len < token.shape[0]:
                token_max_len = token.shape[0]

        padded_cqt = torch.zeros((len(batch), cqt_max_len, batch[0][1].shape[1]))
        padded_token = torch.zeros((len(batch), token_max_len), dtype=torch.long)
        cqt_lens = torch.zeros((len(batch)), dtype=torch.long)
        token_lens = torch.zeros((len(batch)), dtype=torch.long)
        tempos = torch.zeros((len(batch)))
        track_name_list = []
        for i, (track_name, cqt, token, tempo) in enumerate(batch):
            track_name_list += [track_name]
            padded_cqt[i, : len(cqt)] = cqt
            padded_token[i, : len(token)] = token
            cqt_lens[i] = len(cqt)
            token_lens[i] = len(token)
            tempos[i] = tempo

        out = {
            "track_name_list": track_name_list,
            "padded_cqt": padded_cqt,
            "cqt_lens": cqt_lens,
            "tempos": tempos,
            "padded_tokens_gt": padded_token,
            "token_lens_gt": token_lens,
        }
        return out


class CustomDataset(Dataset):
    def __init__(self, track_name_list, data_dir, tokenizer, cache_dataset):
        self.track_name_list = track_name_list
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.cache_dataset = cache_dataset
        self.vocab = tokenizer._create_vocabulary()
        self.eos_idx = self.vocab.__getitem__("EOS_None")
        self.cqt_dir = os.path.join(self.data_dir, "cqt")
        self.midi_dir = os.path.join(self.data_dir, "midi")

        self.check_valid(track_name_list)

        if self.cache_dataset:
            self.manager = Manager()
            self.caches = self.manager.list()
            for _ in range(len(self.track_name_list)):
                self.caches += [None]

    def check_valid(self, track_name_list):
        """Filter out empty MIDI files and MIDI files that have too long sequence length when converted into tokens."""
        with Progress() as p:
            task = p.add_task(
                "Dataset: Checking validity...", total=len(track_name_list)
            )
            invalid_idx_list = []
            for idx, track_name in enumerate(track_name_list):
                cqt = np.load(os.path.join(self.cqt_dir, f"{track_name}.npy"))
                midi = MidiFile(os.path.join(self.midi_dir, f"{track_name}.mid"))
                tokens = self.tokenizer(midi)
                if len(tokens) < 1:
                    print(f"Empty MIDI file: {track_name}. removing from the dataset")
                    invalid_idx_list += [idx]
                elif len(cqt) <= len(tokens[0]):
                    print(
                        f"Token length > CQT length: {track_name}. removing from the dataset"
                    )
                    invalid_idx_list += [idx]
                p.update(task, advance=1)

            if invalid_idx_list:
                invalid_idx_list.reverse()
                for invalid_idx in invalid_idx_list:
                    del self.track_name_list[invalid_idx]

    def __len__(self):
        return len(self.track_name_list)

    def __getitem__(self, idx):
        if self.cache_dataset and self.caches[idx] is not None:
            return self.caches[idx]

        cqt_path = os.path.join(self.cqt_dir, f"{self.track_name_list[idx]}.npy")
        midi_path = os.path.join(self.midi_dir, f"{self.track_name_list[idx]}.mid")
        cqt = np.load(cqt_path)
        midi = MidiFile(midi_path)
        tokens = self.tokenizer(midi)[0]
        tokens += [self.eos_idx]

        tempo = midi.tempo_changes[0].tempo

        sample = (
            self.track_name_list[idx],
            torch.from_numpy(cqt),
            torch.tensor(tokens, dtype=torch.long),
            tempo,
        )

        if self.cache_dataset:
            self.caches[idx] = sample

        return sample
