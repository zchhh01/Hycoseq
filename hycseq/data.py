import csv
from pathlib import Path
from random import random

import torch
from torch.utils.data import DataLoader, Dataset

from .tokens import encode_bases, encode_kmer_onehot, encode_kmer_tokens, real_kmer_steps


COMPLEMENT = str.maketrans("ACTGactg", "TGACtgac")


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


class SequenceTable(Dataset):
    def __init__(self, path, settings, augment=False):
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise ValueError("dataset must contain a header and at least one row: " + str(path))
        self.sequences = [row[0] for row in rows[1:]]
        self.labels = [int(row[1]) for row in rows[1:]]
        self.settings = settings
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        if self.augment and random() > 0.5:
            sequence = reverse_complement(sequence)
        tensor = self._encode(sequence)
        length = self._effective_length(sequence)
        return tensor, torch.tensor(self.labels[index], dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def _effective_length(self, sequence):
        if len(sequence) > self.settings.max_bases:
            raise ValueError("sequence exceeds max_bases")
        if self.settings.representation == "base":
            return max(1, len(sequence))
        return max(1, real_kmer_steps(len(sequence), self.settings.kmer, self.settings.token_stride))

    def _encode(self, sequence):
        if self.settings.representation == "base":
            result = encode_bases(sequence, self.settings.max_bases)
            return torch.from_numpy(result)
        if self.settings.representation == "kmer_onehot":
            result = encode_kmer_onehot(sequence, self.settings.max_bases, self.settings.kmer, self.settings.token_stride)
            return torch.from_numpy(result)
        result = encode_kmer_tokens(sequence, self.settings.max_bases, self.settings.kmer, self.settings.token_stride)
        return torch.from_numpy(result).long()


def split_paths(settings):
    root = Path(settings.data_root)
    if settings.corpus == "TEB":
        paths = (
            root / ("train_" + settings.task + ".csv"),
            root / ("valid_" + settings.task + ".csv"),
            root / ("test_" + settings.task + ".csv"),
        )
    else:
        task_root = root / settings.task
        paths = (task_root / "train.csv", task_root / "dev.csv", task_root / "test.csv")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing dataset files:\n" + "\n".join(missing))
    return paths


def build_loaders(settings):
    train_path, valid_path, test_path = split_paths(settings)
    train_set = SequenceTable(train_path, settings, augment=settings.reverse_complement)
    valid_set = SequenceTable(valid_path, settings)
    test_set = SequenceTable(test_path, settings)
    return (
        DataLoader(train_set, batch_size=settings.batch, shuffle=True),
        DataLoader(valid_set, batch_size=settings.batch, shuffle=False),
        DataLoader(test_set, batch_size=settings.batch, shuffle=False),
    )
