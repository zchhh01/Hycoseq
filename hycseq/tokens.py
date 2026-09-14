from dataclasses import dataclass
from typing import Optional

import numpy as np


BASES = ("A", "C", "T", "G")
BASE_INDEX = {base: index for index, base in enumerate(BASES)}


@dataclass(frozen=True)
class TokenLayout:
    input_width: int
    sequence_steps: int
    vocabulary_size: Optional[int]
    unknown_id: Optional[int]
    padding_id: Optional[int]


def kmer_steps(max_bases, kmer, stride):
    if max_bases < kmer:
        raise ValueError("max_bases must be at least kmer")
    return ((max_bases - kmer) // stride) + 1


def real_kmer_steps(length, kmer, stride):
    if length < kmer:
        return 0
    return ((length - kmer) // stride) + 1


def resolve_token_layout(representation, max_bases, kmer, stride, token_width):
    if max_bases < 1 or kmer < 1 or stride < 1 or token_width < 1:
        raise ValueError("token layout dimensions must be positive")
    if representation == "base":
        return TokenLayout(5, max_bases, None, None, None)
    steps = kmer_steps(max_bases, kmer, stride)
    unknown_id = 4 ** kmer
    if representation == "kmer_onehot":
        return TokenLayout(unknown_id + 1, steps, unknown_id + 1, unknown_id, None)
    if representation == "kmer_embed":
        return TokenLayout(token_width, steps, unknown_id + 2, unknown_id, unknown_id + 1)
    raise ValueError("unsupported representation: " + representation)


def kmer_id(fragment, kmer):
    if len(fragment) != kmer:
        raise ValueError("invalid k-mer width")
    value = 0
    for base in fragment:
        if base not in BASE_INDEX:
            return 4 ** kmer
        value = value * 4 + BASE_INDEX[base]
    return value


def encode_bases(sequence, max_bases):
    sequence = (sequence or "").upper()
    if len(sequence) > max_bases:
        raise ValueError("sequence exceeds max_bases")
    lookup = {"A": 0, "C": 1, "T": 2, "G": 3, "N": 4}
    encoded = np.zeros((5, max_bases), dtype=np.float32)
    if sequence:
        rows = [lookup.get(base, 4) for base in sequence]
        encoded[rows, np.arange(len(sequence))] = 1.0
    return encoded


def encode_kmer_onehot(sequence, max_bases, kmer, stride):
    sequence = (sequence or "").upper()
    if len(sequence) > max_bases:
        raise ValueError("sequence exceeds max_bases")
    steps = kmer_steps(max_bases, kmer, stride)
    encoded = np.zeros(((4 ** kmer) + 1, steps), dtype=np.float32)
    for position in range(real_kmer_steps(len(sequence), kmer, stride)):
        start = position * stride
        encoded[kmer_id(sequence[start:start + kmer], kmer), position] = 1.0
    return encoded


def encode_kmer_tokens(sequence, max_bases, kmer, stride):
    sequence = (sequence or "").upper()
    if len(sequence) > max_bases:
        raise ValueError("sequence exceeds max_bases")
    steps = kmer_steps(max_bases, kmer, stride)
    encoded = np.full((steps,), (4 ** kmer) + 1, dtype=np.int64)
    for position in range(real_kmer_steps(len(sequence), kmer, stride)):
        start = position * stride
        encoded[position] = kmer_id(sequence[start:start + kmer], kmer)
    return encoded
