"""Genomic feature extraction using BioPython and BLAST."""
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
from Bio import SeqIO

from src.config import K_MER_SIZES


def _generate_kmer_alphabet(k: int) -> List[str]:
    return ["".join(p) for p in product("ACGT", repeat=k)]


def _read_fasta(path: Path) -> tuple[str, List[int]]:
    sequences = []
    lengths = []
    for record in SeqIO.parse(str(path), "fasta"):
        sequences.append(str(record.seq).upper())
        lengths.append(len(record.seq))
    return "".join(sequences), lengths


def _filter_valid_nucleotides(sequence: str) -> str:
    return "".join(c for c in sequence if c in "ACGT")


def _compute_composition(sequence: str) -> Dict[str, float]:
    g, c, a, t = (sequence.count(n) for n in "GCAT")
    total = g + c + a + t

    return {
        "gc_content": (g + c) / total if total else 0.0,
        "gc_skew": (g - c) / (g + c + 1e-9),
        "at_skew": (a - t) / (a + t + 1e-9),
    }


def _compute_kmer_frequencies(sequence: str, k: int) -> Dict[str, float]:
    alphabet = _generate_kmer_alphabet(k)
    counts = Counter(sequence[i:i + k] for i in range(len(sequence) - k + 1))
    total = sum(counts.values()) + 1e-9

    return {f"k{k}_{kmer}": counts[kmer] / total for kmer in alphabet}


def extract_compositional_features(fasta_path: Path) -> Dict[str, float]:
    """Extract GC content, k-mer frequencies, and sequence stats from a FASTA file."""
    full_sequence, contig_lengths = _read_fasta(fasta_path)
    valid_sequence = _filter_valid_nucleotides(full_sequence)

    features = {
        "total_length": len(full_sequence),
        "valid_length": len(valid_sequence),
        "n_contigs": len(contig_lengths),
        "avg_contig_length": np.mean(contig_lengths) if contig_lengths else 0,
        "max_contig_length": max(contig_lengths) if contig_lengths else 0,
    }
    features.update(_compute_composition(valid_sequence))

    for k in K_MER_SIZES:
        features.update(_compute_kmer_frequencies(valid_sequence, k))

    return features