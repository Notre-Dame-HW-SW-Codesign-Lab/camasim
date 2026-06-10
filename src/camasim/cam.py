"""
CAMASim - Content-Addressable Memory Simulation Framework
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from cam.config import CAMConfig
from cam.evacam import EVACAMConfig
from cam._merge import horizontal_and, vertical_union
from cam._mapping import default_write_mapping, default_query_mapping


@dataclass
class WriteResult:
    latency: Optional[float]  # ns
    energy: Optional[float]   # J


@dataclass
class QueryResult:
    indices: Optional[list]
    latency: Optional[float]  # ps
    energy: Optional[float]   # J


class CAMASim:
    # TODO: I need to make sure the config is possible
    # TODO: quantization needs to be addressed, we need to make sure the sizes are doable (fefet might be able to do 3bit)
    def __init__(self, config: CAMConfig):
        self.config = config

        self.num_banks = config.num_banks
        self.num_mats = config.num_mats
        self.num_subarrays = config.num_subarrays
        self.subarray_rows = config.subarray_rows
        self.subarray_cols = config.subarray_cols

        self.cam = np.zeros((self.num_banks, self.num_mats, self.num_subarrays, self.subarray_rows, self.subarray_cols))
        self.evacam = EVACAMConfig()
        self.col_splits = 1

        # Pluggable strategies
        self.horizontal_merge: Callable = horizontal_and
        self.vertical_merge: Callable = vertical_union
        self.write_mapping: Callable = default_write_mapping
        self.query_mapping: Callable = default_query_mapping

    # TODO: I need to have quantization
    # TODO: I need to validate the input sizes with the data we need good errors
    def write(self, data: np.ndarray) -> WriteResult:
        N, D = data.shape
        total_rows = self.num_banks * self.num_mats * self.num_subarrays * self.subarray_rows
        self.col_splits = math.ceil(D / self.subarray_cols)

        padded = self.write_mapping(data, total_rows, self.col_splits, self.subarray_cols)
        padded, energy, latency = self.evacam.write(padded)

        self.cam = padded.reshape(
            self.num_banks, self.num_mats, self.num_subarrays,
            self.subarray_rows, self.col_splits, self.subarray_cols
        )

        return WriteResult(latency=latency, energy=energy)
    
    # TODO: I need to add selective matching both on columns and rows
    # TODO: I need to have quantization
    # TODO: I need to make sure the accumulation of energy/latency is correct
    def query(self, queries: np.ndarray) -> QueryResult:
        N, D = queries.shape
        total_energy = 0.0
        total_latency = 0.0

        total_rows = self.num_banks * self.num_mats * self.num_subarrays * self.subarray_rows
        rows = self.cam.reshape(total_rows, self.col_splits, self.subarray_cols)
        q_splits = self.query_mapping(queries, self.col_splits, self.subarray_cols)

        all_matches = []
        for q in q_splits:
            row_matches = []
            for row_idx, row in enumerate(rows):
                col_results = [self.evacam.match(row[c], q[c]) for c in range(self.col_splits)]
                matched, energy, latency = self.horizontal_merge(col_results)
                total_energy += energy
                total_latency += latency
                if matched:
                    row_matches.append(row_idx)
            all_matches.append(row_matches)

        indices = self.vertical_merge(all_matches)
        return QueryResult(indices=indices, latency=total_latency, energy=total_energy)
