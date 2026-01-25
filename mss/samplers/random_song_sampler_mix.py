import random
from torch.utils.data import Dataset
import numpy as np


class RandomSongSamplerMix:
    def __init__(self, dataset: Dataset, max_intra_source_mix):
        r"""Randomly sample indexes of different stems of a dataset without 
        replacement. Execute this process infinitely.
        """

        self.dataset = dataset
        self.stems = dataset.stems
        self.mix_num = max_intra_source_mix

        self.indices = {stem: self.random_permutation(len(self.dataset), self.mix_num) for stem in self.stems}
        # E.g., {"bg": [3, 7, 0, ...], "target":, [4, 1, 9, ...]}

        self.ptrs = {stem: 0 for stem in self.indices.keys()}  # pointers

    def __iter__(self) -> dict:
        r"""Yiled an index_dict."""

        while True:

            out = {}

            for stem in self.indices.keys():

                # Reshuffle indices. Reset pointer.
                if self.ptrs[stem] == len(self.indices[stem]):
                    self.indices[stem] = self.random_permutation(len(self.dataset), self.mix_num)
                    self.ptrs[stem] = 0

                out[stem] = self.indices[stem][self.ptrs[stem]]
                self.ptrs[stem] += 1
            
            yield out  # E.g., {"vocals": [94, 13], "drums": [13, 26], "other": [0, 22], "vocals": [6, 88]}

    def random_permutation(self, n: int, mix_num: int) -> np.ndarray:

        indices = np.zeros((n, mix_num), dtype=np.int64)
        for m in range(mix_num):
            x = list(range(n))
            random.shuffle(x)
            indices[:, m] = x
        
        return indices