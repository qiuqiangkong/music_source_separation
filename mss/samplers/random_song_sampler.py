import random
from torch.utils.data import Dataset


class RandomSongSampler:
    def __init__(self, dataset: Dataset):
        r"""Randomly sample indexes of different stems of a dataset without 
        replacement. Execute this process infinitely.
        """

        self.dataset = dataset
        self.stems = dataset.stems
        self.bg_stems = dataset.bg_stems
        self.target_stems = dataset.target_stems
        self.time_align = dataset.time_align
        
        self.idxes = self.get_indexes_dict()
        # E.g., {"bg": [3, 7, 0, ...], "target":, [4, 1, 9, ...]}
        
        self.ps = {key: 0 for key in self.idxes.keys()}  # pointers

    def __iter__(self) -> dict:
        r"""Yiled an index_dict."""

        while True:

            buffer = {}

            for key in self.idxes.keys():

                if self.ps[key] == len(self.idxes[key]):
                    # Traversed all data of a key. Reshuffle indexes. Reset pointer.
                    random.shuffle(self.idxes[key])
                    self.ps[key] = 0

                buffer[key] = self.idxes[key][self.ps[key]]
                self.ps[key] += 1

            index_dict = self.buffer_dict_to_indexes_dict(buffer)
            # E.g., {"vocals": 94, "drums": 94, "other": 35, "vocals": 35}
            
            yield index_dict

    def get_indexes_dict(self) -> dict:

        N = len(self.dataset)

        if self.time_align == "strict":
            idxes = {name: self.random_permutation(N) for name in ["song"]}

        elif self.time_align == "group":
            idxes = {name: self.random_permutation(N) for name in ["target", "bg"]}

        elif self.time_align == "random":
            idxes = {name: self.random_permutation(N) for name in self.stems}

        else:
            raise ValueError(self.time_align)

        return idxes

    def random_permutation(self, n) -> list:
        x = list(range(n))
        random.shuffle(x)
        return x

    def buffer_dict_to_indexes_dict(self, buffer: dict) -> dict:

        if self.time_align == "strict":
            idx_dict = {stem: buffer["song"] for stem in self.stems}

        elif self.time_align == "group":
            tar_idx_dict = {stem: buffer["target"] for stem in self.target_stems}
            bg_idx_dict = {stem: buffer["bg"] for stem in self.bg_stems}
            idx_dict = {**tar_idx_dict, **bg_idx_dict}  # Merge dict

        elif self.time_align == "random":
            idx_dict = buffer

        else:
            raise ValueError(self.time_align)

        return idx_dict