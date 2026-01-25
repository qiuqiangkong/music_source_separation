import torch
from torch import Tensor

from mss.augmentations.torch.eq import BatchRandomEQ


class RandomEQ(BatchRandomEQ):
    def __init__(self, 
        min_db=-6.0,
        max_db=6.0,
        n_bands=8
    ):
        super().__init__(min_db, max_db, n_bands)

    def __call__(self, x: Tensor) -> Tensor:
        r"""Batch EQ.

        c: channels_num
        l: audio_samples

        Args:
            x: (c, l)

        Returns:
            x: (c, l)
        """
        return super().__call__(torch.from_numpy(x)[None, :, :])[0].numpy()