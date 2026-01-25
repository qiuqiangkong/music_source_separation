import random
from torch import Tensor


class RandomGain:
    r"""Applies random gain to the audio."""
    def __init__(self, min_db=-6.0, max_db=6.0):
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, x: Tensor) -> Tensor:
        r"""Random gain.

        c: audio_channels
        l: audio_samples

        Args:
            x: (c, l)

        Output:
            out: (c, l)
        """
        db = random.uniform(self.min_db, self.max_db)
        gain = db_to_scale(db)
        out = gain * x
        # print("db:", db)
        return out


def scale_to_db(scale):
    db = 20 * np.log10(scale)
    return db

def db_to_scale(db):
    scale = 10 ** (db / 20.)
    return scale