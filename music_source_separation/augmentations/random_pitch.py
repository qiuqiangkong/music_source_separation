import random
import librosa


class RandomPitch:
    
    def __init__(self, sr, min_pitch=-3, max_pitch=3):
        self.sr = sr
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def __call__(self, x):

        pitch = random.uniform(self.min_pitch, self.max_pitch)
        x = librosa.effects.pitch_shift(x, sr=self.sr, n_steps=pitch)

        return x