import random


class RandomGain:
    
    def __init__(self, min_db=-12, max_db=6):
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, x):

        db = random.uniform(self.min_db, self.max_db)
        gain = 10 ** (db / 20.)
        x *= gain

        return x