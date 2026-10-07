import math

import torch
from torch.utils.data import Sampler


class LengthGroupedSampler(Sampler):
    """Length-grouped sampler with optional easy -> medium -> hard curriculum."""

    def __init__(
        self,
        global_batch_size,
        lengths,
        difficulties=None,
        generator=None,
        curriculum=True,
    ):
        self.global_batch_size = global_batch_size
        self.lengths = lengths
        self.difficulties = difficulties
        self.generator = generator
        self.curriculum = curriculum

    def _length_grouped_shuffle(self, indices):
        """Shuffle indices, then group by length within mega-batches."""
        perm = torch.randperm(len(indices), generator=self.generator).tolist()
        shuffled = [indices[i] for i in perm]
        mega = self.global_batch_size * 50
        result = []
        for start in range(0, len(shuffled), mega):
            window = shuffled[start : start + mega]
            window.sort(key=lambda i: self.lengths[i], reverse=True)
            result.extend(window)
        return result

    def __iter__(self):
        if self.difficulties is not None and self.curriculum:
            groups = {"easy": [], "medium": [], "hard": []}
            for i, diff in enumerate(self.difficulties):
                groups.get(diff, groups["medium"]).append(i)
            result = []
            for diff in ["easy", "medium", "hard"]:
                if groups[diff]:
                    result.extend(self._length_grouped_shuffle(groups[diff]))
        else:
            result = self._length_grouped_shuffle(list(range(len(self.lengths))))
        if len(result) % self.global_batch_size != 0:
            pad = self.global_batch_size - (len(result) % self.global_batch_size)
            result.extend(result[:pad])
        return iter(result)

    def __len__(self):
        return (
            math.ceil(len(self.lengths) / self.global_batch_size)
            * self.global_batch_size
        )
