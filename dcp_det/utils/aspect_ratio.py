"""Aspect-ratio grouped sampling used by the formal training setup."""

import bisect
import math
from collections import defaultdict
from itertools import chain, repeat

import numpy as np
from torch.utils.data.sampler import BatchSampler, Sampler


def _repeat_to_at_least(values, count):
    repeated = chain.from_iterable(repeat(values, math.ceil(count / len(values))))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """Yield fixed-size batches whose samples share an aspect-ratio group."""

    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler must be a torch Sampler")
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = int(batch_size)

    def __iter__(self):
        buffers = defaultdict(list)
        samples = defaultdict(list)
        yielded = 0
        for index in self.sampler:
            group_id = self.group_ids[index]
            buffers[group_id].append(index)
            samples[group_id].append(index)
            if len(buffers[group_id]) == self.batch_size:
                yield buffers[group_id]
                yielded += 1
                del buffers[group_id]

        remaining_batches = len(self) - yielded
        for group_id, values in sorted(
            buffers.items(), key=lambda item: len(item[1]), reverse=True
        ):
            if remaining_batches == 0:
                break
            needed = self.batch_size - len(values)
            values.extend(_repeat_to_at_least(samples[group_id], needed)[:needed])
            yield values
            remaining_batches -= 1
        if remaining_batches != 0:
            raise RuntimeError("Could not form all grouped batches")

    def __len__(self):
        return len(self.sampler) // self.batch_size


def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = []
    for index in range(len(dataset)):
        height, width = dataset.get_height_and_width(index)
        aspect_ratios.append(float(width) / float(height))
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = [bisect.bisect_right(bins, ratio) for ratio in aspect_ratios]
    counts = np.unique(groups, return_counts=True)[1]
    print("Aspect-ratio bins: {}".format([0] + bins + [np.inf]))
    print("Images per bin: {}".format(counts.tolist()))
    return groups
