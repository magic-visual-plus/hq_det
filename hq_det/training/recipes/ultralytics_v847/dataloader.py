# Adapted from Ultralytics 8.4.7, ultralytics/data/build.py.
# This file is distributed under the AGPL-3.0 license:
# https://ultralytics.com/license

import os
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import dataloader

from hq_det.training.interfaces import DataLoaderStrategy


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(dataloader.DataLoader):
    """Reuse worker processes while exposing one finite epoch per iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        self.iterator = self._get_iterator()

    def __del__(self):
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for worker in self.iterator._workers:
                if worker.is_alive():
                    worker.terminate()
            self.iterator._shutdown_workers()
        except Exception:
            pass


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class UltralyticsV847DataLoader(DataLoaderStrategy):
    def build(
        self,
        dataset,
        batch_size,
        workers,
        collate_fn,
        shuffle=True,
        sampler=None,
        rank=-1,
        drop_last=False,
        pin_memory=True,
    ):
        batch_size = min(int(batch_size), len(dataset))
        device_count = torch.cuda.device_count()
        workers = min(
            (os.cpu_count() or 1) // max(device_count, 1), int(workers)
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + int(rank))
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle) and sampler is None,
            num_workers=workers,
            sampler=sampler,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=device_count > 0 and bool(pin_memory),
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=bool(drop_last) and len(dataset) % batch_size != 0,
        )


__all__ = ["InfiniteDataLoader", "UltralyticsV847DataLoader", "seed_worker"]
