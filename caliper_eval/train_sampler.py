# encoding: utf-8
"""Build identity-balanced batches and resample frames or tracklets each epoch."""
import copy
import random
from collections import defaultdict

from torch.utils.data import Sampler


class ResamplingIdentitySampler(Sampler):
    """Build identity-balanced batches with a fresh sample of frames from each tracklet every
    epoch."""

    def __init__(self, data_source, tracklet_ids, batch_size, num_instances, cap=32, seed=0):
        assert batch_size % num_instances == 0, \
            "batch_size {} must be divisible by num_instances {}".format(batch_size, num_instances)
        assert len(tracklet_ids) == len(data_source), "tracklet_ids must be parallel to data_source"
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.cap = cap
        self.seed = seed
        self.epoch = 0

        # pid -> tracklet -> [indices]
        self.by_pid = defaultdict(lambda: defaultdict(list))
        for idx, item in enumerate(data_source):
            self.by_pid[item[1]][tracklet_ids[idx]].append(idx)
        self.pids = sorted(self.by_pid)
        self._length = self._estimate()

    def _draw(self, rng):
        """This epoch's index pool: <=cap frames per tracklet, grouped by pid."""
        out = {}
        for pid, tks in self.by_pid.items():
            idxs = []
            for idxs_t in tks.values():
                if self.cap and len(idxs_t) > self.cap:
                    idxs.extend(rng.sample(idxs_t, self.cap))
                else:
                    idxs.extend(idxs_t)
            out[pid] = idxs
        return out

    def _estimate(self):
        pool = self._draw(random.Random(self.seed))
        return sum(max(len(v), self.num_instances) -
                   max(len(v), self.num_instances) % self.num_instances for v in pool.values())

    def _epoch_rng(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        return rng

    def __iter__(self):
        rng = self._epoch_rng()
        pool = self._draw(rng)

        # chunk each pid's draw into K-sized groups
        batches = defaultdict(list)
        for pid, idxs in pool.items():
            idxs = list(idxs)
            if len(idxs) < self.num_instances:
                # a 1-chip "tracklet" is a detection, not a track; repeat rather than drop the identity
                idxs = [idxs[rng.randrange(len(idxs))] for _ in range(self.num_instances)]
            rng.shuffle(idxs)
            for i in range(0, len(idxs) - len(idxs) % self.num_instances, self.num_instances):
                batches[pid].append(idxs[i:i + self.num_instances])

        avail = [p for p in self.pids if batches[p]]
        final = []
        while len(avail) >= self.num_pids_per_batch:
            for pid in rng.sample(avail, self.num_pids_per_batch):
                final.extend(batches[pid].pop(0))
                if not batches[pid]:
                    avail.remove(pid)
        self._length = len(final)
        return iter(final)

    def __len__(self):
        return self._length


class MixingIdentitySampler(ResamplingIdentitySampler):
    """Draw one weighted variant per group each epoch, then sample frames for identity-balanced
    batches."""

    def __init__(self, data_source, tracklet_ids, batch_size, num_instances, cap=32, seed=0,
                 variant_groups=()):
        self.variant_groups = [(list(g), list(w)) for g, w in variant_groups]
        known = set(tracklet_ids)
        for g, w in self.variant_groups:
            if len(g) != len(w):
                raise ValueError("variant group %r has %d ids but %d weights" % (g, len(g), len(w)))
            missing = [t for t in g if t not in known]
            if missing:
                raise ValueError("variant group references tracklets absent from the item list: %r"
                                 % (missing[:3],))
        # every tracklet that appears in SOME group is off unless that group selects it
        self._grouped = {t for g, _ in self.variant_groups for t in g}
        super(MixingIdentitySampler, self).__init__(
            data_source, tracklet_ids, batch_size, num_instances, cap=cap, seed=seed)

    def _select(self, rng):
        """This epoch's chosen tracklet from each group."""
        chosen = set()
        for g, w in self.variant_groups:
            total = float(sum(w))
            r = rng.random() * total
            acc = 0.0
            pick = g[-1]
            for tid, wi in zip(g, w):
                acc += wi
                if r < acc:
                    pick = tid
                    break
            chosen.add(pick)
        return chosen

    def _draw(self, rng):
        """<=cap frames per tracklet, restricted to this epoch's selected variants."""
        keep = self._select(rng)
        out = {}
        for pid, tks in self.by_pid.items():
            idxs = []
            for tid, idxs_t in tks.items():
                if tid in self._grouped and tid not in keep:
                    continue
                if self.cap and len(idxs_t) > self.cap:
                    idxs.extend(rng.sample(idxs_t, self.cap))
                else:
                    idxs.extend(idxs_t)
            out[pid] = idxs
        return out

    def epoch_composition(self, epoch=None):
        """Return the total and synthetic sample counts for the given epoch."""
        rng = random.Random(self.seed + (self.epoch if epoch is None else epoch))
        keep = self._select(rng)
        total = synth = 0
        for tks in self.by_pid.values():
            for tid, idxs_t in tks.items():
                if tid in self._grouped and tid not in keep:
                    continue
                n = min(len(idxs_t), self.cap) if self.cap else len(idxs_t)
                total += n
                if str(tid).startswith('sim:'):
                    synth += n
        return total, synth


class MixingTrackletSampler(Sampler):
    """Draw one weighted variant per group each epoch and build identity-balanced batches of
    tracklets."""

    def __init__(self, data_source, tracklet_ids, num_instances=4, seed=0, variant_groups=()):
        assert len(tracklet_ids) == len(data_source), "tracklet_ids must be parallel to data_source"
        self.data_source = data_source
        self.num_instances = num_instances
        self.seed = seed
        self.epoch = 0

        self.variant_groups = [(list(g), list(w)) for g, w in variant_groups]
        known = set(tracklet_ids)
        for g, w in self.variant_groups:
            if len(g) != len(w):
                raise ValueError("variant group %r has %d ids but %d weights" % (g, len(g), len(w)))
            missing = [t for t in g if t not in known]
            if missing:
                raise ValueError("variant group references tracklets absent from the item list: %r"
                                 % (missing[:3],))
        self._grouped = {t for g, _ in self.variant_groups for t in g}

        # pid -> [(tracklet_id, index)]
        self.by_pid = defaultdict(list)
        for idx, item in enumerate(data_source):
            self.by_pid[item[1]].append((tracklet_ids[idx], idx))
        self.pids = sorted(self.by_pid)
        self._length = self._estimate()

    # ---- variant selection (identical in spirit to MixingIdentitySampler._select) ---------------

    def _select(self, rng):
        """This epoch's chosen tracklet from each group."""
        chosen = set()
        for g, w in self.variant_groups:
            total = float(sum(w))
            r = rng.random() * total
            acc = 0.0
            pick = g[-1]
            for tid, wi in zip(g, w):
                acc += wi
                if r < acc:
                    pick = tid
                    break
            chosen.add(pick)
        return chosen

    def _pool(self, rng):
        """pid -> [indices] for this epoch, restricted to the selected variants."""
        keep = self._select(rng)
        out = {}
        for pid, entries in self.by_pid.items():
            out[pid] = [idx for tid, idx in entries
                        if tid not in self._grouped or tid in keep]
        return out

    def _estimate(self):
        pool = self._pool(random.Random(self.seed))
        n = 0
        for idxs in pool.values():
            k = max(len(idxs), self.num_instances)     # short identities are oversampled, not dropped
            n += k - k % self.num_instances
        return n

    def _epoch_rng(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        return rng

    def __iter__(self):
        rng = self._epoch_rng()
        pool = self._pool(rng)

        # chunk each identity's tracklets into K-sized groups -- upstream's logic, with a seeded RNG
        # instead of the global one so the draw is reproducible from --seed.
        batches = []
        for pid in self.pids:
            idxs = list(pool[pid])
            if not idxs:
                continue
            if len(idxs) < self.num_instances:
                idxs = [idxs[rng.randrange(len(idxs))] for _ in range(self.num_instances)]
            rng.shuffle(idxs)
            for i in range(0, len(idxs) - len(idxs) % self.num_instances, self.num_instances):
                batches.append(idxs[i:i + self.num_instances])
        rng.shuffle(batches)

        ret = [i for b in batches for i in b]
        self._length = len(ret)
        return iter(ret)

    def __len__(self):
        return self._length

    def epoch_composition(self, epoch=None):
        """Return the total and synthetic sample counts for the given epoch."""
        rng = random.Random(self.seed + (self.epoch if epoch is None else epoch))
        keep = self._select(rng)
        total = synth = 0
        for entries in self.by_pid.values():
            for tid, _idx in entries:
                if tid in self._grouped and tid not in keep:
                    continue
                total += 1
                if str(tid).startswith('sim:'):
                    synth += 1
        return total, synth
