# encoding: utf-8
"""Pre-decoded, pre-resized chip cache -- removes ~59% of dataloader CPU work.

Decode and resize can dominate CPU data-loading cost. Example per-operation profile:

    Resize bicubic -> 256x128   35%
    JPEG decode                 24%     <- these two are identical every epoch
    ToTensor                    23%
    Normalize                   10%
    Pad + RandomCrop             8%

Decode and resize are pure repeated work: the same JPEG through the same decoder gives the same
pixels every time, and the same resize gives the same output. Doing them once and reusing the result
removes 59% of the cost, for ~25 GB of tmpfs.

DOES THIS CHANGE RESULTS? No, provided two rules hold, and both are enforced here:

  1. The cache is built with EXACTLY the transform it replaces -- `T.Resize([256,128],
     interpolation=3)`, bicubic, via the same torchvision call the training pipeline uses. Same
     input, same operation, same output.
  2. The caller must DELETE `Resize` from the train transform when reading the cache, not leave it
     to no-op on an already-256x128 image. PIL does not guarantee a same-size resize is a
     pass-through, so leaving it in would be an unaudited second resample.

Deliberately NOT cached:
  * the VAL path. Val uses `T.Resize(SIZE_TEST)` with no interpolation argument, i.e. BILINEAR, not
    the train path's bicubic. One cache cannot serve both, and silently switching val to bicubic
    would move the val numbers.
  * X-TFCLIP, which trains at 288x144 rather than 256x128. It keeps the JPEG path, or gets its own
    cache built at its own size.

TMPFS IS SWAP-BACKED. A cache in /dev/shm can be paged out under memory pressure, which would undo
the point of it. Size the cache against the available memory budget.

Build:
    python -m caliper_eval.chip_cache --out /dev/shm/reid-chips-256x128 --workers 2
Read:
    cache = ChipCache('/dev/shm/reid-chips-256x128')
    arr = cache.get(abs_or_rel_chip_path)     # (256, 128, 3) uint8, ready for Image.fromarray
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

DEFAULT_SIZE = (256, 128)          # (h, w), matching cfg.INPUT.SIZE_TRAIN for bot/personvit/clip-reid
DEFAULT_OUT = '/dev/shm/reid-chips-256x128'
DATA = 'chips.u8'
INDEX = 'index.json'


class ChipCache(object):
    """Read-only view of a built cache. Cheap to construct; the memmap is lazy."""

    def __init__(self, path=DEFAULT_OUT):
        self.path = path
        with open(osp.join(path, INDEX)) as f:
            meta = json.load(f)
        self.size = tuple(meta['size'])
        self.interpolation = meta['interpolation']
        self.chip_root = meta['chip_root']
        self.index = meta['index']            # rel chip_path -> row
        self.n = meta['n']
        h, w = self.size
        self._arr = np.memmap(osp.join(path, DATA), dtype=np.uint8, mode='r',
                              shape=(self.n, h, w, 3))

    def rel(self, p):
        """Accept either an absolute path under the original chip root, or a relative one."""
        if p in self.index:
            return p
        r = osp.relpath(p, self.chip_root)
        if r in self.index:
            return r
        # a caller may have staged chips elsewhere (e.g. /dev/shm) -- fall back on the tail that
        # matches how the manifest stores paths, i.e. everything from the version dir onward
        parts = p.split(os.sep)
        for i, seg in enumerate(parts):
            cand = os.sep.join(parts[i:])
            if cand in self.index:
                return cand
        raise KeyError("chip not in cache: {}".format(p))

    def get(self, p):
        return self._arr[self.index[self.rel(p)]]

    def __len__(self):
        return self.n


def strip_resize(transform):
    """Return `transform` with its Resize removed -- the transform a cache reader must use.

    Derived, never hand-written. The cache stores already-resized pixels, so the resize has to go;
    but writing out a second copy of the transform list by hand would mean the verifier and the
    training code could silently disagree, and a bit-identity test between two copies of the same
    mistake proves nothing. Both callers pass the repo's REAL transform through this function, so
    they cannot drift.

    Asserts exactly one Resize was found: zero means the caller passed something unexpected, more
    than one means the pipeline is doing something this cache does not model.
    """
    import torchvision.transforms as T
    ops = list(transform.transforms)
    resizes = [i for i, op in enumerate(ops) if isinstance(op, T.Resize)]
    if len(resizes) != 1:
        raise SystemExit(
            "strip_resize: expected exactly 1 Resize in the transform, found {}. Pipeline: {}"
            .format(len(resizes), [type(o).__name__ for o in ops]))
    if resizes[0] != 0:
        raise SystemExit(
            "strip_resize: Resize is at position {}, not first. Removing it would change what the "
            "preceding ops see. Pipeline: {}"
            .format(resizes[0], [type(o).__name__ for o in ops]))
    return T.Compose(ops[1:])


class CachedImageDataset(object):
    """Drop-in for a repo's ImageDataset that reads pixels from the cache instead of decoding JPEG.

    Emits the 5-tuple PersonViT's and CLIP-ReID's collate functions expect:
    (img, pid, camid, trackid, img_path). reid-strong-baseline's ImageDataset unpacks 3 and resizes
    with BILINEAR, so it is deliberately NOT supported here -- see verify_cache.py.

    The transform passed in MUST come from strip_resize(), not from a hand-written copy.
    """

    def __init__(self, dataset, cache, transform=None):
        from PIL import Image
        self._Image = Image
        self.dataset = dataset
        self.cache = cache
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = self._Image.fromarray(np.asarray(self.cache.get(img_path)))
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, trackid, img_path


def _worker(args):
    out, size, interpolation, n, lo, hi, rels, root = args
    from PIL import Image
    import torchvision.transforms as T
    resize = T.Resize(list(size), interpolation=interpolation)
    h, w = size
    arr = np.memmap(osp.join(out, DATA), dtype=np.uint8, mode='r+', shape=(n, h, w, 3))
    bad = []
    for i in range(lo, hi):
        try:
            im = Image.open(osp.join(root, rels[i])).convert('RGB')
            arr[i] = np.asarray(resize(im), dtype=np.uint8)
        except Exception as e:                       # a truncated chip should not kill the build
            bad.append((rels[i], repr(e)))
    arr.flush()
    del arr
    return lo, hi, bad


def build(out=DEFAULT_OUT, size=DEFAULT_SIZE, interpolation=3, chip_root=None, workers=2):
    """Decode+resize every chip in the ground-field corpus once into a flat uint8 memmap."""
    import multiprocessing as mp
    from .train_corpus import GroundFieldCorpus

    kw = {'chip_root': chip_root} if chip_root else {}
    corpus = GroundFieldCorpus(**kw)           # no holdout: the cache covers the whole corpus
    root = corpus.chip_root
    items, _ = corpus.image_items_with_tracklets()
    rels = sorted({osp.relpath(p, root) for (p, _pid, _cam, _cl) in items})
    n = len(rels)
    h, w = size
    nbytes = n * h * w * 3

    os.makedirs(out, exist_ok=True)
    print("chips: {:,}   size {}x{}   -> {:.1f} GB at {}".format(n, h, w, nbytes / 2 ** 30, out))
    # preallocate so workers can write disjoint slices of one file
    with open(osp.join(out, DATA), 'wb') as f:
        f.truncate(nbytes)

    chunk = (n + workers - 1) // workers
    jobs = [(out, size, interpolation, n, i, min(i + chunk, n), rels, root)
            for i in range(0, n, chunk)]
    bad_all = []
    if workers > 1:
        with mp.Pool(workers) as pool:
            for lo, hi, bad in pool.imap_unordered(_worker, jobs):
                bad_all += bad
                print("  rows {:,}-{:,} done{}".format(lo, hi, "  ({} failed)".format(len(bad))
                                                       if bad else ""))
    else:
        for j in jobs:
            lo, hi, bad = _worker(j)
            bad_all += bad
            print("  rows {:,}-{:,} done".format(lo, hi))

    with open(osp.join(out, INDEX), 'w') as f:
        json.dump({'size': list(size), 'interpolation': interpolation, 'chip_root': root,
                   'n': n, 'index': {r: i for i, r in enumerate(rels)},
                   'failed': bad_all}, f)
    print("wrote index; {} chips failed to decode".format(len(bad_all)))
    for r, e in bad_all[:5]:
        print("  FAILED {}: {}".format(r, e))
    return out


def verify(out=DEFAULT_OUT, chip_root=None, n=200, seed=0):
    """Re-decode a random sample and assert the cache is bit-identical to the live transform."""
    import random
    from PIL import Image
    import torchvision.transforms as T
    cache = ChipCache(out)
    resize = T.Resize(list(cache.size), interpolation=cache.interpolation)
    root = chip_root or cache.chip_root
    rels = random.Random(seed).sample(list(cache.index), min(n, len(cache.index)))
    mism = 0
    for r in rels:
        live = np.asarray(resize(Image.open(osp.join(root, r)).convert('RGB')), dtype=np.uint8)
        if not np.array_equal(live, cache.get(r)):
            mism += 1
    print("verified {} chips: {} mismatched".format(len(rels), mism))
    return mism == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--chip-root', default='/dev/shm/reid-chips')
    ap.add_argument('--size', nargs=2, type=int, default=list(DEFAULT_SIZE), metavar=('H', 'W'))
    ap.add_argument('--interpolation', type=int, default=3, help='3 = bicubic, matching T.Resize')
    ap.add_argument('--workers', type=int, default=2,
                    help='keep low while training jobs are running -- this competes for the same '
                         'six physical cores they are already saturating')
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    if not a.verify_only:
        build(a.out, tuple(a.size), a.interpolation, a.chip_root, a.workers)
    ok = verify(a.out, a.chip_root)
    raise SystemExit(0 if ok else 1)
