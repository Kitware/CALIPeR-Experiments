# encoding: utf-8
"""Copy ground-field training crops to local storage for faster loading."""
import argparse
import os
import os.path as osp
import subprocess
import sys
import tempfile

from caliper_eval.train_corpus import GroundFieldCorpus, CHIP_ROOT, SIM_CHIP_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dest', default='/dev/shm/reid-chips')
    ap.add_argument('--src', default=CHIP_ROOT)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--sim', action='store_true',
                    help="stage the SYNTHETIC chips instead of the real ones (all three turbulence "
                         "strengths, ~1.5 GiB). They live under a different root, so they are a "
                         "second, separate staging pass -- run this once with and once without.")
    args = ap.parse_args()

    if args.sim:
        if args.src == CHIP_ROOT:
            args.src = SIM_CHIP_ROOT
        if args.dest == '/dev/shm/reid-chips':
            args.dest = '/dev/shm/reid-chips-sim'
        c = GroundFieldCorpus(sim_mode='add', sim_chip_root=args.src)
    else:
        c = GroundFieldCorpus(chip_root=args.src)
    # copy in tracklet order: chips of one tracklet sit in one directory, so this reads the HDD in
    # something close to inode order instead of seeking per file
    rels = []
    want_sim = bool(args.sim)
    for t in sorted(c._tk.values(), key=lambda x: str(x['tid'])):
        if bool(str(t['tid']).startswith('sim:')) != want_sim:
            continue
        rels.extend(t['paths'])
    # sample RANDOMLY, not the first N: rels is in tracklet order, so a head slice is one or two
    # tracklets and badly misestimates the mean chip size (it under-read by ~2x in practice).
    import random as _r
    probe = _r.Random(0).sample(rels, min(2000, len(rels)))
    est = sum(os.path.getsize(osp.join(args.src, r)) for r in probe) / len(probe) * len(rels) / 2**30
    print("staging {:,} chips (~{:.1f} GiB)\n  from {}\n  to   {}".format(
        len(rels), est, args.src, args.dest))
    if args.dry_run:
        return

    free = os.statvfs(osp.dirname(args.dest) or '/')
    avail = free.f_bavail * free.f_frsize / 2**30
    if avail < est * 1.1:
        sys.exit("ABORT: {:.1f} GiB free at {}, need ~{:.1f} GiB".format(avail, args.dest, est))

    # group-writable + setgid, so whoever stages it first, the rest of the group can top it up or
    # rebuild it after a reboot. Without this the tree is owned r-x by the group and only the original
    # stager can re-run.
    os.makedirs(args.dest, exist_ok=True)
    try:
        os.chmod(args.dest, 0o2775)
    except OSError:
        pass
    with tempfile.NamedTemporaryFile('w', suffix='.lst', delete=False) as f:
        f.write("\n".join(rels) + "\n")
        lst = f.name
    try:
        # --files-from recreates the directory tree; -a preserves times so rsync can skip on re-run
        cmd = ['rsync', '-a', '--chmod=Dg+ws,Fg+w', '--info=progress2', '--files-from=' + lst,
               args.src.rstrip('/') + '/', args.dest.rstrip('/') + '/']
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit("rsync failed with {}".format(rc))
    finally:
        os.unlink(lst)

    missing = [r for r in rels[::500] if not osp.exists(osp.join(args.dest, r))]
    print("\nspot-check {} of {} chips: {} missing".format(len(rels[::500]), len(rels), len(missing)))
    if missing:
        sys.exit("ABORT: staged tree is incomplete")
    print("done. train with --chip-root {}".format(args.dest))


if __name__ == '__main__':
    main()
