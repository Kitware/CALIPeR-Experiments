# encoding: utf-8
"""Load CALIPeR ground-field training and validation data for BoT."""
import os
import os.path as osp
import sys

from .bases import BaseImageDataset

try:
    from caliper_eval.train_corpus import GroundFieldCorpus
except ImportError:
    # Fall back to the repository root so this works without installing the package.
    # Override with CALIPER_EVAL_HOME if caliper_eval lives elsewhere.
    _REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..', '..'))
    sys.path.insert(0, os.environ.get('CALIPER_EVAL_HOME', _REPO_ROOT))
    from caliper_eval.train_corpus import GroundFieldCorpus


class CaliperGround(BaseImageDataset):
    """Adapt the ground-field corpus and its held-out splits to the method’s data loader."""

    dataset_dir = 'caliper_ground'

    def __init__(self, root='', verbose=True, holdout=0, holdout_rule='random', seed=1234,
                 chip_root=None, corpus=None, sim_mode=None, sim_p=0.5,
                 sim_chip_root=None, sim_severities=None, **kwargs):
        super(CaliperGround, self).__init__()
        kw = {}
        if chip_root:
            kw['chip_root'] = chip_root
        if corpus:
            kw['corpus'] = corpus
        if sim_chip_root:
            kw['sim_chip_root'] = sim_chip_root
        self.corpus = GroundFieldCorpus(holdout=holdout, holdout_rule=holdout_rule, seed=seed,
                                        sim_mode=sim_mode, sim_p=sim_p,
                                        sim_severities=sim_severities, **kw)
        # [(tracklet_ids, weights)] -- one member of each group is served per epoch. Empty
        # unless sim_mode is set, which is what decides whether the mixing sampler is needed.
        self.variant_groups = self.corpus.variant_groups()

        items, tracklets = self.corpus.image_items_with_tracklets()
        self.train = [(p, pid, camid) for (p, pid, camid, _clothes) in items]
        self.train_tracklets = tracklets
        # clothes ids are carried for CAL; unused here but kept so one adapter serves every method
        self.train_clothes = [c for (_p, _pid, _cam, c) in items]

        self.query, self.gallery = [], []
        if holdout:
            g, q = self.corpus.val_split()
            # val_split returns tracklets; flatten to per-image rows for this image-based repo
            self.gallery = [(p, cid, cam) for (paths, cid, cam, _t) in g for p in paths]
            self.query = [(p, cid, cam) for (paths, cid, cam, _t) in q for p in paths]

        if verbose:
            print("=> CALIPeR ground-field training corpus loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = \
            self.get_imagedata_info(self.query) if self.query else (0, 0, 0)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = \
            self.get_imagedata_info(self.gallery) if self.gallery else (0, 0, 0)

    def print_dataset_statistics(self, train, query, gallery):
        n_tr_pid, n_tr_img, n_tr_cam = self.get_imagedata_info(train)
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(n_tr_pid, n_tr_img, n_tr_cam))
        for name, sub in (("query", query), ("gallery", gallery)):
            if sub:
                a, b, c = self.get_imagedata_info(sub)
                print("  {:8s} | {:5d} | {:8d} | {:9d}".format(name, a, b, c))
            else:
                print("  {:8s} |     - |        - |         -   (no holdout)".format(name))
        print("  ----------------------------------------")
        print("  tracklets: {}".format(len(set(self.train_tracklets))))
        if self.corpus.sim_mode:
            print("  sim_mode: {} (p={})   variant groups: {}".format(
                self.corpus.sim_mode, self.corpus.sim_p, len(self.variant_groups)))
            print("  range composition -- subjects / tracklets / chips / capped-img"
                  " (ALL variants, so an upper bound; the sampler reports the realised epoch):")
            for tok, v in sorted(self.corpus.range_composition().items(),
                                 key=lambda kv: kv[0].rjust(6)):
                print("    {:>6s}  {:4d} / {:5d} / {:8d} / {:7d}".format(tok, *v))
