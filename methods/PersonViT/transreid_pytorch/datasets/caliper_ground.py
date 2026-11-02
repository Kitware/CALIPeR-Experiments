# encoding: utf-8
"""Load CALIPeR ground-field training and validation data for PersonViT."""
import os
import os.path as osp
import sys

try:
    from caliper_eval.train_corpus import GroundFieldCorpus
except ImportError:
    # Fall back to a sibling checkout so this works without installing the package.
    sys.path.insert(0, osp.expanduser(os.environ.get('CALIPER_EVAL_HOME', osp.abspath(osp.join(osp.dirname(__file__), '../../../../')))))
    from caliper_eval.train_corpus import GroundFieldCorpus


class CaliperGround(object):
    """Adapt the ground-field corpus and its held-out splits to the method’s data loader."""

    def __init__(self, root='', verbose=True, holdout=0, holdout_rule='random', seed=1234,
                 chip_root=None, corpus=None, sim_mode=None, sim_p=0.5, sim_chip_root=None,
                 sim_severities=None, **kwargs):
        kw = {}
        if chip_root:
            kw['chip_root'] = chip_root
        if corpus:
            kw['corpus'] = corpus
        if sim_chip_root:
            kw['sim_chip_root'] = sim_chip_root
        # Pass the selected strengths through to the corpus.
        if sim_severities:
            kw['sim_severities'] = sim_severities
        self.corpus = GroundFieldCorpus(holdout=holdout, holdout_rule=holdout_rule, seed=seed,
                                        sim_mode=sim_mode, sim_p=sim_p, **kw)
        # [(tracklet_ids, weights)] -- one member of each group is served per epoch. Empty
        # unless sim_mode is set, which is what decides whether the mixing sampler is needed.
        self.variant_groups = self.corpus.variant_groups()

        items, tracklets = self.corpus.image_items_with_tracklets()
        self.train = [(p, pid, camid, 0) for (p, pid, camid, _clothes) in items]
        self.train_tracklets = tracklets
        # clothes ids are carried for CAL; unused here but kept so one adapter serves every method
        self.train_clothes = [c for (_p, _pid, _cam, c) in items]

        self.query, self.gallery = [], []
        if holdout:
            g, q = self.corpus.val_split()
            # val_split returns tracklets; flatten to per-image rows for this image-based repo
            self.gallery = [(p, cid, cam, 0) for (paths, cid, cam, _t) in g for p in paths]
            self.query = [(p, cid, cam, 0) for (paths, cid, cam, _t) in q for p in paths]

        self.num_train_pids = len({t[1] for t in self.train})
        self.num_train_cams = len({t[2] for t in self.train})
        self.num_train_vids = 1
        self.num_query_pids = len({t[1] for t in self.query})
        self.num_gallery_pids = len({t[1] for t in self.gallery})

        if verbose:
            print("=> CALIPeR ground-field training corpus loaded")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(
                self.num_train_pids, len(self.train), self.num_train_cams))
            for name, sub in (("query", self.query), ("gallery", self.gallery)):
                if sub:
                    print("  {:8s} | {:5d} | {:8d} | {:9d}".format(
                        name, len({t[1] for t in sub}), len(sub), len({t[2] for t in sub})))
                else:
                    print("  {:8s} |     - |        - |         -   (no holdout)".format(name))
            print("  ----------------------------------------")
            print("  tracklets: {}".format(len(set(self.train_tracklets))))
