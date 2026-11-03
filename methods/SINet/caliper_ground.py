# encoding: utf-8
"""Load CALIPeR ground-field training and validation data for SINet."""
import os
import os.path as osp
import sys

try:
    from caliper_eval.train_corpus import GroundFieldCorpus
except ImportError:
    _REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.environ.get('CALIPER_EVAL_HOME', _REPO_ROOT))
    from caliper_eval.train_corpus import GroundFieldCorpus


class CaliperGround(object):
    """Adapt the ground-field corpus and its held-out splits to the method’s data loader."""

    def __init__(self, root=None, verbose=True, holdout=0, holdout_rule='random', seed=1234,
                 chip_root=None, corpus=None, sim_mode=None, sim_p=0.5, sim_chip_root=None,
                 sim_severities=None, **kwargs):
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
        # [(tracklet_ids, weights)] -- one member of each group is served per epoch, which is
        # what MixingTrackletSampler is for. Empty unless sim_mode is set.
        self.variant_groups = self.corpus.variant_groups()

        # Keep the source tracklet IDs alongside the loader items for variant sampling.
        items, self.train_tracklets = self.corpus.tracklets_with_ids()
        self.train = [(paths, pid, camid) for (paths, pid, camid, _t) in items]

        self.query, self.gallery = [], []
        if holdout:
            g, q = self.corpus.val_split()
            # Map canonical validation subjects to the numeric labels expected by SINet.
            self.val_label = {c: i for i, c in enumerate(sorted({t[1] for t in g} | {t[1] for t in q}))}
            self.gallery = [(paths, self.val_label[cid], cam) for (paths, cid, cam, _t) in g]
            self.query = [(paths, self.val_label[cid], cam) for (paths, cid, cam, _t) in q]

        self.num_train_pids = len({t[1] for t in self.train})
        self.num_train_cams = len({t[2] for t in self.train})
        self.num_train_vids = 1
        self.num_query_pids = len({t[1] for t in self.query})
        self.num_gallery_pids = len({t[1] for t in self.gallery})

        if verbose:
            n_img = sum(len(t[0]) for t in self.train)
            print("=> CALIPeR ground-field training corpus loaded (video / tracklets)")
            print("  ------------------------------------------")
            print("  subset   | # ids | # tracklets | # images")
            print("  ------------------------------------------")
            print("  train    | {:5d} | {:11d} | {:8d}".format(
                self.num_train_pids, len(self.train), n_img))
            for name, sub in (("query", self.query), ("gallery", self.gallery)):
                if sub:
                    print("  {:8s} | {:5d} | {:11d} | {:8d}".format(
                        name, len({t[1] for t in sub}), len(sub),
                        sum(len(t[0]) for t in sub)))
                else:
                    print("  {:8s} |     - |           - |        -   (no holdout)".format(name))
            print("  ------------------------------------------")
