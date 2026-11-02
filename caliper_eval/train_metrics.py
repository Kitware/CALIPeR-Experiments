# encoding: utf-8
"""Score held-out queries by camera range and record validation metrics for each epoch."""
import json
import os.path as osp

# camid -> range token. Set by caliper_eval.train_corpus.CAMID; duplicated as a plain dict so this module
# stays importable without it.
CAMID_NAMES = {0: '10m', 1: '160m', 2: '250m', 3: '400m', 4: '500m', 5: '1000m'}


def make_by_range_metric(base_cls, eval_func, output_dir, eval_period=1, logger=None,
                         filename='val_metrics.jsonl', return_super=False):
    """Build a validation metric that records overall and per-range retrieval scores."""
    import numpy as np
    import torch

    path = osp.join(output_dir, filename)

    class R1_mAP_ByRange(base_cls):
        def __init__(self, *a, **kw):
            super(R1_mAP_ByRange, self).__init__(*a, **kw)
            # Track validation epochs separately from the trainer.
            self._n_calls = 0

        def compute(self):
            passthrough = super(R1_mAP_ByRange, self).compute() if return_super else None
            self._n_calls += 1
            epoch = self._n_calls * eval_period

            feats = torch.cat(self.feats, dim=0)
            # forks differ: reid-strong-baseline stores the string 'yes', PersonViT a bool/str it
            # tests for truthiness. Normalise unless it is explicitly off.
            if self.feat_norm not in (False, None, '', 'no'):
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            qf, gf = feats[:self.num_query], feats[self.num_query:]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])

            m, n = qf.shape[0], gf.shape[0]
            distmat = (torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t())
            distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
            distmat = distmat.cpu().numpy()

            subsets = [('all', np.ones(m, dtype=bool))]
            for cam in sorted(set(q_camids.tolist())):
                subsets.append((CAMID_NAMES.get(cam, 'cam%s' % cam), q_camids == cam))

            out = {}
            for name, mask in subsets:
                if not mask.any():
                    continue
                # The evaluation function may return additional values after CMC and mAP.
                res = eval_func(distmat[mask], q_pids[mask], g_pids, q_camids[mask], g_camids)
                cmc, mAP = res[0], res[1]
                out[name] = (cmc, mAP)
                # Count queries with a valid gallery match after camera exclusion.
                valid = int(((g_pids[None, :] == q_pids[mask][:, None]) &
                             (g_camids[None, :] != q_camids[mask][:, None])).any(axis=1).sum())
                row = dict(epoch=epoch, subset=name, mAP=float(mAP),
                           R1=float(cmc[0]), R5=float(cmc[4]) if len(cmc) > 4 else None,
                           R10=float(cmc[9]) if len(cmc) > 9 else None,
                           n_query=int(mask.sum()), n_gallery=int(n), n_valid_query=valid)
                with open(path, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                if logger is not None:
                    logger.info("  val[{:>5}] epoch {:3d}  mAP {:.1%}  R1 {:.1%}  "
                                "({} queries, {} with a match)"
                                .format(name, epoch, mAP, cmc[0], int(mask.sum()), valid))
            return passthrough if return_super else out['all']

    return R1_mAP_ByRange
