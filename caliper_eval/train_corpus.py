# encoding: utf-8
"""Read ground-field training crops and group them by subject and tracklet."""
import csv
import json
import os.path as osp
import random

CORPUS = './data/splits/combined_brc1_brc2_train_corpus'
CHIP_ROOT = './data/reid-chips'
MANIFEST = 'train_manifest.csv'

# Synthetic crops share the source tracklet and frame identifiers of the real 400m crops.
SIM_CORPUS = './data/splits/combined_brc1_brc2_sim_train_corpus'
SIM_CHIP_ROOT = './data/reid-chips-sim'
SIM_MANIFEST = 'sim_train_manifest.csv'
SIM_SEVERITIES = ('cn2_1e-13', 'cn2_4e-13', 'cn2_1e-12')   # weak / medium / strong
SIM_TOK = '1000m'      # what the synthetic chips are LABELLED as simulating

# Select how real 400m tracklets and their synthetic variants enter each epoch.
SIM_MODES = ('add', 'dup', 'mix', 'swapall')
# range token -> dense camid. CALIPeR's analogue of a camera: what SIE camera embeddings key on
# (X-TFCLIP sets SIE_CAMERA: True) and what camera-aware triplet/eval logic can use.
TOKENS = ['10m', '160m', '250m', '400m', '500m', '1000m']
CAMID = {t: i for i, t in enumerate(TOKENS)}
NORMALIZE = {'260m': '250m', '380m': '400m'}   # native path tokens -> the normalized names


# Select validation identities by range coverage and the number of long-range samples.
HOLDOUT_RULES = {
    'random':       (('400m', '500m', '1000m'), 'random'),
    'thin':         (('400m', '500m', '1000m'), 'fewest_1000m'),
    'thin-relaxed': (('400m', '1000m'),         'fewest_1000m'),
}


class GroundFieldCorpus(object):
    """Read labelled ground-field crops, select held-out identities, and group real or synthetic
    training examples."""

    def __init__(self, corpus=CORPUS, chip_root=CHIP_ROOT, holdout=0, seed=1234, manifest=MANIFEST,
                 holdout_rule='random', sim_mode=None, sim_p=0.5, sim_corpus=SIM_CORPUS,
                 sim_chip_root=SIM_CHIP_ROOT, sim_severities=None):
        self.chip_root = chip_root
        self.sim_chip_root = sim_chip_root
        self.sim_mode = sim_mode
        self.sim_p = sim_p
        # Use all available turbulence strengths by default.
        self.sim_severities = tuple(sim_severities) if sim_severities else SIM_SEVERITIES
        unknown = [s for s in self.sim_severities if s not in SIM_SEVERITIES]
        if unknown:
            raise ValueError("unknown sim severity %r; expected a subset of %s"
                             % (unknown, list(SIM_SEVERITIES)))
        if sim_mode is not None and sim_mode not in SIM_MODES:
            raise ValueError("unknown sim_mode %r; expected one of %s" % (sim_mode, list(SIM_MODES)))
        tk = {}
        with open(osp.join(corpus, manifest)) as f:
            for r in csv.DictReader(f):
                if r['kind'] != 'field' or r['cls'] != 'ground':
                    continue
                t = tk.get(r['tracklet_id'])
                if t is None:
                    tok = NORMALIZE.get(r['tok'], r['tok'])
                    t = tk[r['tracklet_id']] = dict(
                        tid=r['tracklet_id'], cid=r['canonical_id'], label=int(r['label']),
                        camid=CAMID.get(tok, 0), tok=tok,
                        outfit=(r['collection'], r['set']), paths=[])
                t['paths'].append(r['chip_path'])

        ids = sorted({v['cid'] for v in tk.values()})
        self.val_ids = set()
        self.holdout_rule = holdout_rule
        self.eligible = []
        if holdout:
            if holdout_rule not in HOLDOUT_RULES:
                raise ValueError("unknown holdout_rule %r; expected one of %s"
                                 % (holdout_rule, sorted(HOLDOUT_RULES)))
            need, how = HOLDOUT_RULES[holdout_rule]
            cov, n1k = {}, {}
            for v in tk.values():
                cov.setdefault(v['cid'], set()).add(v['tok'])
                if v['tok'] == '1000m':
                    n1k[v['cid']] = n1k.get(v['cid'], 0) + len(v['paths'])
            self.eligible = sorted(i for i in ids if set(need) <= cov.get(i, set()))
            n = min(holdout, len(self.eligible))
            if how == 'random':
                pick = random.Random(seed).sample(self.eligible, n)
            else:
                # key on (count, cid) so ties break deterministically, not by dict order
                pick = sorted(self.eligible, key=lambda i: (n1k.get(i, 0), i))[:n]
            self.val_ids = set(pick)

        self.train_ids = [i for i in ids if i not in self.val_ids]
        if self.val_ids:
            # holdout removes identities, so the manifest's labels are no longer contiguous
            self.label = {c: i for i, c in enumerate(self.train_ids)}
        else:
            self.label = {v['cid']: v['label'] for v in tk.values()}   # the manifest's own labels
        outfits = sorted({v['outfit'] + (v['cid'],) for v in tk.values() if v['cid'] not in self.val_ids})
        self.clothes = {o: i for i, o in enumerate(outfits)}           # CAL only
        self._tk = tk
        self._variants = []          # [[tid, ...]] -- the sampler picks exactly ONE per group per epoch
        self._variant_w = []         # matching selection weights
        if sim_mode is not None:
            self._attach_sim(sim_corpus)

    # ---- synthetic ------------------------------------------------------------------------------

    def _attach_sim(self, sim_corpus):
        """Match synthetic variants to their real source crops for the selected training mode."""
        by_src = {}          # real chip_path -> {severity: sim chip_path}
        with open(osp.join(sim_corpus, SIM_MANIFEST)) as f:
            for r in csv.DictReader(f):
                by_src.setdefault(r['src_chip_path'], {})[r['severity']] = r['chip_path']

        real400 = [t for t in self._tk.values() if t['tok'] == '400m']
        # Each selected real crop must have a matching synthetic crop.
        missing = [p for t in real400 for p in t['paths'] if p not in by_src]
        if missing:
            raise SystemExit(
                "%d of %d real 400m chips have no synthetic "
                "counterpart (e.g. %s). The sim manifest is stale relative to train_manifest.csv."
                % (len(missing), sum(len(t['paths']) for t in real400), missing[0]))

        for t in real400:
            t['_root'] = self.chip_root
            group, weights = [], []
            if self.sim_mode == 'mix':
                group.append(t['tid']); weights.append(1.0 - self.sim_p)
            elif self.sim_mode == 'swapall':
                # weight 0 -- grouped, so never served, but still ACCOUNTED for. Leaving it ungrouped
                # would keep the real tracklet permanently on and inflate the epoch to 'add' size.
                group.append(t['tid']); weights.append(0.0)
            if self.sim_mode == 'dup':
                d = dict(t, tid='dup:%s' % t['tid'], paths=list(t['paths']), _root=self.chip_root)
                self._tk[d['tid']] = d          # always on: a second, independent 32-frame draw
            else:
                for sev in self.sim_severities:
                    s = dict(t, tid='sim:%s:%s' % (t['tid'], sev), tok=SIM_TOK,
                             camid=CAMID[SIM_TOK], _root=self.sim_chip_root,
                             paths=[by_src[p][sev] for p in t['paths']])
                    self._tk[s['tid']] = s
                    group.append(s['tid'])
                    # The synthetic mass is split evenly across whatever strengths are enabled, so
                    # the real-vs-synthetic rate stays exactly sim_p however many there are.
                    weights.append(self.sim_p / len(self.sim_severities)
                                   if self.sim_mode == 'mix' else 1.0 / len(self.sim_severities))
            if group:
                # 'add' keeps the real tracklet permanently on and rotates only the strength;
                # 'mix' and 'swapall' put the real tracklet into the choice (or omit it entirely).
                self._variants.append(group)
                self._variant_w.append(weights)

    def variant_groups(self):
        """Return the tracklet IDs and sampling weights for each group of variants."""
        return list(zip(self._variants, self._variant_w))

    def _sel(self, val):
        return [t for t in self._tk.values() if (t['cid'] in self.val_ids) == val]

    def abspath(self, rel, root=None):
        return osp.join(root or self.chip_root, rel)

    def image_items_with_tracklets(self, val=False):
        """Return image training items and their tracklet IDs before frame sampling."""
        items, tks = [], []
        for t in self._sel(val):
            pid = self.label.get(t['cid'], -1)
            cl = self.clothes.get(t['outfit'] + (t['cid'],), -1)
            root = t.get('_root')
            for p in t['paths']:
                items.append((self.abspath(p, root), pid, t['camid'], cl))
                tks.append(t['tid'])
        return items, tks

    def image_items(self, epoch_seed=0, frames_per_tracklet=32, val=False):
        """Return image training items with the requested frame cap."""
        rng = random.Random(epoch_seed)
        out = []
        for t in self._sel(val):
            ps = t['paths']
            if frames_per_tracklet and len(ps) > frames_per_tracklet:
                ps = [ps[i] for i in sorted(rng.sample(range(len(ps)), frames_per_tracklet))]
            pid = self.label.get(t['cid'], -1)
            cl = self.clothes.get(t['outfit'] + (t['cid'],), -1)
            root = t.get('_root')
            out.extend((self.abspath(p, root), pid, t['camid'], cl) for p in ps)
        rng.shuffle(out)
        return out

    def tracklets(self, val=False):
        """Group crop paths by tracklet and return their subject, camera, and track labels."""
        return [(tuple(self.abspath(p, t.get('_root')) for p in t['paths']),
                 self.label.get(t['cid'], -1), t['camid'], i)
                for i, t in enumerate(sorted(self._sel(val), key=lambda x: x['cid']))]

    def tracklets_with_ids(self, val=False):
        """Return tracklet items and the matching tracklet IDs in the same order."""
        ts = sorted(self._sel(val), key=lambda x: x['cid'])
        items = [(tuple(self.abspath(p, t.get('_root')) for p in t['paths']),
                  self.label.get(t['cid'], -1), t['camid'], i)
                 for i, t in enumerate(ts)]
        return items, [t['tid'] for t in ts]

    def val_split(self):
        """paperGR-shaped: gallery 400m, query 500m + 1000m, over the held-out ids."""
        g, q = [], []
        for i, t in enumerate(sorted(self._sel(True), key=lambda x: x['cid'])):
            rec = (tuple(self.abspath(p) for p in t['paths']), t['cid'], t['camid'], i)
            (g if t['tok'] == '400m' else q if t['tok'] in ('500m', '1000m') else []).append(rec)
        return g, q

    def stats(self):
        tr, va = self._sel(False), self._sel(True)
        d = dict(train_ids=len(self.train_ids), train_tracklets=len(tr),
                 train_chips=sum(len(t['paths']) for t in tr),
                 val_ids=len(self.val_ids), val_tracklets=len(va),
                 clothes_classes=len(self.clothes))
        if self.sim_mode:
            sim = [t for t in tr if t['tid'].startswith('sim:')]
            d.update(sim_mode=self.sim_mode, sim_p=self.sim_p,
                     sim_severities=list(self.sim_severities),
                     sim_tracklets=len(sim), sim_chips=sum(len(t['paths']) for t in sim),
                     variant_groups=len(self._variants))
        return d

    def range_composition(self, cap=32, val=False):
        """Count subjects, tracklets, crops, and capped images for each camera range."""
        out = {}
        for t in self._sel(val):
            s, tr, ch, cp = out.get(t['tok'], (set(), 0, 0, 0))
            s = s | {t['cid']}
            out[t['tok']] = (s, tr + 1, ch + len(t['paths']), cp + min(len(t['paths']), cap))
        return {k: (len(v[0]), v[1], v[2], v[3]) for k, v in out.items()}
