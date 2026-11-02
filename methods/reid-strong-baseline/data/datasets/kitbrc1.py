# encoding: utf-8
"""Load crop-level KITBRC1 gallery and query splits for BoT."""
import os
import json
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset

try:
    from caliper_eval import paths
except ImportError:
    import sys
    sys.path.insert(0, osp.expanduser(os.environ.get('CALIPER_EVAL_HOME', osp.abspath(osp.join(osp.dirname(__file__), '../../../../')))))
    from caliper_eval import paths


def load_split_kwcoco(coco_path, max_per_appearance=0):
    """Return [(crop_path, pid, camid), ...] from one crop-level kwcoco split file."""
    with open(coco_path) as f:
        d = json.load(f)
    cam_of_video = {v['id']: v['camera_id'] for v in d['videos']}
    subj_of_track = {t['id']: t['subject_id'] for t in d['tracks']}
    ann_of_image = {a['image_id']: a for a in d['annotations']}

    # records: (path, pid, camid, video_id, track_id, frame_index)
    recs = []
    for im in d['images']:
        a = ann_of_image.get(im['id'])
        if a is None:
            continue
        tid = a['track_id']
        recs.append((im['file_name'], subj_of_track[tid], cam_of_video[im['video_id']],
                     im['video_id'], tid, im.get('frame_index', 0)))

    if max_per_appearance and max_per_appearance > 0:
        groups = defaultdict(list)
        for r in recs:
            groups[(r[3], r[1])].append(r)            # appearance = (video_id, pid)
        kept = []
        for g in groups.values():
            g.sort(key=lambda r: r[5])                # by frame
            n = len(g)
            if n > max_per_appearance:
                idx = sorted({round(i * (n - 1) / (max_per_appearance - 1)) for i in range(max_per_appearance)})
                g = [g[i] for i in idx]
            kept.extend(g)
        recs = kept

    return [(r[0], r[1], r[2]) for r in recs]


class KITBRC1(BaseImageDataset):
    """CALIPeR KITBRC1 ground-range ReID benchmark (eval-only, per-crop / image-level)."""
    dataset_dir = 'kitbrc1'

    def __init__(self, root='./data', verbose=True, **kwargs):
        super(KITBRC1, self).__init__()
        split = os.environ.get('KITBRC1_SPLIT', kwargs.get('split', 'ground_range'))
        max_per = int(os.environ.get('KITBRC1_MAX_PER_APPEARANCE', kwargs.get('max_per_appearance', 16)))
        splits_root = os.environ.get('KITBRC1_SPLITS_DIR', kwargs.get('splits_dir', paths.splits_root()))

        self.split_dir = osp.join(splits_root, split)
        self.gallery_file = osp.join(self.split_dir, 'gallery.kwcoco.json')
        self.query_files = [osp.join(self.split_dir, 'query_500m.kwcoco.json'),
                            osp.join(self.split_dir, 'query_1000m.kwcoco.json')]
        self._check_before_run()

        gallery = load_split_kwcoco(self.gallery_file, max_per)
        query = []
        for qf in self.query_files:
            query += load_split_kwcoco(qf, max_per)
        train = gallery  # dummy; num_train_pids only sizes the classifier (skipped by load_param)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> KITBRC1 loaded (split={}, max_per_appearance={})".format(split, max_per))
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        for p in [self.split_dir, self.gallery_file] + self.query_files:
            if not osp.exists(p):
                raise RuntimeError("'{}' is not available".format(p))
