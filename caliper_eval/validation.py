"""Check feature caches and select compatible evaluation settings for reports."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(path.parent),
                                         delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, str(path))
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def validate_caches(caches, poolings=None):
    """Check feature dimensions, identity mappings, and run settings across gallery and query
    archives."""
    if len(caches) < 2:
        raise ValueError('evaluation requires a gallery and at least one query cache')
    identities, reverse, tokens = {}, {}, set()
    shared = None
    for index, cache in enumerate(caches):
        required = ('feats', 'pid', 'camid', 'video', 'frame', 'dive_track_id', 'crop_path',
                    'pid_to_subject', 'crop_root', 'split_role', 'camera_token', 'feat_dim',
                    'method', 'source', 'split', 'neck_feat', 'weight')
        missing = [key for key in required if key not in cache]
        if missing:
            raise ValueError('feature cache missing fields: {}'.format(', '.join(missing)))
        features = cache['feats']
        if features.ndim != 2 or not all(features.shape) or not np.issubdtype(features.dtype, np.floating):
            raise ValueError('features must be a nonempty numeric (rows, dimensions) array')
        if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
            raise ValueError('feature vectors must be finite and nonzero')
        for key in ('pid', 'camid', 'video', 'frame', 'dive_track_id', 'crop_path'):
            if cache[key].ndim != 1 or len(cache[key]) != len(features):
                raise ValueError('feature-cache row lengths disagree: ' + key)
        for key in ('pid', 'camid', 'frame', 'dive_track_id'):
            if not np.issubdtype(cache[key].dtype, np.integer):
                raise ValueError(key + ' must contain integer labels')
        for key in required[7:]:
            if cache[key].ndim != 0:
                raise ValueError(key + ' must be scalar metadata')
        if int(cache['feat_dim']) != features.shape[1]:
            raise ValueError('feat_dim disagrees with the feature array')
        role = str(cache['split_role'])
        if role != ('gallery' if index == 0 else 'query'):
            raise ValueError('expected gallery first, followed by query roles')
        token = str(cache['camera_token'])
        if not token or (index and token in tokens):
            raise ValueError('query camera tokens must be nonempty and unique')
        if index:
            tokens.add(token)
        signature = tuple(str(cache[k]) for k in ('method', 'source', 'split', 'neck_feat', 'weight', 'feat_dim'))
        if shared is not None and signature != shared:
            raise ValueError('mixed model/checkpoint/preprocessing metadata across feature caches')
        shared = signature
        restriction = str(cache.get('pooling', ''))
        if restriction not in ('', 'image', 'video'):
            raise ValueError('invalid cache pooling restriction')
        if restriction and poolings and any(level != restriction for level in poolings):
            raise ValueError("cache requires --pooling " + restriction)
        mapping = json.loads(str(cache['pid_to_subject']))
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('pid_to_subject must be a nonempty JSON object')
        for pid, subject in mapping.items():
            if (str(int(pid)) != pid or not isinstance(subject, str) or not subject
                    or subject != subject.strip()):
                raise ValueError('invalid canonical identity mapping')
            if identities.setdefault(pid, subject) != subject or reverse.setdefault(subject, pid) != pid:
                raise ValueError('identity labels differ between feature caches')
        if any(str(int(pid)) not in mapping for pid in cache['pid']):
            raise ValueError('feature identity absent from pid_to_subject')
        videos = {}
        for video, pid, camera in zip(cache['video'], cache['pid'], cache['camid']):
            label = (int(pid), int(camera))
            if not str(video) or videos.setdefault(str(video), label) != label:
                raise ValueError('one video must describe one subject and camera')


def published_protocol(result):
    """Require explicit published settings; filenames alone are not evidence."""
    return (result.get('pool_op') == 'mean' and result.get('metric') == 'cosine'
            and result.get('cam_exclusion') == 'none'
            and type(result.get('min_track_crops')) is int and result['min_track_crops'] == 0
            and type(result.get('max_per_video')) is int and result['max_per_video'] == 0 and 'video' in result.get('results', {}))
