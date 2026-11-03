# encoding: utf-8
"""Save training settings and identity splits alongside checkpoints. Record the iteration at the
end of each epoch."""
import json
import os
import os.path as osp
import subprocess
import sys
import time


def _git_output(repo_dir, *args):
    try:
        output = subprocess.check_output(['git', *args], cwd=repo_dir, stderr=subprocess.DEVNULL)
        return output.decode().strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _repository_record(repo_dir):
    status = _git_output(repo_dir, 'status', '--porcelain')
    return {
        'dir': osp.abspath(repo_dir),
        'sha': _git_output(repo_dir, 'rev-parse', 'HEAD'),
        'branch': _git_output(repo_dir, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': None if status is None else bool(status),
    }


def write_run_json(output_dir, cfg=None, args=None, corpus=None, repo_dir='.',
                   extra=None, filename='run.json'):
    """Write the run settings, repository revision, and training and validation identities.
    Return the record path."""
    rec = {
        'written': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'host': os.uname()[1],
        'argv': sys.argv,
        'python': sys.executable,
        'output_dir': osp.abspath(output_dir),
        'repo': _repository_record(repo_dir),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    if args is not None:
        rec['args'] = dict(vars(args))
    if cfg is not None:
        rec['cfg'] = str(cfg)          # yacs CfgNode -> full resolved dump
    if corpus is not None:
        rec['corpus'] = {
            'stats': corpus.stats(),
            'chip_root': corpus.chip_root,
            'holdout_rule': getattr(corpus, 'holdout_rule', None),
            'n_eligible': len(getattr(corpus, 'eligible', [])),
            'val_ids': sorted(corpus.val_ids),
            'train_ids': sorted(corpus.train_ids),
        }
    if extra:
        rec.update(extra)
    path = osp.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return path


def append_epoch_map(output_dir, epoch, global_step, filename='epoch_map.jsonl'):
    """Append the completed epoch and its global iteration to the checkpoint index."""
    with open(osp.join(output_dir, filename), 'a') as f:
        f.write(json.dumps({'epoch': int(epoch), 'global_step': int(global_step)}) + '\n')
