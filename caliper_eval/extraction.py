"""Extract gallery and query features and reuse complete caches with matching inputs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from . import features_io, paths
from .validation import atomic_json, validate_caches


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_directory(directory, splits_dir, method, source, split, weight):
    expected, caches = {}, []
    for role, filename, token in features_io.discover_role_files(str(splits_dir)):
        records, internal, crop_root = features_io.load_crops_full(filename, role)
        token = token or internal
        name = features_io.feature_filename(role, token)
        if name in expected:
            raise ValueError('split roles produce duplicate feature filenames: ' + name)
        cache = features_io.load_npz(str(directory / name))
        expected[name] = sha256(directory / name)
        caches.append(cache)
        if any(str(cache[k]) != value for k, value in
               [('method', method), ('source', source), ('split', split),
                ('camera_token', token), ('split_role', role)]):
            raise ValueError('extracted metadata disagrees with requested run: ' + name)
        if Path(str(cache['weight'])).resolve() != Path(weight).resolve():
            raise ValueError('extractor used a different checkpoint: ' + name)
        subjects = json.loads(str(cache['pid_to_subject']))
        wanted = Counter((r['video'], r['subject'], r['frame'], r['camid'],
                          str(Path(r['path']).resolve())) for r in records)
        actual = Counter((str(v), subjects[str(int(pid))], int(frame), int(cam),
                          str((Path(str(cache['crop_root'])) / str(crop)).resolve()))
                         for v, pid, frame, cam, crop in zip(cache['video'], cache['pid'],
                         cache['frame'], cache['camid'], cache['crop_path']))
        if actual != wanted:
            raise ValueError('extracted frames/identities disagree with split: ' + name)
    if set(p.name for p in directory.glob('*.npz')) != set(expected):
        raise ValueError('cache contains missing or stale feature roles')
    validate_caches(caches)
    return expected


def source_fingerprints(command):
    """Invalidate reuse when the method's Python/config source changes."""
    roots = [Path.cwd(), Path(__file__).parent]
    found = {}
    ignored = {'weights', 'outputs', 'checkpoints', '__pycache__', 'venv', 'env'}
    for index, root in enumerate(roots):
        for directory, subdirs, files in os.walk(str(root)):
            subdirs[:] = [d for d in subdirs if not d.startswith('.') and d not in ignored]
            for filename in files:
                path = Path(directory) / filename
                if path.suffix in ('.py', '.sh', '.yml', '.yaml'):
                    found['{}:{}'.format(index, path.relative_to(root))] = sha256(path)
    return found


def run(args):
    import fcntl
    from importlib import metadata
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or '{output}' not in command:
        raise ValueError('extractor command must include an {output} argument')
    for value in (args.split, args.dataset, args.method, args.source):
        if not value or value in ('.', '..') or '/' in value or '\\' in value:
            raise ValueError('split/dataset/method/source must be individual directory names')
    assets = [Path(args.weight)] + [Path(p) for p in args.asset]
    for asset in assets:
        if not asset.is_file():
            raise FileNotFoundError('required checkpoint/configuration asset missing: ' + str(asset))
    split_dir = Path(paths.splits_dir(args.split))
    roles = features_io.discover_role_files(str(split_dir))
    if not any(r == 'gallery' for r, _, _ in roles) or not any(r == 'query' for r, _, _ in roles):
        raise ValueError('split must contain gallery and query roles')
    signature = dict(version=1, command=command, method=args.method, source=args.source,
                     split=args.split, dataset=args.dataset, python=sys.version,
                     assets={str(p.resolve()): sha256(p) for p in assets},
                     splits={Path(p).name: sha256(p) for _, p, _ in roles},
                     code=source_fingerprints(command),
                     packages=sorted((d.metadata['Name'], d.version) for d in metadata.distributions()
                                     if d.metadata.get('Name')))
    directory = Path(args.output_dir or paths.features_dir(args.split, args.method, args.source, dataset=args.dataset))
    if directory.is_symlink():
        raise ValueError('cache directory must not be a symbolic link')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / ('.' + directory.name + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = directory / 'feature_manifest.json'
        if not args.force and manifest.is_file():
            try:
                saved = json.loads(manifest.read_text())
                # JSON turns tuples into lists.
                if saved['signature'] == json.loads(json.dumps(signature)):
                    hashes = validate_directory(directory, split_dir, args.method, args.source, args.split, args.weight)
                    if saved['outputs'] == hashes:
                        print('Reusing verified complete cache: ' + str(directory))
                        return
            except (OSError, ValueError, KeyError, TypeError):
                pass
        temporary = Path(tempfile.mkdtemp(prefix='.extract-', dir=str(directory.parent)))
        archive = None
        try:
            subprocess.run([str(temporary) if v == '{output}' else v for v in command], check=True)
            hashes = validate_directory(temporary, split_dir, args.method, args.source, args.split, args.weight)
            # Inputs cannot change while extraction is running.
            if any(sha256(p) != signature['assets'][str(p.resolve())] for p in assets):
                raise ValueError('checkpoint/configuration changed during extraction')
            if any(sha256(p) != signature['splits'][Path(p).name] for _, p, _ in roles):
                raise ValueError('split changed during extraction')
            if features_io.discover_role_files(str(split_dir)) != roles or source_fingerprints(command) != signature['code']:
                raise ValueError('split roles or source code changed during extraction')
            atomic_json(temporary / 'feature_manifest.json', dict(signature=signature, outputs=hashes))
            if directory.exists():
                archive = Path(paths.ROOT) / 'cache-archive' / args.dataset / args.split / args.method / args.source / uuid.uuid4().hex
                archive.parent.mkdir(parents=True, exist_ok=True)
                os.rename(str(directory), str(archive))
            try:
                os.rename(str(temporary), str(directory))
            except BaseException:
                if archive is not None:
                    os.rename(str(archive), str(directory))
                raise
            if archive is not None:
                print('Previous cache and evaluations preserved at ' + str(archive))
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('method', 'source', 'weight'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', default='paperGR')
    parser.add_argument('--dataset', default='combined_brc1_brc2')
    parser.add_argument('--output-dir', help='optional separate cache for an ablation')
    parser.add_argument('--asset', action='append', default=[])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('method extraction environments require Python 3.10 or newer')
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, 'Extraction failed: {}\n'.format(exc))


if __name__ == '__main__':
    main()
