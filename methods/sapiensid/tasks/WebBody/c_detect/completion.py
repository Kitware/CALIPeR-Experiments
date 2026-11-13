"""Verify completed WebBody output shards before source deletion (stdlib only)."""
import hashlib
import json
from pathlib import Path
import sys

OUTPUTS = ('train.rec', 'train.idx', 'train.tsv', 'done_list.txt',
           'body_kps_xyn.csv', 'bbox_xyxyn.csv', 'confidence.csv')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inputs(source):
    source = Path(source)
    return {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}


def finish(directory, source_hashes, expected, processed, detections):
    directory = Path(directory)
    if expected <= 0 or processed != expected:
        raise ValueError('not all source rows were processed; source must be retained')
    data = dict(inputs=source_hashes, expected=expected, processed=processed, detections=detections,
                outputs={name: digest(directory / name) for name in OUTPUTS})
    temporary = directory / 'completion.tmp'
    temporary.write_text(json.dumps(data))
    temporary.replace(directory / 'completion.json')


def verify(source, output, ranks):
    source, output = Path(source), Path(output)
    hashes = inputs(source)
    if not hashes or ranks < 1:
        raise ValueError('source is empty or rank count is invalid')
    for rank in range(ranks):
        directory = output / (source.name + '_rank_' + str(rank))
        data = json.loads((directory / 'completion.json').read_text())
        if data['inputs'] != hashes or data['expected'] <= 0 or data['expected'] != data['processed']:
            raise ValueError('source changed or output shard is incomplete')
        if data['outputs'] != {name: digest(directory / name) for name in OUTPUTS}:
            raise ValueError('output files changed after completion')
        for name in ('train.tsv', 'done_list.txt', 'body_kps_xyn.csv', 'bbox_xyxyn.csv', 'confidence.csv'):
            with (directory / name).open() as stream:
                rows = sum(1 for _ in stream) - int(name.endswith('.csv'))
            if rows != data['detections']:
                raise ValueError('output row counts disagree: ' + name)


if __name__ == '__main__':
    try:
        verify(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    except (OSError, ValueError, KeyError, IndexError) as error:
        sys.exit('Incomplete extraction; retaining source: ' + str(error))
