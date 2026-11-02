"""Write synthetic training crop lists and evaluation splits with shared subject and camera IDs."""

from collections import defaultdict
import json
from pathlib import Path


TRAIN_FIELDS = ["chip_path", "tracklet_id", "src_tracklet_id", "canonical_id", "tok", "collection"]


def synthetic_tracklet(frame, variant):
    return f"sim:{frame.tracklet_id}:{variant}"


def evaluation_registries(selection):
    frames = selection.sources + [row for rows in selection.references.values() for row in rows]
    subjects = {name: i + 1 for i, name in enumerate(sorted({r.canonical_id for r in frames}))}
    cameras = {name: i + 1 for i, name in enumerate(sorted({r.physical_camera for r in frames}))}
    return subjects, cameras


def build_kwcoco(frames, subjects, cameras, *, outputs=None, token=None):
    """Export a crop-level split with physical camera IDs and separate simulated range labels."""
    doc = dict(info={"crop_root": ""}, subjects=[dict(id=i, name=n) for n, i in subjects.items()],
               videos=[], tracks=[], images=[], annotations=[])
    grouped = defaultdict(list)
    for row in frames:
        grouped[(row.collection, row.relpath, row.tracklet_id)].append(row)
    for vid, (_, rows) in enumerate(sorted(grouped.items()), 1):
        row = rows[0]
        doc["videos"].append(dict(id=vid, name=row.relpath, camera_id=cameras[row.physical_camera],
                                  camera_token=token or row.tok))
        doc["tracks"].append(dict(id=vid, subject_id=subjects[row.canonical_id], dive_track_id=row.dive_track_id))
        for row in sorted(rows, key=lambda r: r.frame):
            image_id = len(doc["images"]) + 1
            path = row.path if outputs is None else outputs(row)
            doc["images"].append(dict(id=image_id, file_name=str(path), video_id=vid, frame_index=row.frame))
            doc["annotations"].append(dict(id=image_id, image_id=image_id, track_id=vid))
    return doc


def write_evaluation_references(output, selection, subjects, cameras):
    if not selection.references:
        return []
    written = []
    gallery = selection.references["gallery"]
    if not gallery:
        raise ValueError("No usable gallery chips remain in the requested split/collections")
    docs = {"gallery.kwcoco.json": build_kwcoco(gallery, subjects, cameras)}
    by_token = defaultdict(list)
    for row in selection.references["query"]:
        by_token[row.tok].append(row)
    for token, rows in sorted(by_token.items()):
        docs[f"query_{token}.kwcoco.json"] = build_kwcoco(rows, subjects, cameras)
    output.mkdir(parents=True, exist_ok=False)
    for name, doc in docs.items():
        path = output / name
        path.write_text(json.dumps(doc) + "\n")
        written.append(str(path))
    return written
