"""Prepare Figure 6 conditions from complete generation manifests without copying chips."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .inputs import InputFrame, _collection
from .cameras import range_token
from .reid_protocol import PROTOCOL, SCOPES, SEVERITIES, STEMS, in_scope


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frame_key(frame):
    return frame.collection, frame.relpath, frame.tracklet_id, frame.frame


def _group(frames):
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame.collection, frame.relpath, frame.tracklet_id)].append(frame)
    result = []
    for (collection, relpath, tracklet), rows in sorted(groups.items()):
        if len({r.canonical_id for r in rows}) != 1 or len({r.frame for r in rows}) != len(rows):
            raise ValueError(f"Inconsistent identity or duplicate frames in {relpath}")
        result.append(dict(key=digest_json([collection, relpath, tracklet]), collection=collection,
                           relpath=relpath, tracklet_id=tracklet, canonical_id=rows[0].canonical_id,
                           frames=[dict(frame=r.frame, path=r.path) for r in sorted(rows, key=lambda r: r.frame)]))
    return result


def _kwcoco(path, expected_token):
    doc = json.loads(path.read_text())
    def registry(key):
        rows = doc[key]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key} IDs in {path}")
        return {r["id"]: r for r in rows}
    subjects, videos, tracks, images = [registry(key) for key in ("subjects", "videos", "tracks", "images")]
    annotations = defaultdict(list)
    for annotation in doc["annotations"]:
        if annotation["image_id"] not in images:
            raise ValueError(f"Annotation references an absent image in {path}")
        annotations[annotation["image_id"]].append(annotation)
    rows = []
    for image in images.values():
        annotations_for_image = annotations[image["id"]]
        if len(annotations_for_image) != 1:
            raise ValueError(f"Expected one person annotation per chip in {path}")
        track = tracks[annotations_for_image[0]["track_id"]]
        video = videos[image["video_id"]]
        if video["camera_token"] != expected_token:
            raise ValueError(f"Expected only {expected_token} chips in {path}")
        relpath = video["name"]
        if range_token(Path(relpath).parent.name) != expected_token:
            raise ValueError(f"Physical camera disagrees with {expected_token} condition in {path}")
        chip = Path(image["file_name"])
        if not chip.is_absolute():
            root = Path(doc.get("info", {}).get("crop_root", ""))
            chip = (root if root.is_absolute() else path.parent / root) / chip
        rows.append(InputFrame(str(chip.resolve()), str(subjects[track["subject_id"]]["name"]),
                               _collection(relpath), relpath, str(track["id"]), int(image["frame_index"]),
                               expected_token, Path(relpath).parent.name))
    return rows


def _variant_conditions(variants):
    result = {}
    for condition, strength in SEVERITIES.items():
        matches = [v for v in variants if v["simulator"] == "pybsm" and v["target_camera"] == "1000m"
                   and math.isclose(v["cn2"], strength, rel_tol=1e-9, abs_tol=0)]
        if len(matches) != 1:
            raise ValueError(f"Figure 6 requires exactly one pyBSM 1000m variant at Cn2={strength:g}")
        result[condition] = matches[0]
    return result


def prepare_conditions(generation_dir, scopes=("combined",), *, hash_images=True):
    """Read and validate all five conditions; hash_images=False is for previews only."""
    root = Path(generation_dir).resolve()
    generation = json.loads((root / "generation.json").read_text())
    settings = generation["settings"]
    if (generation.get("kind") != "synthetic_chips" or settings["input"].get("purpose", "reid") != "reid"
            or settings["input"]["partition"] != "test" or settings["source_camera"] != "500m"):
        raise ValueError("Figure 6 requires a completed test-chip Re-ID generation with source_camera=500m")
    if not scopes or set(scopes) - set(SCOPES):
        raise ValueError("Unknown or empty Figure 6 scopes")
    variants = _variant_conditions(settings["variants"])
    sources = [InputFrame(**r) for r in json.loads((root / "selection.json").read_text())["sources"]]
    source_lookup = {frame_key(r): r for r in sources}
    if not sources or len(source_lookup) != len(sources) or len({r.path for r in sources}) != len(sources):
        raise ValueError("Missing or duplicate source chips")
    if any(r.tok != "500m" or r.physical_camera != "500m" or _collection(r.relpath) != r.collection for r in sources):
        raise ValueError("Source chips must have consistent collection metadata and physical 500m camera")
    outputs = {v["name"]: {} for v in variants.values()}
    variants_by_name = {v["name"]: v for v in variants.values()}
    with (root / "chips.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["name"] not in outputs:
                continue
            source = InputFrame(**row["source"])
            key = frame_key(source)
            expected = variants_by_name[row["name"]]
            if (source_lookup.get(key) != source or key in outputs[row["name"]]
                    or any(row.get(k) != v for k, v in expected.items()) or row.get("source_camera") != "500m"):
                raise ValueError(f"Duplicate chip or inconsistent synthetic provenance: {row['name']}/{key}")
            outputs[row["name"]][key] = row["chip_path"]
    for name, frames in outputs.items():
        if set(frames) != set(source_lookup):
            raise ValueError(f"Incomplete synthetic frame pairing: {name}")
        if len({str(Path(p).resolve()) for p in frames.values()}) != len(frames):
            raise ValueError(f"Multiple source frames refer to the same synthetic chip: {name}")
    gallery = _group(_kwcoco(root / "eval" / "gallery.kwcoco.json", "400m"))
    target = _group(_kwcoco(root / "eval" / "query_1000m.kwcoco.json", "1000m"))
    queries = {"real-500m": _group(sources), "real-1000m": target}
    for condition, variant in variants.items():
        queries[condition] = _group([InputFrame(**{**r.to_dict(), "path": outputs[variant["name"]][frame_key(r)]})
                                      for r in sources])
    gallery_relpaths = {t["relpath"] for t in gallery}
    if gallery_relpaths & {t["relpath"] for rows in queries.values() for t in rows}:
        raise ValueError("A tracklet appears in both gallery and queries")
    gallery_paths = {str(Path(f["path"]).resolve()) for t in gallery for f in t["frames"]}
    query_paths = {str(Path(f["path"]).resolve()) for rows in queries.values() for t in rows for f in t["frames"]}
    if gallery_paths & query_paths:
        raise ValueError("A chip appears in both gallery and queries")
    # Hash each unique image once even when real/source conditions share paths.
    hashes = {}
    for t in gallery + [t for rows in queries.values() for t in rows]:
        for f in t["frames"]:
            p = Path(f["path"])
            if not p.is_file():
                raise FileNotFoundError(p)
            if hash_images:
                if str(p) not in hashes:
                    hashes[str(p)] = sha256_file(p)
                f["sha256"] = hashes[str(p)]
    summary = {}
    for scope in scopes:
        g = [t for t in gallery if in_scope(t, scope)]
        ids = {t["canonical_id"] for t in g}
        counts = {}
        for condition in STEMS:
            q = [t for t in queries[condition] if in_scope(t, scope)]
            scored = [t for t in q if t["canonical_id"] in ids]
            if not g or not scored:
                raise ValueError(f"No mated gallery/query tracklets for {scope}/{condition}")
            counts[condition] = dict(tracklets=len(q), scored=len(scored), unmated=len(q) - len(scored),
                                     chips=sum(len(t["frames"]) for t in q))
        summary[scope] = dict(gallery_tracklets=len(g), conditions=counts)
    return dict(format_version=1, protocol=PROTOCOL, generation_dir=str(root), scopes=list(scopes),
                gallery=gallery, queries=queries, summary=summary,
                generation_sha256=sha256_file(root / "generation.json"), variants=variants,
                population_note="Synthetic severities and real-500m share source frames; real-1000m is a different query population.")


def prepare(config):
    destination = Path(config["paths"]["output_dir"]) / "prepared"
    if destination.exists():
        raise FileExistsError(f"Prepared conditions already exist: {destination}")
    manifest = prepare_conditions(config["paths"]["generation_dir"], config["scopes"])
    destination.mkdir(parents=True)
    write_json(destination / "conditions.json", manifest)
    return manifest
