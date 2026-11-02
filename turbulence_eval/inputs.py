"""Select source crops, queries, and galleries from CALIPeR manifests."""

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import random

from .cameras import camera_label, range_token


@dataclass(frozen=True)
class InputFrame:
    path: str
    canonical_id: str
    collection: str
    relpath: str
    tracklet_id: str
    frame: int
    tok: str
    physical_camera: str
    dive_track_id: int = -1
    outfit: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Selection:
    sources: list[InputFrame]
    references: dict[str, list[InputFrame]]
    excluded: dict[str, int]
    real_frames: list[InputFrame] = field(default_factory=list)

    def summary(self):
        return dict(
            source_chips=len(self.sources),
            subjects=len({r.canonical_id for r in self.sources}),
            tracklets=len({(r.collection, r.tracklet_id) for r in self.sources}),
            collections=dict(Counter(r.collection for r in self.sources)),
            reference_chips={role: len(rows) for role, rows in self.references.items()},
            real_chips=len(self.real_frames),
            excluded=self.excluded,
        )


def read_csv(path):
    with Path(path).open(newline="") as stream:
        yield from csv.DictReader(stream)


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in value:
        raise ValueError(f"Expected a relative dataset path: {value!r}")
    return path


def _collection(relpath):
    prefix = relative_path(relpath).parts[0]
    try:
        return {"KITBRC1": "brc1", "brc2_in_brc1_format": "brc2"}[prefix]
    except KeyError as error:
        raise ValueError(f"Unknown collection mount in {relpath!r}") from error


def _sample(rows, options, seed):
    requested = set(options["subjects"])
    present = {r.canonical_id for r in rows}
    if requested - present:
        raise ValueError(f"Requested subjects have no eligible source chips: {sorted(requested - present)}")
    subjects = sorted(requested or present)
    if options["num_subjects"] is not None:
        subjects = subjects[:options["num_subjects"]]
    grouped = defaultdict(list)
    for row in rows:
        if row.canonical_id in subjects:
            grouped[row.canonical_id].append(row)
    selected = []
    for subject, frames in sorted(grouped.items()):
        frames.sort(key=lambda r: (r.collection, r.relpath, r.frame))
        # A subject's draw does not depend on other subjects' eligibility.
        digest = hashlib.sha256(f"{seed}:{subject}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        limit = options["max_images"]
        if limit != -1 and len(frames) > limit:
            frames = rng.sample(frames, limit)
        selected.extend(frames)
    return sorted(selected, key=lambda r: (r.canonical_id, r.collection, r.relpath, r.frame))


def _load_index(settings):
    options, paths = settings["input"], settings["paths"]
    index, excluded = {}, Counter()
    with Path(paths["chip_index"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            relpath = row["relpath"]
            relative_path(relpath)
            if row.get("status") != "ok" or not row.get("frames"):
                excluded["unusable_index_records"] += 1
                continue
            if row.get("media") != "video" or row.get("selector") != "ground":
                continue
            if row.get("protocol_version") != options["protocol_version"]:
                raise ValueError(f"Chip protocol mismatch for {relpath}")
            if relpath in index:
                raise ValueError(f"Duplicate chip-index relpath: {relpath}")
            index[relpath] = row
    return index, excluded


def _indexed_frames(rec, settings, *, tracklet_id=None, outfit=""):
    relpath = relative_path(rec["relpath"])
    collection = _collection(str(relpath))
    frames = [int(frame) for frame in rec["frames"]]
    if len(set(frames)) != len(frames) or any(frame < 0 for frame in frames):
        raise ValueError(f"Invalid/duplicate frame indices: {relpath}")
    root = Path(settings["paths"]["chip_root"])
    folder = root / settings["input"]["protocol_version"] / relpath.with_suffix("")
    return [InputFrame(
        path=str(folder / f"{frame:06d}.jpg"), canonical_id=rec["canon"],
        collection=collection, relpath=str(relpath),
        tracklet_id=str(tracklet_id if tracklet_id is not None else rec["key"]),
        frame=frame, tok=range_token(rec["tok"]), physical_camera=relpath.parent.name,
        dive_track_id=int(rec["track_id"]) if rec.get("track_id") is not None else -1,
        outfit=outfit,
    ) for frame in sorted(frames)]


def _training_sources(settings, index, partitions, excluded):
    corpus = Path(settings["paths"]["corpus_dir"])
    options = settings["input"]
    classes = json.loads((corpus / "train_classes.json").read_text())
    tracks = list(read_csv(corpus / "train_tracklets.csv"))
    keep_keys = None
    if options["deduplicate"]:
        manifest = list(read_csv(corpus / "train_manifest.csv"))
        manifest_keys = {r["tracklet_key"] for r in manifest}
        manifest_ids = {r["canonical_id"] for r in manifest}
        keep_keys = {r["tracklet_key"] for r in tracks
                     if r["tracklet_key"] in manifest_keys or r["canonical_id"] not in manifest_ids}
    chosen = {}
    for row in tracks:
        if row["modality"] != "field" or row["cls"] != "ground":
            continue
        if row["collection"] not in options["collections"] or range_token(row["tok"]) != settings["source_token"]:
            continue
        if keep_keys is not None and row["tracklet_key"] not in keep_keys:
            excluded["shared_source_tracklets"] += 1
            continue
        if row["canonical_id"] not in partitions["train"] or row["canonical_id"] not in classes:
            raise ValueError(f"Non-training identity in training corpus: {row['canonical_id']}")
        relpath = row["source_relpath"]
        rec = index.get(relpath)
        if rec is None:
            raise ValueError(f"Training tracklet absent/unusable in chip index: {relpath}")
        if (rec["canon"] != row["canonical_id"] or rec["key"] != row["tracklet_key"]
                or _collection(relpath) != row["collection"]
                or range_token(rec["tok"]) != range_token(row["tok"])):
            raise ValueError(f"Corpus/index metadata disagree: {relpath}")
        if row["tracklet_id"] in chosen:
            raise ValueError(f"Duplicate corpus tracklet ID: {row['tracklet_id']}")
        chosen[row["tracklet_id"]] = {
            r.frame: r for r in _indexed_frames(rec, settings, tracklet_id=row["tracklet_id"], outfit=row["set"])
        }
    rows = []
    seen = set()
    for row in read_csv(corpus / "train_chips.csv"):
        available = chosen.get(row["tracklet_id"])
        if available is None:
            continue
        frame = int(row["frame"])
        item = available.get(frame)
        path = Path(settings["paths"]["chip_root"]) / relative_path(row["chip_path"])
        if item is None or str(path) != item.path or row["canonical_id"] != item.canonical_id:
            raise ValueError(f"Training chip disagrees with index: {row['chip_path']}")
        key = (item.tracklet_id, item.frame)
        if key in seen:
            raise ValueError(f"Duplicate training chip: {key}")
        seen.add(key)
        rows.append(item)
    return rows


def select_inputs(settings):
    options, paths = settings["input"], settings["paths"]
    references, real_frames = {}, []
    if options["mode"] != "chips":
        root = Path(paths["input_dir"])
        if not root.is_dir():
            raise NotADirectoryError(root)
        groups = ([(p.name, p / settings["source_camera"]) for p in sorted(root.iterdir()) if p.is_dir()]
                  if options["mode"] == "subject-tree" else [("images", root)])
        rows = []
        for subject, folder in groups:
            if not folder.is_dir():
                continue
            for frame, path in enumerate(sorted(folder.iterdir())):
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    rows.append(InputFrame(str(path), subject, "", str(path.relative_to(root)),
                                           subject, frame, settings["source_token"], settings["source_camera"]))
        excluded = Counter()
    else:
        partitions = json.loads(Path(paths["partition_file"]).read_text())
        partitions = {key: set(partitions[key]) for key in ("train", "test")}
        if partitions["train"] & partitions["test"]:
            raise ValueError("Train and test identity partitions overlap")
        index, excluded = _load_index(settings)
        if options.get("purpose") == "quality":
            tokens = {settings["source_token"], *(v["target_token"] for v in settings["variants"])}
            allowed = partitions[options["partition"]] if options["partition"] != "all" else None
            for rec in index.values():
                if (_collection(rec["relpath"]) in options["collections"] and (allowed is None or rec["canon"] in allowed)
                        and range_token(rec["tok"]) in tokens):
                    real_frames.extend(_indexed_frames(rec, settings))
            target_subjects = {r.canonical_id for r in real_frames if r.tok != settings["source_token"]}
            rows = [r for r in real_frames if r.tok == settings["source_token"] and r.canonical_id in target_subjects]
        elif options["partition"] == "train":
            rows = _training_sources(settings, index, partitions, excluded)
        else:
            split = json.loads(Path(paths["split_json"]).read_text())
            if set(split) != {"gallery", "query"}:
                raise ValueError("Evaluation split must contain gallery and query mappings")
            for role in ("gallery", "query"):
                references[role] = []
                for subject, relpaths in sorted(split[role].items()):
                    # Preserve gallery-only distractors, but never admit train identities.
                    if subject in partitions["train"] or (role == "query" and subject not in partitions["test"]):
                        raise ValueError(f"Identity outside test partition in {role}: {subject}")
                    seen = set()
                    for relpath in sorted(relpaths):
                        relative_path(relpath)
                        if relpath in seen:
                            raise ValueError(f"Duplicate split entry: {relpath}")
                        seen.add(relpath)
                        rec = index.get(relpath)
                        if rec is None:
                            excluded[f"{role}_unusable_or_absent_records"] += 1
                            continue
                        if rec["canon"] != subject:
                            raise ValueError(f"Split/index canonical identity disagree: {relpath}")
                        if _collection(relpath) in options["collections"]:
                            references[role].extend(_indexed_frames(rec, settings))
            rows = [r for r in references[options["source_role"]] if r.tok == settings["source_token"]]
    rows = _sample(rows, options, settings["sample_seed"])
    if not rows:
        raise ValueError("Selection contains no source images")
    if len({r.path for r in rows}) != len(rows):
        raise ValueError("Multiple selected records refer to the same source chip")
    if real_frames:
        subjects = {r.canonical_id for r in rows}
        real_frames = sorted((r for r in real_frames if r.canonical_id in subjects),
                             key=lambda r: (r.canonical_id, r.collection, r.relpath, r.frame))
    if references:
        subjects = {r.canonical_id for r in rows}
        query = references["query"]
        references["query"] = [r for r in query if r.canonical_id in subjects]
        excluded["query_chips_without_selected_source_subject"] += len(query) - len(references["query"])
    if options["mode"] == "chips":
        missing = {r.collection for r in rows} - set(settings["calibration"])
        if missing:
            raise ValueError(f"Explicit calibration profile required for collections: {sorted(missing)}")
        for row in rows:
            profile = settings["calibration"][row.collection]
            if camera_label(row.physical_camera, profile) != camera_label(settings["source_camera"], profile):
                raise ValueError(f"Physical camera disagrees with source calibration: {row.relpath}")
    # Fail before creating outputs if even one selected/reference chip is missing.
    for row in rows:
        if not Path(row.path).is_file():
            raise FileNotFoundError(row.path)
    for frames in [*references.values(), real_frames]:
        for row in frames:
            if not Path(row.path).is_file():
                raise FileNotFoundError(row.path)
    return Selection(rows, references, dict(excluded), real_frames)


def sources_for_variant(selection, variant):
    """A quality comparison generates only subjects with real target chips."""
    if not selection.real_frames:
        return selection.sources
    subjects = {r.canonical_id for r in selection.real_frames if r.tok == variant["target_token"]}
    return [r for r in selection.sources if r.canonical_id in subjects]
