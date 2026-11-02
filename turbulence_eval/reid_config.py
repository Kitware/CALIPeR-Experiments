"""Configuration for independent Figure 6 preparation, scoring and reporting."""

from configparser import ConfigParser
import json
from pathlib import Path

from .reid_protocol import DEFAULT_MODELS, SCOPES, folder_name


def _words(value):
    return value.replace(",", " ").split()


def load_config(path):
    path = Path(path).resolve()
    config = ConfigParser(interpolation=None)
    with path.open() as stream:
        config.read_file(stream)
    if config.defaults():
        raise ValueError("Re-ID configuration does not support DEFAULT options")
    allowed = {
        "paths": {"generation_dir", "output_dir", "cache_dir"},
        "evaluation": {"models", "scopes", "device", "batch_size", "max_rank", "gallery_unit", "unmated"},
    }
    model_keys = {"factory", "checkpoint", "python_path", "input_kind", "label", "corpus", "color",
                  "assets", "code_paths", "options"}
    for section in config.sections():
        keys = model_keys if section.startswith("model:") else allowed.get(section)
        if keys is None or set(config[section]) - keys:
            raise ValueError(f"Unknown Re-ID section or options: {section}")

    def resolve(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent / p).resolve())

    paths = {key: resolve(config["paths"][key]) for key in ("generation_dir", "output_dir")}
    paths["cache_dir"] = resolve(config["paths"].get("cache_dir", str(Path(paths["output_dir"]) / "features")))
    generation, output = Path(paths["generation_dir"]), Path(paths["output_dir"])
    if output == generation or generation in output.parents or output in generation.parents:
        raise ValueError("Re-ID output_dir and generation_dir must be separate directory trees")
    cache = Path(paths["cache_dir"])
    if cache == generation or generation in cache.parents:
        raise ValueError("Feature cache must be outside generation_dir")
    options = config["evaluation"]
    scopes, names = _words(options.get("scopes", "combined")), _words(options["models"])
    if not scopes or len(set(scopes)) != len(scopes) or set(scopes) - set(SCOPES):
        raise ValueError("scopes must be unique entries from combined, brc1_only, brc2_only")
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide unique model names")
    if options.get("gallery_unit", "tracklet") != "tracklet" or options.get("unmated", "exclude") != "exclude":
        raise ValueError("Figure 6 requires gallery_unit=tracklet and unmated=exclude")
    models = {}
    for name in names:
        folder_name(name)
        section = config[f"model:{name}"]
        factory = section["factory"]
        if factory.count(":") != 1 or not all(factory.split(":")):
            raise ValueError(f"{name}: factory must be module:function")
        kind = section.get("input_kind", "image")
        if kind not in ("image", "video"):
            raise ValueError(f"{name}: input_kind must be image or video")
        label, corpus, color = DEFAULT_MODELS.get(name, (name, "", ""))
        extra = json.loads(section.get("options", "{}"))
        if not isinstance(extra, dict) or set(extra) & {"checkpoint", "device", "batch_size"}:
            raise ValueError(f"{name}: options must be a JSON object without reserved factory arguments")
        # Multiline paths permit spaces in filenames; one path per line.
        models[name] = dict(factory=factory, checkpoint=resolve(section["checkpoint"]), input_kind=kind,
                            label=section.get("label", label), corpus=section.get("corpus", corpus),
                            color=section.get("color", color), options=extra,
                            python_path=resolve(section["python_path"]) if section.get("python_path") else None,
                            **{key: [resolve(v.strip()) for v in section.get(key, "").splitlines() if v.strip()]
                               for key in ("assets", "code_paths")})
    batch_size, max_rank = options.getint("batch_size", 32), options.getint("max_rank", 20)
    if batch_size < 1 or max_rank < 2:
        raise ValueError("batch_size must be positive and max_rank must be >= 2")
    return dict(config_file=str(path), paths=paths, scopes=scopes, models=models,
                device=options.get("device", "auto"), batch_size=batch_size, max_rank=max_rank)
