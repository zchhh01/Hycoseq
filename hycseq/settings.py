import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .tokens import TokenLayout, resolve_token_layout


def boolean_value(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def integer_series(value):
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(integer_series(item))
        return result
    normalized = str(value).replace("[", "").replace("]", "").replace(",", " ")
    return [int(item) for item in normalized.split()]


@dataclass
class ExperimentSettings:
    run_id: str
    corpus: str
    task: str
    data_root: str
    save_root: Optional[str]
    accelerator: str
    seed: int
    class_count: int
    epochs: int
    batch: int
    max_bases: int
    resume_from: Optional[str]
    learning_rate: float
    decay: float
    geometry_learning_rate: float
    geometry_decay: float
    optimizer_name: str
    schedule: bool
    schedule_steps: List[int]
    schedule_factor: float
    width: int
    stages: int
    head_width: int
    layerwise_curvature: bool
    learn_curvature: bool
    curvature_scale: float
    merge_weights: str
    skip_weight: float
    transform_weight: float
    transform_weight_init: float
    rescale_output: bool
    rescale_mode: str
    rescale_init: float
    representation: str
    kmer: int
    token_stride: int
    ambiguous_policy: str
    token_width: int
    reverse_complement: bool
    shape_trace: bool
    layout: TokenLayout


def _parser():
    parser = argparse.ArgumentParser(description="hycseq experiment runner")
    parser.add_argument("-r", "--recipe")
    parser.add_argument("--run_id", default="hycseq_run")
    parser.add_argument("--corpus", choices=["TEB", "GUE", "GB"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_root", default="artifacts")
    parser.add_argument("--accelerator", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--class_count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--max_bases", type=int, default=200)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--decay", type=float, default=0.1)
    parser.add_argument("--geometry_learning_rate", type=float, default=2e-2)
    parser.add_argument("--geometry_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer_name", choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"], default="RiemannianAdam")
    parser.add_argument("--schedule", type=boolean_value, default=True)
    parser.add_argument("--schedule_steps", nargs="+", default=[60, 85])
    parser.add_argument("--schedule_factor", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--head_width", type=int, default=528)
    parser.add_argument("--layerwise_curvature", type=boolean_value, default=False)
    parser.add_argument("--learn_curvature", type=boolean_value, default=True)
    parser.add_argument("--curvature_scale", type=float, default=1.0)
    parser.add_argument("--merge_weights", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--skip_weight", type=float, default=1.0)
    parser.add_argument("--transform_weight", type=float, default=1.0)
    parser.add_argument("--transform_weight_init", type=float, default=1.0)
    parser.add_argument("--rescale_output", type=boolean_value, default=False)
    parser.add_argument("--rescale_mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--rescale_init", type=float, default=1.0)
    parser.add_argument("--representation", choices=["base", "kmer_onehot", "kmer_embed"], default="base")
    parser.add_argument("--kmer", type=int, default=3)
    parser.add_argument("--token_stride", type=int, default=1)
    parser.add_argument("--ambiguous_policy", choices=["unknown"], default="unknown")
    parser.add_argument("--token_width", type=int, default=64)
    parser.add_argument("--reverse_complement", type=boolean_value, default=False)
    parser.add_argument("--shape_trace", type=boolean_value, default=False)
    return parser


def validate_settings(settings):
    positive = {
        "class_count": settings.class_count,
        "epochs": settings.epochs,
        "batch": settings.batch,
        "max_bases": settings.max_bases,
        "learning_rate": settings.learning_rate,
        "geometry_learning_rate": settings.geometry_learning_rate,
        "width": settings.width,
        "stages": settings.stages,
        "head_width": settings.head_width,
        "curvature_scale": settings.curvature_scale,
        "skip_weight": settings.skip_weight,
        "transform_weight": settings.transform_weight,
        "transform_weight_init": settings.transform_weight_init,
        "rescale_init": settings.rescale_init,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("positive values required for: " + ", ".join(invalid))
    if settings.width < 2:
        raise ValueError("width must be at least 2")
    if settings.decay < 0 or settings.geometry_decay < 0:
        raise ValueError("decay values must be non-negative")
    if not 0 < settings.schedule_factor <= 1:
        raise ValueError("schedule_factor must be in (0, 1]")
    return settings


def parse_settings(argv=None):
    source_argv = list(sys.argv[1:] if argv is None else argv)
    recipe_probe = argparse.ArgumentParser(add_help=False)
    recipe_probe.add_argument("-r", "--recipe")
    recipe_value, _ = recipe_probe.parse_known_args(source_argv)
    recipe_arguments = []
    if recipe_value.recipe:
        for raw_line in Path(recipe_value.recipe).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError("invalid recipe line: " + raw_line)
            key, raw_value = line.split("=", 1)
            recipe_arguments.append("--" + key.strip())
            recipe_arguments.extend(shlex.split(raw_value.strip()))
    values = _parser().parse_args(recipe_arguments + source_argv)
    values.schedule_steps = integer_series(values.schedule_steps)
    layout = resolve_token_layout(
        representation=values.representation,
        max_bases=values.max_bases,
        kmer=values.kmer,
        stride=values.token_stride,
        token_width=values.token_width,
    )
    parsed = vars(values)
    parsed.pop("recipe", None)
    settings = ExperimentSettings(layout=layout, **parsed)
    return validate_settings(settings)
