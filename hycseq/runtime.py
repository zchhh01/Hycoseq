from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiStepLR

from geometry.geoopt import ManifoldParameter
from geometry.geoopt.optim import RiemannianAdam, RiemannianSGD

from .network import HycSeqClassifier


def build_model(settings):
    merge_options = {
        "weighting": settings.merge_weights,
        "skip_weight": settings.skip_weight,
        "transform_weight": settings.transform_weight,
        "transform_weight_init": settings.transform_weight_init,
        "rescale_output": settings.rescale_output,
        "rescale_mode": settings.rescale_mode,
        "rescale_init": settings.rescale_init,
    }
    embedded = settings.representation == "kmer_embed"
    return HycSeqClassifier(
        class_count=settings.class_count,
        sequence_steps=settings.layout.sequence_steps,
        input_width=settings.layout.input_width,
        width=settings.width,
        head_width=settings.head_width,
        stages=settings.stages,
        layerwise_curvature=settings.layerwise_curvature,
        learn_curvature=settings.learn_curvature,
        curvature_scale=settings.curvature_scale,
        vocabulary_size=settings.layout.vocabulary_size if embedded else None,
        padding_id=settings.layout.padding_id if embedded else None,
        token_width=settings.token_width,
        shape_trace=settings.shape_trace,
        merge_options=merge_options,
    )


def parameter_groups(model, settings):
    ordinary_decay = []
    ordinary_no_decay = []
    manifold = []
    curvature = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if name.endswith(".k"):
            curvature.append(parameter)
        elif isinstance(parameter, ManifoldParameter):
            manifold.append(parameter)
        elif any(marker in lowered for marker in ("bias", "bn", "norm", "raw_")):
            ordinary_no_decay.append(parameter)
        else:
            ordinary_decay.append(parameter)
    groups = [
        {"name": "network", "params": ordinary_decay},
        {"name": "network_no_decay", "params": ordinary_no_decay, "weight_decay": 0.0},
        {"name": "manifold", "params": manifold, "lr": settings.geometry_learning_rate, "weight_decay": settings.geometry_decay},
        {"name": "curvature", "params": curvature, "lr": 1e-4, "weight_decay": 0.0},
    ]
    return [group for group in groups if group["params"]]


def build_optimizer(model, settings):
    groups = parameter_groups(model, settings)
    common = {"lr": settings.learning_rate, "weight_decay": settings.decay}
    if settings.optimizer_name == "RiemannianAdam":
        optimizer = RiemannianAdam(groups, stabilize=1, **common)
    elif settings.optimizer_name == "RiemannianSGD":
        optimizer = RiemannianSGD(groups, momentum=0.9, nesterov=True, stabilize=1, **common)
    elif settings.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(groups, **common)
    else:
        optimizer = torch.optim.SGD(groups, momentum=0.9, nesterov=True, **common)
    assigned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(expected):
        raise RuntimeError("optimizer parameter coverage is invalid")
    scheduler = None
    if settings.schedule:
        scheduler = MultiStepLR(optimizer, milestones=settings.schedule_steps, gamma=settings.schedule_factor)
    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, scheduler, settings, epoch, score):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "hycseq-v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "settings": asdict(settings),
        "epoch": epoch,
        "score": score,
    }
    torch.save(state, target)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    scheduler_state = state.get("scheduler", state.get("lr_scheduler"))
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    return state
