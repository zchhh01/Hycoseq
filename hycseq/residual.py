import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(value):
    if value <= 0:
        raise ValueError("adaptive parameters must start above zero")
    if value > 20:
        return value
    return math.log(math.expm1(value))


class WeightedLorentzMerge(nn.Module):
    def __init__(
        self,
        weighting="fixed",
        skip_weight=1.0,
        transform_weight=1.0,
        transform_weight_init=1.0,
        rescale_output=False,
        rescale_mode="fixed",
        rescale_init=1.0,
        epsilon=1e-8,
    ):
        super().__init__()
        if weighting not in {"fixed", "adaptive"}:
            raise ValueError("weighting must be fixed or adaptive")
        if rescale_mode not in {"fixed", "adaptive"}:
            raise ValueError("rescale_mode must be fixed or adaptive")
        if min(skip_weight, transform_weight, transform_weight_init, rescale_init) <= 0:
            raise ValueError("merge weights and scales must be positive")
        self.weighting = weighting
        self.rescale_output = rescale_output
        self.rescale_mode = rescale_mode
        self.epsilon = float(epsilon)
        self.register_buffer("skip_weight", torch.tensor(float(skip_weight)))
        if weighting == "fixed":
            self.register_buffer("transform_weight", torch.tensor(float(transform_weight)))
        else:
            self.raw_transform_weight = nn.Parameter(torch.tensor(inverse_softplus(float(transform_weight_init))))
        if rescale_output:
            if rescale_mode == "fixed":
                self.register_buffer("output_scale", torch.tensor(float(rescale_init)))
            else:
                self.raw_output_scale = nn.Parameter(torch.tensor(inverse_softplus(float(rescale_init))))

    def effective_transform_weight(self, reference):
        if self.weighting == "fixed":
            return self.transform_weight.to(reference)
        return F.softplus(self.raw_transform_weight).to(reference) + self.epsilon

    def effective_scale(self, reference):
        if self.rescale_mode == "fixed":
            return self.output_scale.to(reference)
        return F.softplus(self.raw_output_scale).to(reference) + self.epsilon

    def forward(self, skip, transformed, manifold):
        if skip.shape != transformed.shape:
            raise ValueError("Lorentz residual operands must have identical shapes")
        combined = self.skip_weight.to(transformed) * skip
        combined = combined + self.effective_transform_weight(transformed) * transformed
        norm = manifold.inner(combined, combined, keepdim=True).abs().clamp_min(self.epsilon).sqrt()
        curvature = manifold.k.to(combined).detach() if not manifold.k.requires_grad else manifold.k.to(combined)
        if torch.any(curvature.detach() <= 0) or torch.any(~torch.isfinite(curvature.detach())):
            raise RuntimeError("Lorentz curvature scale must remain finite and positive")
        merged = manifold.projx(torch.sqrt(curvature) * combined / norm)
        if self.rescale_output:
            space = merged.narrow(-1, 1, merged.shape[-1] - 1)
            merged = manifold.projx(manifold.add_time(self.effective_scale(merged) * space))
        return merged
