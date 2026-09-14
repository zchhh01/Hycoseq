import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.hyperboloid.layers import LorentzBatchNorm1d, LorentzConv1d, LorentzFullyConnected, LorentzMLR, LorentzReLU
from geometry.hyperboloid.geometry import CustomLorentz

from .residual import WeightedLorentzMerge


class LorentzTransform(nn.Module):
    def __init__(self, manifold, input_width, output_width):
        super().__init__()
        self.layers = nn.Sequential(
            LorentzConv1d(manifold, input_width, output_width, kernel_size=9, padding=4),
            LorentzBatchNorm1d(manifold, output_width),
            LorentzReLU(manifold),
            LorentzConv1d(manifold, output_width, output_width, kernel_size=9, padding=4),
            LorentzBatchNorm1d(manifold, output_width),
        )

    def forward(self, values):
        return self.layers(values)


class ResidualStage(nn.Module):
    def __init__(self, manifold, width, merge_options):
        super().__init__()
        self.transform = LorentzTransform(manifold, width, width)
        self.merge = WeightedLorentzMerge(**merge_options)
        self.activation = LorentzReLU(manifold)

    def forward(self, values, manifold):
        return self.activation(self.merge(values, self.transform(values), manifold))


class HycSeqClassifier(nn.Module):
    def __init__(
        self,
        class_count,
        sequence_steps,
        input_width,
        width,
        head_width,
        stages=3,
        layerwise_curvature=False,
        learn_curvature=True,
        curvature_scale=1.0,
        vocabulary_size=None,
        padding_id=None,
        token_width=64,
        shape_trace=False,
        merge_options=None,
    ):
        super().__init__()
        if stages < 1 or width < 2 or sequence_steps < 1:
            raise ValueError("stages, width, and sequence_steps are invalid")
        self.sequence_steps = int(sequence_steps)
        self.width = int(width)
        self.shape_trace = bool(shape_trace)
        self._traced = False
        self.embedding = None
        spatial_input_width = int(input_width)
        if vocabulary_size is not None:
            if padding_id is None:
                raise ValueError("padding_id is required with a token vocabulary")
            self.embedding = nn.Embedding(vocabulary_size, token_width, padding_idx=padding_id)
            spatial_input_width = int(token_width)
        manifold_count = stages + 1 if layerwise_curvature else 1
        self.manifolds = nn.ModuleList(
            [CustomLorentz(k=curvature_scale, learnable=learn_curvature) for _ in range(manifold_count)]
        )
        self.stem = LorentzTransform(self.manifolds[0], spatial_input_width + 1, width)
        self.stem_activation = LorentzReLU(self.manifolds[0])
        options = dict(merge_options or {})
        self.residual_stages = nn.ModuleList(
            [ResidualStage(self._manifold(index), width, options) for index in range(1, stages)]
        )
        self.head_manifold_index = stages if layerwise_curvature else 0
        head_manifold = self._manifold(self.head_manifold_index)
        flattened_width = 1 + sequence_steps * (width - 1)
        self.projection = LorentzFullyConnected(head_manifold, flattened_width, head_width, bias=True)
        self.head_activation = LorentzReLU(head_manifold)
        self.classifier = LorentzMLR(head_manifold, head_width, class_count)

    def _manifold(self, stage_index):
        return self.manifolds[min(stage_index, len(self.manifolds) - 1)]

    def _prepare(self, values):
        if self.embedding is not None:
            values = self.embedding(values.long())
        else:
            values = values.permute(0, 2, 1)
        return self.manifolds[0].projx(F.pad(values, (1, 0)))

    def forward(self, values, lengths=None):
        values = self._prepare(values)
        values = self.stem_activation(self.stem(values))
        previous_manifold = self.manifolds[0]
        for stage_index, stage in enumerate(self.residual_stages, start=1):
            current_manifold = self._manifold(stage_index)
            if current_manifold is not previous_manifold:
                values = current_manifold.switch_man(values, previous_manifold)
            values = stage(values, current_manifold)
            previous_manifold = current_manifold
        head_manifold = self._manifold(self.head_manifold_index)
        if head_manifold is not previous_manifold:
            values = head_manifold.switch_man(values, previous_manifold)
            previous_manifold = head_manifold
        if values.shape[1:] != (self.sequence_steps, self.width):
            raise ValueError("unexpected encoder output shape: " + str(tuple(values.shape)))
        if self.shape_trace and not self._traced:
            print("encoder_output=" + str(tuple(values.shape)))
        values = previous_manifold.lorentz_flatten(values.unsqueeze(2))
        if self.shape_trace and not self._traced:
            print("flattened_output=" + str(tuple(values.shape)))
            self._traced = True
        values = self.head_activation(self.projection(values))
        return self.classifier(values)
