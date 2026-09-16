"""Opt-in PointDINO registrations; existing detector registries are unchanged."""
from .pointdino import PointDINO
from .pointdino_fidt_head import PointDINOFIDTAuxHead
from .pointdino_head import PointDINOHead
from .pointdino_layers import (PointDINOCdnQueryGenerator,
                              PointDINOTransformerDecoder)
from .pointdino_match_cost import PointDINOL1Cost
from .pointdino_metric import PointDINOMetric
from .pointdino_transforms import (PointDINOLoadAnnotations,
                                   PointDINOPackDetInputs, PointDINORandomCrop,
                                   PointDINORandomFlip, PointDINOResize)

__all__ = [
    'PointDINO', 'PointDINOHead', 'PointDINOTransformerDecoder',
    'PointDINOCdnQueryGenerator', 'PointDINOFIDTAuxHead', 'PointDINOL1Cost',
    'PointDINOMetric', 'PointDINOLoadAnnotations', 'PointDINOPackDetInputs',
    'PointDINORandomCrop', 'PointDINORandomFlip', 'PointDINOResize',
]
