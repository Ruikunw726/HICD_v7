from .nfa import NeighborFeatureAggregation
from .tfm import TemporalFusionModule, MultiScaleTemporalFusion
from .task_adapter import TaskSpecificAdapter, MultiScaleAdapter

# CLIP module is optional (removed in v7+)
try:
    from .clip_module import LightweightCLIP, LearnableStateEmbeddings
except ImportError:
    LightweightCLIP = None
    LearnableStateEmbeddings = None

from .interaction import ChangeGuidedROI, SpatialModulation, BoxConsistencyLoss
from .label_structure import (
    LabelStructure, GNNLabelStructure, HierarchicalConditionDecoder, PixelStateHead,
    VALID_COMBINATIONS, TARGET_VALID_STATES, TARGET_NAMES, STATE_NAMES,
    CATEGORY_HIERARCHY, NUM_TARGETS, NUM_STATES,
)
from .frequency_branch import FrequencyBranch
