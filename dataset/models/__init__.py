from .variant1_mamba_experts import MambaExpertMoEModel
from .variant2_modality_specialist import ModalitySpecialistMoEModel
from .variant3_hierarchical_moe import HierarchicalMoEModel
from .variant4_context_conditioned import ContextConditionedMoEModel

MODEL_REGISTRY = {
    "variant1": MambaExpertMoEModel,
    "variant2": ModalitySpecialistMoEModel,
    "variant3": HierarchicalMoEModel,
    "variant4": ContextConditionedMoEModel,
}
