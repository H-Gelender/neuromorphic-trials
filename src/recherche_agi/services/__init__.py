"""Services réutilisables du module adaptatif."""
from .callbacks import EpochCallback, MisclassificationCallback
from .checkpoint import list_checkpoints, load_model_state, save_model
from .llm import GeneratedCode, LLMService, server_healthy
from .network_analysis import (
    analyze_neurons,
    analyze_routing,
    compute_activations,
    denormalize_mnist,
    find_similar_pairs,
    prune_neurons,
    summarize_model,
)

__all__ = [
    # callbacks
    "EpochCallback",
    "MisclassificationCallback",
    # checkpoint
    "list_checkpoints",
    "load_model_state",
    "save_model",
    # llm
    "GeneratedCode",
    "LLMService",
    "server_healthy",
    # network analysis
    "analyze_neurons",
    "analyze_routing",
    "compute_activations",
    "denormalize_mnist",
    "find_similar_pairs",
    "prune_neurons",
    "summarize_model",
]
