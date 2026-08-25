"""Module adaptatif (recherche AGI).

Organisation :
- ``data``       : chargement MNIST et filtrage par chiffres.
- ``training``   : entraînement, early stopping, évaluation.
- ``services``   : callbacks, analyse de réseau, service LLM.

Le **modèle** (TinyExpert / TinyMoE) vit dans le notebook : on le fait évoluer
itérativement là-bas. Les services y accèdent par duck-typing (nn.Module).
"""
from .data import filter_by_digits, load_mnist
from .enhanced_reservoir import EnhancedReservoir, competitive_pooling, multi_axis_signature, retina_filter
from .image_graph import image_to_graph, physarum_from_image, superpixels
from .local_ssm import LocalSSM, SSMLayer, surprise_to_delta
from .neuromodulated import NeuromodulatedReservoir, SSMNeuromodulatedReservoir, lateral_inhibition, metabolic_n_iter, surprise_eta
from .physarum import PhysarumGraph, classify_by_drainage, grid_graph_from_image
from .predictive_physarum import HybridBlobPredictive, Tube, train_readout
from .sensory_bundle import SensoryBundle, PredictiveEncoder, oja_hebbian_update, surprise_rate
from .synaptic_physarum import SynapticReservoir, dendritic_pooling, hebbian_plasticity, synaptic_signature
from .training import (EarlyStopping, evaluate, freeze_experts, train_expert_on_dataset,
                       train_router, train_tiny_moe, unfreeze_experts, verify_frozen)
from .unsupervised import AnchorNeurons, dynamic_k, homeostatic_threshold, topdown_feedback

# Rendre les services disponibles au niveau racine du paquet.
from .services import (  # noqa: E402
    EpochCallback,
    MisclassificationCallback,
    GeneratedCode,
    LLMService,
    server_healthy,
    analyze_neurons,
    analyze_routing,
    compute_activations,
    denormalize_mnist,
    find_similar_pairs,
    prune_neurons,
    summarize_model,
    save_model,
    load_model_state,
    list_checkpoints,
)

__all__ = [
    # données
    "filter_by_digits",
    "load_mnist",
    # réservoir synaptique amélioré
    "EnhancedReservoir",
    "competitive_pooling",
    "multi_axis_signature",
    "retina_filter",
    # boucle neuromodulée
    "NeuromodulatedReservoir",
    "SSMNeuromodulatedReservoir",
    "lateral_inhibition",
    "metabolic_n_iter",
    "surprise_eta",
    # image -> graphe (superpixels)
    "image_to_graph",
    "superpixels",
    "physarum_from_image",
    # SSM local (mémoire temporelle)
    "LocalSSM",
    "SSMLayer",
    "surprise_to_delta",
    # fourre-tout sensoriel
    "SensoryBundle",
    "PredictiveEncoder",
    "oja_hebbian_update",
    "surprise_rate",
    # classification non supervisée
    "AnchorNeurons",
    "dynamic_k",
    "homeostatic_threshold",
    "topdown_feedback",
    # physarum
    "PhysarumGraph",
    "classify_by_drainage",
    "grid_graph_from_image",
    # hybride blob + predictive coding
    "HybridBlobPredictive",
    "Tube",
    "train_readout",
    # réservoir synaptique
    "SynapticReservoir",
    "dendritic_pooling",
    "hebbian_plasticity",
    "synaptic_signature",
    # entraînement
    "EarlyStopping",
    "evaluate",
    "freeze_experts",
    "train_expert_on_dataset",
    "train_router",
    "train_tiny_moe",
    "unfreeze_experts",
    "verify_frozen",
    # services
    "EpochCallback",
    "MisclassificationCallback",
    "GeneratedCode",
    "LLMService",
    "server_healthy",
    "analyze_neurons",
    "analyze_routing",
    "compute_activations",
    "denormalize_mnist",
    "find_similar_pairs",
    "prune_neurons",
    "summarize_model",
    "save_model",
    "load_model_state",
    "list_checkpoints",
]
