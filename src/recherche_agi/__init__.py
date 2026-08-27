"""Module adaptatif — recherche AGI (pipeline COCO final).

Système 100% non supervisé d'apprentissage hiérarchique sur COCO Stuff :
- Modern Hopfield Network (softmax βWx) remplace le WTA
- neurogenèse libre (sans plafond)
- hiérarchie profonde (couches créées par convergence)
- skip connections sans plafond (inter + intra-couche)
- message passing (consensus + inhibition)
- condition de fin = équilibre (surprise convergente)
"""
from .unsupervised import (AnchorNeurons, DynamicAnchorNeurons, dynamic_k,
                           homeostatic_threshold, image_to_patches,
                           topdown_feedback)
from .texture_features import color_texture_features, extract_all_features
from .evolutive_coco import HierarchicalCOCO
from .message_passing import (build_grid_adjacency, conductivity_physarum,
                              message_passing_step, spatial_consensus,
                              message_passing_train, update_physarum_conductance)
from .skip_connections import SkipConnections
from .topdown_projection import (activate_deep_neuron, backprop_topdown,
                                 spatial_mask_c1, project_to_pixels,
                                 topdown_projection, multi_instance_topdown)
from .modern_hopfield import (project_hopfield, reconstruct, surprise,
                              winner_hopfield, oja_hopfield_update)

__all__ = [
    # caractéristiques de texture/couleur
    "color_texture_features",
    "extract_all_features",
    # entraînement évolutif COCO (pipeline final)
    "HierarchicalCOCO",
    # message passing (graphe de nœuds)
    "build_grid_adjacency",
    "conductivity_physarum",
    "message_passing_step",
    "spatial_consensus",
    "message_passing_train",
    "update_physarum_conductance",
    # skip connections (transversales inter/intra-couches)
    "SkipConnections",
    # projection top-down guidée (masque sémantique 1×1)
    "activate_deep_neuron",
    "backprop_topdown",
    "spatial_mask_c1",
    "project_to_pixels",
    "topdown_projection",
    "multi_instance_topdown",
    # modern hopfield network (remplace le WTA)
    "project_hopfield",
    "reconstruct",
    "surprise",
    "winner_hopfield",
    "oja_hopfield_update",
    # classification non supervisée (base)
    "AnchorNeurons",
    "DynamicAnchorNeurons",
    "dynamic_k",
    "homeostatic_threshold",
    "image_to_patches",
    "topdown_feedback",
]
