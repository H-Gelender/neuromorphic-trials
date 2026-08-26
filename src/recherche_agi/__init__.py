"""Module adaptatif — recherche AGI.

Pipeline final : classification 100% non supervisée par neurones d'ancrage,
avec fatigue homéostatique (anti-oubli) et élagage Physarum (gestion des
ressources). Le modèle s'adapte au drift (0/1 → 2-9) sans oubli catastrophique
et atteint ~0.95 d'acc globale sur MNIST.

Modules conservés :
- data.py         : chargement MNIST + filtre par chiffres
- unsupervised.py : AnchorNeurons (SOM Hebbien), WTA dynamique, fatigue,
                    top-down feedback, élagage Physarum
- training.py     : entraînement auxiliaire (gardé pour référence)
"""
from .coco_scenes import CocoScenes, PALETTE, STUFF_CLASSES
from .data import filter_by_digits, load_mnist
from .unsupervised import (AnchorNeurons, DynamicAnchorNeurons, dynamic_k,
                           homeostatic_threshold, image_to_patches,
                           topdown_feedback)
from .stable_layers import RespiratoryController, StructuralActivitySignal
from .coco_pipeline import accuracy_per_class, classify_patch, train_class_by_class
from .texture_features import color_texture_features, extract_all_features
from .online_training import EquilibriumCallback
from .visualize import visualize_architecture, visualize_evolution, visualize_input
from .visualize_coco_images import (colormap_mask, visualize_coco_sample,
                                    visualize_patches_rgb)
from .evolutive_coco import HierarchicalCOCO
from .monitor import (architecture_frame, draw_architecture,
                      save_architecture_gif)
from .report import TrainingTracker, generate_report
from .message_passing import (build_grid_adjacency, conductivity_physarum,
                              message_passing_step, spatial_consensus)

__all__ = [
    # scènes COCO (prototype de scale)
    "CocoScenes",
    "PALETTE",
    "STUFF_CLASSES",
    # caractéristiques de texture/couleur
    "color_texture_features",
    "extract_all_features",
    # entraînement en ligne + callback d'équilibre
    "EquilibriumCallback",
    # visualisation
    "visualize_architecture",
    "visualize_evolution",
    "visualize_input",
    # visualisation images COCO claires
    "colormap_mask",
    "visualize_coco_sample",
    "visualize_patches_rgb",
    # entraînement évolutif COCO
    "HierarchicalCOCO",
    # monitoring + rapport
    "TrainingTracker",
    "generate_report",
    # message passing (graphe de nœuds)
    "build_grid_adjacency",
    "conductivity_physarum",
    "message_passing_step",
    "spatial_consensus",
    "draw_architecture",
    "architecture_frame",
    "save_architecture_gif",
    # pipeline COCO classe par classe
    "train_class_by_class",
    "classify_patch",
    "accuracy_per_class",
    # données
    "filter_by_digits",
    "load_mnist",
    # classification non supervisée
    "AnchorNeurons",
    "DynamicAnchorNeurons",
    "dynamic_k",
    "homeostatic_threshold",
    "image_to_patches",
    "topdown_feedback",
    # couches stables / cycle respiratoire
    "RespiratoryController",
    "StructuralActivitySignal",
]
