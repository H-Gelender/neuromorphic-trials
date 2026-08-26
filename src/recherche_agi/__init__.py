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

__all__ = [
    # scènes COCO (prototype de scale)
    "CocoScenes",
    "PALETTE",
    "STUFF_CLASSES",
    # caractéristiques de texture/couleur
    "color_texture_features",
    "extract_all_features",
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
