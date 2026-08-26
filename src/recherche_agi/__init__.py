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
from .data import filter_by_digits, load_mnist
from .unsupervised import (AnchorNeurons, dynamic_k, homeostatic_threshold,
                           image_to_patches, topdown_feedback)

__all__ = [
    # données
    "filter_by_digits",
    "load_mnist",
    # classification non supervisée
    "AnchorNeurons",
    "dynamic_k",
    "homeostatic_threshold",
    "image_to_patches",
    "topdown_feedback",
]
