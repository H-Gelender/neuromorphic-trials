"""Entraînement évolutif COCO : neurogenèse + cycle respiratoire.

Le modèle ÉVOLUE pendant l'entraînement :
- DynamicAnchorNeurons : ajoute des neurones quand la surprise est élevée
- RespiratoryController : gèle/ajoute des couches quand l'oscillation apparaît
- Le suivi enregistre l'évolution de l'architecture (nb neurones, nb couches)
"""
import numpy as np
from recherche_agi import (DynamicAnchorNeurons, RespiratoryController, dynamic_k)


class EvolutiveCOCO:
    """Pipeline d'entraînement avec architecture dynamique."""

    def __init__(self, d_in=35, n_init=10, novelty_threshold=0.5,
                 max_neurons=1000, surprise_unfreeze=2.0):
        self.d_in = d_in
        self.layer = DynamicAnchorNeurons(
            d_in=d_in, n_init=n_init, seed=0, lr=0.1,
            use_homeostasis=True, novelty_threshold=novelty_threshold,
            max_neurons=max_neurons)
        self.controller = RespiratoryController(
            surprise_unfreeze=surprise_unfreeze, unfreeze_streak=5, base_lr=0.1)
        self.history = {'n_neurons': [], 'n_layers': [], 'surprise': [],
                        'n_patches': [], 'phase': []}
        self.n_patch = 0

    def step(self, features, label):
        """Traite un patch : apprend, met à jour le contrôleur, enregistre."""
        zn = features / (np.linalg.norm(features) + 1e-8)
        # surprise de reconstruction
        sim = self.layer.W @ zn
        w = int(np.argmax(sim))
        S = float(np.linalg.norm(zn - self.layer.W[w])**2)
        # apprentissage avec neurogenèse (croissance si surprise élevée)
        self.layer.learn(zn, k=dynamic_k(0.5, 1, 5), label=label)
        # élagage + contrôleur
        self.layer.physarum_prune(0.05)
        self.controller.record_create(1 if self.layer.n_neurons_current > len(self.history['n_neurons']) else 0)
        self.controller.tick()
        # si oscillation -> gel/spawn de couche
        if self.controller.signal.detect_oscillation()['oscillating'] and not self.controller.frozen:
            self.controller.spawn_layer()

        self.n_patch += 1
        self.history['n_neurons'].append(self.layer.n_neurons_current)
        self.history['n_layers'].append(self.controller.layer_count)
        self.history['surprise'].append(S)
        self.history['n_patches'].append(self.n_patch)
        self.history['phase'].append(self.controller.phase)
        return S

    def summary(self):
        return {
            'neurons': self.layer.n_neurons_current,
            'layers': self.controller.layer_count,
            'frozen': self.controller.frozen,
            'phase': self.controller.phase,
            'n_patches': self.n_patch,
        }
