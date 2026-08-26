"""Entraînement évolutif COCO — HIÉRARCHIE PROFONDE.

Stratégie : privilégier la PROFONDEUR plutôt que la largeur.
- Couche 1 : bridée à max_neurons (ex 2000)
- Chaque couche suivante a un nombre de neurones DIVISÉ (C2 = /2, C3 = /4...)
- Le modèle choisit le nombre de neurones par couche (neurogenèse)
- DÉFI : faire créer de nouvelles couches quand la couche actuelle stagne
  (taux de croissance qui chute / surprise qui ne descend plus)
"""
import numpy as np
from recherche_agi import DynamicAnchorNeurons, dynamic_k


class HierarchicalCOCO:
    """Pipeline hiérarchique profond : couches empilées, neurones divisés."""

    def __init__(self, d_in=35, n_init=10, novelty_threshold=0.7,
                 max_neurons=2000, surprise_plateau=0.002, plateau_window=300,
                 min_neurons_before_spawn=50):
        self.d_in = d_in
        self.novelty_threshold = novelty_threshold
        self.base_max = max_neurons          # C1 = 2000
        self.surprise_plateau = surprise_plateau  # si la surprise ne descend plus -> spawn
        self.plateau_window = plateau_window
        self.min_neurons_before_spawn = min_neurons_before_spawn
        self.layers = []                     # couches archivées
        self.layer = self._make_layer(0)     # couche courante
        self.history = {'n_neurons': [], 'n_layers': [], 'surprise': [],
                        'n_patches': []}
        self.n_patch = 0
        self._surprises = []                 # surprise récente (pour le plateau)

    def _make_layer(self, depth):
        """Crée une couche à la profondeur donnée (neurones max divisés par 2^depth)."""
        max_n = max(20, self.base_max // (2 ** depth))
        # n_init petit (la neurogenèse fait croître), jamais au max
        return DynamicAnchorNeurons(
            d_in=self.d_in, n_init=10, seed=0, lr=0.1, use_homeostasis=True,
            novelty_threshold=self.novelty_threshold, max_neurons=max_n)

    def step(self, features, label):
        """Traite un patch. Spawn quand la surprise ne descend plus (plateau)."""
        zn = features / (np.linalg.norm(features) + 1e-8)
        if len(self.layer.W) == 0:
            S = 1.0
        else:
            w = int(np.argmax(self.layer.W @ zn))
            S = float(np.linalg.norm(zn - self.layer.W[w])**2)

        # apprentissage + neurogenèse
        self.layer.learn(zn, k=dynamic_k(0.5, 1, 3), label=label)
        self.layer.physarum_prune(0.02)

        # --- DÉFI : SPAWN PAR PLATEAU DE SURPRISE ---
        # si la surprise moyenne sur une fenêtre ne descend plus (variation < seuil),
        # la couche actuelle a atteint ses limites -> on crée une couche plus profonde.
        self._surprises.append(S)
        if len(self._surprises) > self.plateau_window:
            self._surprises.pop(0)
        if len(self._surprises) >= self.plateau_window:
            half = self.plateau_window // 2
            mean_first = float(np.mean(self._surprises[:half]))
            mean_second = float(np.mean(self._surprises[half:]))
            # plateau : la 2e moitié ne descend plus (ou remonte)
            # ET il faut que la couche ait assez de neurones (pas de spawn prématuré)
            if (mean_first - mean_second) < self.surprise_plateau \
               and self.layer.n_neurons_current >= self.min_neurons_before_spawn:
                self._spawn_layer()

        self.n_patch += 1
        self.history['n_neurons'].append(self.layer.n_neurons_current)
        self.history['n_layers'].append(len(self.layers) + 1)
        self.history['surprise'].append(S)
        self.history['n_patches'].append(self.n_patch)
        return S

    def _spawn_layer(self):
        """Gèle la couche courante et crée une couche plus profonde (neurones /2)."""
        self.layers.append(self.layer)   # archive la couche
        depth = len(self.layers)
        self.layer = self._make_layer(depth)
        # réinitialiser le suivi de surprise
        self._surprises = []

    def summary(self):
        return {
            'neurons': self.layer.n_neurons_current,
            'layers': len(self.layers) + 1,
            'archived': len(self.layers),
            'n_patches': self.n_patch,
            'layer_sizes': [len(l.W) for l in self.layers] + [self.layer.n_neurons_current],
        }
