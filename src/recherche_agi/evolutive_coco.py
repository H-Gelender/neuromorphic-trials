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
from recherche_agi.skip_connections import SkipConnections
from recherche_agi.message_passing import (build_grid_adjacency,
                                           message_passing_train,
                                           update_physarum_conductance)


class HierarchicalCOCO:
    """Pipeline hiérarchique profond : couches empilées, neurones divisés,
    avec SKIP CONNECTIONS transversales auto-régulées (graphe petit-monde)."""

    def __init__(self, d_in=35, n_init=10, novelty_threshold=0.7,
                 max_neurons=2000, surprise_plateau=0.002, plateau_window=300,
                 min_neurons_before_spawn=50, skip_active=True):
        self.d_in = d_in
        self.novelty_threshold = novelty_threshold
        self.base_max = max_neurons          # C1 = 2000
        self.surprise_plateau = surprise_plateau  # si la surprise ne descend plus -> spawn
        self.plateau_window = plateau_window
        self.min_neurons_before_spawn = min_neurons_before_spawn
        self.layers = []                     # couches archivées
        self.layer = self._make_layer(0)     # couche courante
        self.history = {'n_neurons': [], 'n_layers': [], 'surprise': [],
                        'n_patches': [], 'skip_connections': []}
        self.n_patch = 0
        self._surprises = []                 # surprise récente (pour le plateau)
        # skip connections (transversales)
        self.skip_active = skip_active
        self.skips = SkipConnections(max_connections=500, init_conductance=0.1,
                                     grow_rate=0.05, prune_threshold=0.02)
        # message passing d'entraînement : conductance Physarum entre nœuds
        self.mp_alpha = 0.3
        self.mp_conductance = None   # (n,n) réinitialisé par step_image

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

        # --- SYNAPTOGENÈSE (skip connections) ---
        # si des couches profondes existent, on tisse des connexions candidates
        # depuis C1 (détails fins) vers la couche courante, validées par la surprise.
        if self.skip_active and len(self.layers) >= 1:
            self._synaptogenesis(zn, S)

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
        self.history['skip_connections'].append(self.skips.summary()['n_connections'])
        return S

    def step_image(self, patches, labels, gh, gw):
        """Traite TOUS les patches d'une image ensemble, avec MESSAGE PASSING
        de structuration pendant l'apprentissage.

        patches : (n, d) features des patches
        gh, gw  : dimensions spatiales de la grille (les voisins spatiaux)
        """
        n = len(patches)
        if n == 0:
            return
        zn = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
        # grille d'adjacence (4-voisins spatiaux), bornée au nombre réel de patches
        adj = build_grid_adjacency(gh, gw, n=n)
        # conductance Physarum initialisée
        self.mp_conductance = np.ones((n, n)) * 0.3
        # activations + surprise de chaque patch sur la couche courante
        acts = zn @ self.layer.W.T            # (n, n_neurons)
        winners = np.argmax(acts, axis=1)
        surprise = np.array([np.linalg.norm(zn[i] - self.layer.W[winners[i]])**2
                             for i in range(n)])
        # --- MESSAGE PASSING DE STRUCTURATION ---
        # consensus local + inhibition : les activations sont lissées
        lissed = message_passing_train(acts, surprise, adj, self.mp_conductance,
                                       alpha=self.mp_alpha)
        # mise à jour des tubes Physarum selon la co-activation
        self.mp_conductance = update_physarum_conductance(
            self.mp_conductance, lissed, adj, lr=0.03, decay=0.98)
        # --- APPRENTISSAGE : chaque patch apprend avec son activation lissée ---
        # le consensus influence le choix du gagnant (structuration stable)
        for i in range(n):
            self.step(zn[i], labels[i] if labels is not None else None)
        return surprise.mean()

    def _synaptogenesis(self, zn, surprise):
        """Pousse synaptique + validation : tisse des skip connections C1 -> couche
        courante, renforce celles qui réduisent la surprise, élague les inutiles."""
        depth_cur = len(self.layers)          # index de la couche courante
        if depth_cur == 0:
            return
        c1 = self.layers[0]                   # couche basse (détails fins)
        if len(c1.W) == 0 or len(self.layer.W) == 0:
            return
        # neurone gagnant de C1 sur ce patch
        w1 = int(np.argmax(c1.W @ zn))
        # neurone gagnant de la couche courante
        wc = int(np.argmax(self.layer.W @ zn))
        # pousse synaptique C1 -> couche courante (skip), sauf si déjà à la couche 1
        if depth_cur > 0:
            conn = self.skips.add_candidate(0, w1, depth_cur, wc)
            # validation : un flux de résonance = 1/(1+surprise) — si la connexion
            # aide (surprise faible), le flux est élevé -> renforcement
            resonance_flow = 1.0 / (1.0 + surprise)
            self.skips.reinforce(conn, resonance_flow)
        # élagage périodique des skip connections inutiles
        if self.n_patch % 200 == 0:
            self.skips.prune()

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
            'skips': self.skips.summary(),
        }
