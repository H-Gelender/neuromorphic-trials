"""Entraînement évolutif COCO : neurogenèse + cycle respiratoire.

Le modèle ÉVOLUE pendant l'entraînement :
- DynamicAnchorNeurons : ajoute des neurones quand la surprise est élevée
- RespiratoryController : gèle/ajoute des couches quand l'oscillation apparaît
- Le suivi enregistre l'évolution de l'architecture (nb neurones, nb couches)
"""
import numpy as np
from recherche_agi import (DynamicAnchorNeurons, RespiratoryController, dynamic_k)


class EvolutiveCOCO:
    """Pipeline d'entraînement avec architecture dynamique.

    Comportement corrigé :
    - Neurogenèse SÉLECTIVE : n'ajoute un neurone que si le patch est mal
      représenté (similarité max < seuil_nouveaute) — évite l'explosion.
    - Spawn de couche quand la couche est SATURÉE (proche du plafond) : la C1
      gelée, une C2 est créée pour absorber la nouveauté.
    """

    def __init__(self, d_in=35, n_init=10, novelty_threshold=0.7,
                 max_neurons=300, surprise_unfreeze=2.0,
                 saturation_frac=0.85):
        self.d_in = d_in
        self.novelty_threshold = novelty_threshold
        self.max_neurons = max_neurons
        self.saturation_frac = saturation_frac
        self.layer = DynamicAnchorNeurons(
            d_in=d_in, n_init=n_init, seed=0, lr=0.1,
            use_homeostasis=True, novelty_threshold=novelty_threshold,
            max_neurons=max_neurons)
        self.controller = RespiratoryController(
            surprise_unfreeze=surprise_unfreeze, unfreeze_streak=5, base_lr=0.1)
        self.history = {'n_neurons': [], 'n_layers': [], 'surprise': [],
                        'n_patches': [], 'phase': []}
        self.n_patch = 0
        self.layers = []   # couches empilées (chaque couche = un DynamicAnchorNeurons)

    def _best_similarity(self, zn):
        """Similarité max du patch avec les neurones existants."""
        if len(self.layer.W) == 0:
            return 0.0
        sim = self.layer.W @ zn
        return float(sim.max())

    def step(self, features, label):
        """Traite un patch : apprentissage sélectif + évolution d'architecture."""
        zn = features / (np.linalg.norm(features) + 1e-8)
        S = float(np.linalg.norm(zn - self.layer.W[int(np.argmax(self.layer.W @ zn))])**2)

        # --- NEUROGENÈSE SÉLECTIVE : ajouter un neurone seulement si le patch
        #     est mal représenté (similarité < seuil). Pas à chaque patch. ---
        best_sim = self._best_similarity(zn)
        n_before = self.layer.n_neurons_current
        self.layer.learn(zn, k=dynamic_k(0.5, 1, 3), label=label)
        # l'apprentissage normal ajuste les prototypes existants
        # on ne déclenche la neurogenèse que si le patch était vraiment nouveau
        # (gérée par DynamicAnchorNeurons.learn avec novelty_threshold élevé)

        # --- ÉLAGAGE Physarum modéré (compense la croissance) ---
        self.layer.physarum_prune(0.03)

        # --- suivi pour le contrôleur ---
        grew = self.layer.n_neurons_current > n_before
        self.controller.record_create(1 if grew else 0)

        # --- SPAWN DE COUCHE par SATURATION : si la couche atteint ~85% du
        #     plafond, on gèle C1 et on crée une C2 pour absorber la nouveauté ---
        if self.layer.n_neurons_current >= self.saturation_frac * self.max_neurons \
           and len(self.layers) == 0:
            # geler C1 (fin de sa croissance), créer C2
            self.controller.spawn_layer()
            # C2 = nouvelle couche qui apprend sur les représentations de C1
            self.layers.append(self.layer)   # archive C1
            self.layer = DynamicAnchorNeurons(
                d_in=self.d_in, n_init=10, seed=0, lr=0.1,
                use_homeostasis=True, novelty_threshold=self.novelty_threshold,
                max_neurons=self.max_neurons)

        self.n_patch += 1
        self.history['n_neurons'].append(self.layer.n_neurons_current)
        self.history['n_layers'].append(self.controller.layer_count)
        self.history['surprise'].append(S)
        self.history['n_patches'].append(self.n_patch)
        self.history['phase'].append(self.controller.phase)
        return S

    def snapshot(self):
        """Retourne une COPIE du modèle (état figé à ce moment) pour le monitoring."""
        import copy
        return copy.deepcopy(self.layer)

    def summary(self):
        return {
            'neurons': self.layer.n_neurons_current,
            'layers': self.controller.layer_count,
            'archived_layers': len(self.layers),
            'frozen': self.controller.frozen,
            'phase': self.controller.phase,
            'n_patches': self.n_patch,
        }
