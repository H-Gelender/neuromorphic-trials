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
from recherche_agi.modern_hopfield import (project_hopfield, surprise,
                                           oja_hopfield_update)


class HierarchicalCOCO:
    """Pipeline hiérarchique profond : couches empilées, neurones divisés,
    avec SKIP CONNECTIONS transversales auto-régulées (graphe petit-monde)."""

    def __init__(self, d_in=35, n_init=10, novelty_threshold=0.7,
                 max_neurons=10**9, surprise_plateau=0.002, plateau_window=300,
                 skip_active=True, beta=5.0, growth_scale=3.0):
        self.d_in = d_in
        self.beta = beta                  # inverse de température du MHN
        self.novelty_threshold = novelty_threshold
        self.growth_scale = growth_scale  # nb de neurones ajoutés par unité de surprise
        self.base_max = max_neurons          # inutilisé (aucun plafond)
        self.surprise_plateau = surprise_plateau  # (conservé)
        self.plateau_window = plateau_window
        self.stability_surprise = 0.3  # surprise sous laquelle la couche est "stable"
        self.layers = []                     # couches archivées
        self.layer = self._make_layer(0)     # couche courante
        self.history = {'n_neurons': [], 'n_layers': [], 'surprise': [],
                        'n_patches': [], 'skip_connections': []}
        self.n_patch = 0
        self._surprises = []                 # surprise récente (pour le plateau)
        self._layer_usage = []               # patches reçus par chaque couche (pour élagage)
        # skip connections (transversales)
        self.skip_active = skip_active
        self.skips = SkipConnections(max_connections=500, init_conductance=0.1,
                                     grow_rate=0.05, prune_threshold=0.02)
        # message passing d'entraînement : conductance Physarum entre nœuds
        self.mp_alpha = 0.3
        self.mp_conductance = None   # (n,n) réinitialisé par step_image

    def _make_layer(self, depth):
        """Crée une couche. AUCUN plafond : la croissance est libre et la
        stabilité est garantie par la convergence de la surprise (pas un max)."""
        # max_neurons très grand = pas de limite effective (croissance libre)
        return DynamicAnchorNeurons(
            d_in=self.d_in, n_init=10, seed=0, lr=0.1, use_homeostasis=True,
            novelty_threshold=self.novelty_threshold, max_neurons=10**9)

    def step(self, features, label):
        """Traite un patch via MHN. NEUROGENÈSE AGGRESSIVE + détection de stabilité.

        - Neurogenèse : ajoute PLUSIEURS neurones quand la surprise est élevée
          (n_add = int(S * growth_scale)), sans plafond fixe.
        - Stabilité : on suit la variation des prototypes; si la couche converge
          (prototypes stables), on crée une nouvelle couche.
        """
        zn = features / (np.linalg.norm(features) + 1e-8)
        if len(self.layer.W) == 0:
            S = 1.0
        else:
            S = float(surprise(zn, self.layer.W, beta=self.beta)[0])

        # apprentissage : plasticité Oja pondérée par z continu (MHN)
        self.layer.W = oja_hopfield_update(self.layer.W, zn, beta=self.beta,
                                           lr=self.layer.lr)[0]

        # --- NEUROGENÈSE : croissance libre, SANS PLAFOND ---
        # la neurogenèse est pilotée par la surprise. La stabilité est garantie
        # par la convergence : quand la surprise reste faible/stable (la couche
        # représente bien les données), on arrête de croître naturellement.
        n_add = int(S * self.growth_scale)
        if n_add > 0:
            for _ in range(n_add):
                self.layer._grow(zn, label)
        # au moins 1 neurone si la surprise dépasse le seuil de nouveauté
        if S > self.novelty_threshold:
            self.layer._grow(zn, label)
        self.layer.physarum_prune(0.01)

        # --- DÉTECTION DE STABILITÉ par plateau de surprise ---
        # la couche est stable quand la surprise ne descend plus (elle a appris
        # tout ce qu'elle pouvait) ET reste sous un seuil (elle représente bien
        # les données). Pas de minimum de neurones artificiel.
        self._surprises.append(S)
        if len(self._surprises) > self.plateau_window:
            self._surprises.pop(0)
        if len(self._surprises) >= self.plateau_window:
            half = self.plateau_window // 2
            mean_first = float(np.mean(self._surprises[:half]))
            mean_second = float(np.mean(self._surprises[half:]))
            # plateau : la surprise ne descend plus, et elle est raisonnablement basse
            if (mean_first - mean_second) < self.surprise_plateau \
               and mean_second < self.stability_surprise:
                self._spawn_layer()

        # --- ÉLAGAGE DES COUCHES INUTILES ---
        # une couche archivée qui ne reçoit plus de patches (inutile) disparaît
        self._prune_useless_layers()

        # --- SYNAPTOGENÈSE (skip connections) ---
        if self.skip_active and len(self.layers) >= 1:
            self._synaptogenesis(zn, S)

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
        # réactive les couches archivées (elles sont utiles si des patches passent)
        self._reactivate_layers()
        for i in range(n):
            self.step(zn[i], labels[i] if labels is not None else None)
        return surprise.mean()

    def _synaptogenesis(self, zn, surprise):
        """Pousse synaptique + validation : skip connections INTER (C1 -> couche
        courante) et INTRA-couche (neurones co-actifs d'une même couche)."""
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

        # --- INTER-CONNECTION : C1 -> couche courante (skip) ---
        conn = self.skips.add_candidate(0, w1, depth_cur, wc)
        resonance_flow = 1.0 / (1.0 + surprise)
        self.skips.reinforce(conn, resonance_flow)

        # --- INTRA-CONNECTION : connecter les neurones co-actifs de la couche
        #     courante (les 2 meilleurs gagnants sur ce patch) ---
        acts_cur = self.layer.W @ zn
        top2 = np.argsort(-acts_cur)[:2]
        if len(top2) >= 2:
            conn_intra = self.skips.add_candidate(depth_cur, int(top2[0]),
                                                  depth_cur, int(top2[1]))
            self.skips.reinforce(conn_intra, resonance_flow)

        # élagage périodique des skip connections inutiles
        if self.n_patch % 200 == 0:
            self.skips.prune()

    def _spawn_layer(self):
        """Gèle la couche courante et crée une couche plus profonde."""
        self.layers.append(self.layer)   # archive la couche
        self._layer_usage.append(1.0)    # usage initial de la nouvelle couche archivée
        depth = len(self.layers)
        self.layer = self._make_layer(depth)
        # réinitialiser le suivi de surprise
        self._surprises = []

    def _prune_useless_layers(self):
        """Élagage des couches inutiles (graphe Physarum).

        Une couche archivée qui ne reçoit plus de patches (son usage décroît)
        est supprimée. L'usage décroît à chaque step (atrophie Physarum) ;
        il est restauré quand la couche est réactivée (via step_image).
        """
        if not self.layers:
            return
        # décroissance de l'usage (les couches inutilisées s'atrophient)
        for i in range(len(self._layer_usage)):
            self._layer_usage[i] *= 0.99
        # supprimer les couches dont l'usage est tombé très bas (vraiment mortes)
        keep = []
        kept_usage = []
        for i, (layer, usage) in enumerate(zip(self.layers, self._layer_usage)):
            if usage > 0.001:   # seuil très bas : seules les couches réellement mortes disparaissent
                keep.append(layer)
                kept_usage.append(usage)
            else:
                # couche supprimée : retirer ses skip connections
                self.skips.connections = [c for c in self.skips.connections
                                          if c['to_layer'] != i and c['from_layer'] != i]
        self.layers = keep
        self._layer_usage = kept_usage

    def _reactivate_layers(self):
        """Réactive l'usage des couches archivées (appelé quand elles reçoivent
        des patches via les skip connections)."""
        for i in range(len(self._layer_usage)):
            self._layer_usage[i] = min(1.0, self._layer_usage[i] + 0.1)

    def should_stop(self, equilibrium_window=200, equilibrium_tol=0.005):
        """Condition de FIN : l'entraînement s'arrête à l'ÉQUILIBRE.

        Équilibre = la surprise moyenne ne descend plus (elle a convergé) sur
        une fenêtre. On compare la 1re et la 2e moitié de la fenêtre : si la
        différence est inférieure à equilibrium_tol, le modèle a appris tout ce
        qu'il pouvait -> arrêt.

        Retourne (bool, reason).
        """
        surp = self.history['surprise']
        if len(surp) < equilibrium_window:
            return False, 'volume_insuffisant'
        win = surp[-equilibrium_window:]
        half = equilibrium_window // 2
        mean_first = float(np.mean(win[:half]))
        mean_second = float(np.mean(win[half:]))
        if (mean_first - mean_second) < equilibrium_tol:
            return True, 'equilibre'
        return False, 'surprise_decroit_encore'

    def summary(self):
        return {
            'neurons': self.layer.n_neurons_current,
            'layers': len(self.layers) + 1,
            'archived': len(self.layers),
            'n_patches': self.n_patch,
            'layer_sizes': [len(l.W) for l in self.layers] + [self.layer.n_neurons_current],
            'skips': self.skips.summary(),
        }
