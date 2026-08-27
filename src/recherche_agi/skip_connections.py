"""Skip Connections auto-régulées — connexions transversales inter/intra-couches.

Transforme la pyramide rigide en graphe petit-monde (small-world) :
- Pousse synaptique : une connexion candidate C_k -> C_{k+m} se crée (conductance faible)
- Validation par la surprise : si la connexion réduit la surprise prédictive,
  le tube se renforce (Oja/Physarum) ; sinon il reste à faible flux
- Élagage Physarum : les connexions sans flux sont asséchées et supprimées

Support : connexions inter-couches (C_k -> C_j) et intra-couche (même couche).
"""
import numpy as np


class SkipConnections:
    """Gère les connexions transversales entre couches et neurones.

    Chaque connexion est un tube Physarum avec une conductance.
    structure : liste de dict {from_layer, from_neuron, to_layer, to_neuron, conductance}
    """

    def __init__(self, max_connections=10**9, init_conductance=0.1,
                 grow_rate=0.05, prune_threshold=0.01):
        self.max_connections = max_connections  # quasi illimité (pas de plafond)
        self.init_conductance = init_conductance
        self.grow_rate = grow_rate
        self.prune_threshold = prune_threshold
        self.connections = []       # liste de tubes

    def add_candidate(self, from_layer, from_neuron, to_layer, to_neuron):
        """Étape A : pousse synaptique (connexion candidate à faible conductance).
        AUCUN plafond : le nombre est régulé par l'élagage Physarum (prune)."""
        # éviter les doublons
        for c in self.connections:
            if (c['from_layer'], c['from_neuron'], c['to_layer'], c['to_neuron']) == \
               (from_layer, from_neuron, to_layer, to_neuron):
                return c
        conn = {'from_layer': from_layer, 'from_neuron': from_neuron,
                'to_layer': to_layer, 'to_neuron': to_neuron,
                'conductance': self.init_conductance, 'flow': 0.0}
        self.connections.append(conn)
        return conn

    def reinforce(self, conn, resonance_flow, lr=0.1):
        """Étape B : validation par la surprise. Le flux de résonance renforce le tube."""
        if conn is None:
            return
        conn['flow'] += resonance_flow
        # renforcement Oja : la conductance croît avec le flux, saturée
        conn['conductance'] = min(1.0, conn['conductance'] + lr * resonance_flow)

    def flow_through(self, from_layer, from_neuron, to_layer, to_neuron):
        """Flux total des connexions entre deux neurones (0 si aucune)."""
        total = 0.0
        for c in self.connections:
            if (c['from_layer'], c['from_neuron']) == (from_layer, from_neuron) and \
               (c['to_layer'], c['to_neuron']) == (to_layer, to_neuron):
                total += c['conductance']
        return total

    def prune(self):
        """Étape C : élagage Physarum — supprime les tubes à faible flux."""
        kept = []
        n_removed = 0
        for c in self.connections:
            if c['flow'] < self.prune_threshold:
                n_removed += 1
                continue  # asséché et supprimé
            kept.append(c)
            c['flow'] *= 0.9   # dissipation du flux (régulation)
        self.connections = kept
        return n_removed

    def connections_between_layers(self, layer_a, layer_b):
        """Connexions inter-couches entre deux couches données."""
        return [c for c in self.connections
                if {c['from_layer'], c['to_layer']} == {layer_a, layer_b}]

    def summary(self):
        return {
            'n_connections': len(self.connections),
            'inter_layer': sum(1 for c in self.connections if c['from_layer'] != c['to_layer']),
            'intra_layer': sum(1 for c in self.connections if c['from_layer'] == c['to_layer']),
        }
