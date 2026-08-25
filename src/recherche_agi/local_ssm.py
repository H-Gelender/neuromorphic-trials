"""SSM local — l'équation d'intégration temporelle sous chaque nœud/tuyau.

L'idée : chaque nœud (ou tuyau) du Physarum porte une MÉMOIRE RÉCURRENTE h_t
évoluant selon l'équation d'état la plus élémentaire :

    h_t = (1 - Δ_t) · h_{t-1} + Δ_t · x_t

où :
- h_t : mémoire contextuelle locale du nœud/tuyau à l'instant t
- x_t : signal entrant
- Δ_t : constante de temps (pas d'intégration)

La SURPRISE S_t du Predictive Coding pilote Δ_t de façon 100% non supervisée :

    Δ_t = σ(S_t) = σ(||z_t - ẑ_t||)

- S ≈ 0 (prévisible)  → Δ → 0 : le SSM garde sa mémoire (h_t ≈ h_{t-1}), inertie.
- S ≫ 0 (rupture)     → Δ → 1 : la surprise réinitialise la mémoire (h_t ≈ x_t).

Ainsi :
- Physarum synaptique → structure et espace (corrélations spatiales)
- Micro-SSM sous le nœud → temps et contexte (inertie temporelle)
- Surprise (PC) → vitesse à laquelle le temps s'écoule et se réinitialise
"""
from __future__ import annotations

import numpy as np

__all__ = ["LocalSSM", "surprise_to_delta", "SSMLayer"]


def surprise_to_delta(S: float, beta: float = 3.0) -> float:
    """Constante de temps Δ = σ(S) (sigmoïde de la surprise).

    S≈0 → Δ≈0 (mémoire gardée) ; S≫0 → Δ≈1 (mémoire réinitialisée).
    """
    return 1.0 / (1.0 + np.exp(-beta * S))


class LocalSSM:
    """SSM local : une variable d'état récurrente par nœud/tuyau.

    Pour un vecteur de N nœuds :
        h_t = (1 - Δ_t) · h_{t-1} + Δ_t · x_t

    Δ_t peut être scalaire (même surprise partout) ou un vecteur (par nœud).
    """

    def __init__(self, n_nodes: int, init: float = 0.0):
        self.n_nodes = n_nodes
        self.state = np.full(n_nodes, init, dtype=float)
        self.history = []

    def step(self, x: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
        """Intègre une entrée x avec la constante de temps Δ."""
        x = np.asarray(x, dtype=float)
        d = np.asarray(delta, dtype=float)
        # broadcasting : Δ scalaire ou (n_nodes,)
        self.state = (1 - d) * self.state + d * x
        self.history.append(self.state.copy())
        return self.state.copy()

    def reset(self):
        self.state = np.full(self.n_nodes, 0.0, dtype=float)
        self.history = []


class SSMLayer:
    """Couche SSM intégrée à un réservoir : mémoire temporelle par nœud.

    À chaque image, on :
      1. calcule la surprise S (fournie) → Δ = σ(S)
      2. intègre les activations du graphe dans le SSM local
      3. utilise l'état h_t comme signature (mémoire temporelle)

    Le paramètre `surprise_callback` fournit S_t pour chaque image.
    """

    def __init__(self, n_nodes: int, surprise_callback=None, beta: float = 3.0,
                 init: float = 0.0):
        self.ssm = LocalSSM(n_nodes, init)
        self.surprise_callback = surprise_callback  # callable -> S
        self.beta = beta
        self.last_delta = 0.0
        self.last_surprise = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Intègre x dans la mémoire SSM, pilotée par la surprise.

        Retourne l'état h_t (mémoire temporelle fusionnée).
        """
        S = self.surprise_callback(x) if self.surprise_callback else 0.0
        delta = surprise_to_delta(S, self.beta)
        self.last_surprise = S
        self.last_delta = delta
        return self.ssm.step(x, delta)

    def reset(self):
        self.ssm.reset()
        self.last_delta = 0.0
        self.last_surprise = 0.0
