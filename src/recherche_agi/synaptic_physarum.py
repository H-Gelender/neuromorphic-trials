"""Physarum synaptique — des arêtes passives aux synapses actives.

Transforme le réseau Physarum en un réservoir de synapses actives :

  1. FLUX NON-LINÉAIRE :  Q_ij = D_ij * tanh(alpha * (p_i - p_j))
     Les petites variations de pression (bruit) sont étouffées, les fortes
     différences (vrais traits) sont amplifiées — seuil de déclenchement
     synaptique.

  2. PLASTICITÉ HEBBENNIENNE LOCALE (STDP-like) :
        D_ij += eta * (sigma(p_i) * sigma(p_j)) - gamma * D_ij
     Si deux nœuds sont sous pression ensemble, le tuyau s'élargit — extraction
     locale des corrélations de forme.

  3. INTÉGRATION DENDRITIQUE (pooling par sous-zones) :
        z_k = sum_{(i,j) in Zone_k} ReLU(Q_ij)
     Les flux sont regroupés par zones spatiales (8 ou 16) pour produire un
     vecteur compact et non linéaire, envoyé au Predictive Coding / couche lue.

Références :
- Tero et al. (2007), Physarum transport network.
- Plasticité Hebbienne : Hebb (1949) ; STDP : Bi & Poo (1998).
- Intégration dendritique : modèles biophysiques de sommation non linéaire.
"""
from __future__ import annotations

import numpy as np

from .physarum import PhysarumGraph, grid_graph_from_image

__all__ = ["synaptic_flow", "hebbian_plasticity", "dendritic_pooling",
           "synaptic_signature", "SynapticReservoir"]


# --------------------------------------------------------------------------- #
# 1. Flux non-linéaire (synapse)
# --------------------------------------------------------------------------- #
def synaptic_flow(graph: PhysarumGraph, sources: np.ndarray, sinks: list[int],
                  alpha: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Résout le flux avec activation synaptique non-linéaire.

    D'abord on résout les pressions par la loi de Poiseuille linéaire (le champ
    de pression est physique), puis on applique la NON-LINÉARITÉ sur chaque arête :
        Q_ij = D_ij * tanh(alpha * (p_i - p_j))

    Retourne (pressions, flux_nonlinéaires, flux_linéaires).
    """
    from .physarum import poiseuille_flow
    p, Q_lin = poiseuille_flow(graph, sources, sinks)

    # flux synaptique non linéaire par arête
    Q = np.zeros(graph.n_edges)
    for k, (i, j, _) in enumerate(graph.edges):
        delta = p[i] - p[j]
        Q[k] = graph.D[k] * np.tanh(alpha * delta)

    return p, Q, Q_lin


# --------------------------------------------------------------------------- #
# 2. Plasticité Hebbienne locale (STDP-like)
# --------------------------------------------------------------------------- #
def hebbian_plasticity(graph: PhysarumGraph, pressures: np.ndarray,
                       eta: float = 0.1, gamma: float = 0.1,
                       beta: float = 5.0) -> None:
    """Met à jour les conductances par co-activation Hebbienne.

        D_ij += eta * (sigma(p_i) * sigma(p_j)) - gamma * D_ij

    sigma est une sigmoïde (activation "sous pression"). Les nœuds fortement
    pressurisés ensemble renforcent leur tuyau.
    """
    def sigma(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-beta * x))

    for k, (i, j, _) in enumerate(graph.edges):
        co_act = sigma(pressures[i]) * sigma(pressures[j])
        graph.D[k] = graph.D[k] + eta * co_act - gamma * graph.D[k]
        graph.D[k] = max(graph.D[k], 1e-6)


# --------------------------------------------------------------------------- #
# 3. Intégration dendritique (pooling par sous-zones)
# --------------------------------------------------------------------------- #
def _zone_assignment(centroids: np.ndarray, n_zones: int) -> np.ndarray:
    """Assigne chaque nœud à une zone spatiale (grille sqrt(n_zones) x sqrt(n_zones))."""
    n_nodes = len(centroids)
    n_side = int(np.sqrt(n_zones))
    # normaliser les centroïdes en [0, n_side)
    cx = centroids[:, 0]
    cy = centroids[:, 1]
    x_bin = np.clip((cx / 28.0 * n_side).astype(int), 0, n_side - 1)
    y_bin = np.clip((cy / 28.0 * n_side).astype(int), 0, n_side - 1)
    return y_bin * n_side + x_bin   # zone id par nœud


def dendritic_pooling(graph: PhysarumGraph, Q: np.ndarray, centroids: np.ndarray,
                      n_zones: int = 16) -> np.ndarray:
    """Regroupe les flux par sous-zones spatiales (intégration dendritique).

        z_k = sum_{(i,j) in Zone_k} ReLU(Q_ij)

    Retourne le vecteur z de taille n_zones (compact et non linéaire).
    """
    node_zone = _zone_assignment(centroids, n_zones)
    z = np.zeros(n_zones)
    for k, (i, j, _) in enumerate(graph.edges):
        # une arête appartient à la zone de son nœud de plus forte pression
        zone = node_zone[i] if Q[k] >= 0 else node_zone[j]
        z[zone] += max(Q[k], 0.0)     # ReLU
    return z


# --------------------------------------------------------------------------- #
# Signature synaptique complète
# --------------------------------------------------------------------------- #
def synaptic_signature(img_np: np.ndarray, alpha: float = 5.0, n_iter: int = 10,
                       n_zones: int = 16, downscale: int = 14,
                       eta: float = 0.1, gamma: float = 0.1,
                       beta: float = 5.0) -> np.ndarray:
    """Signature synaptique d'une image : flux non linéaire + Hebbien + pooling.

    Retourne le vecteur z (n_zones) compact, après n_iter itérations de la
    dynamique synaptique (flux tanh + plasticité Hebbienne).
    """
    graph, sources, info = grid_graph_from_image(img_np, downscale=downscale)
    gh, gw = info['gh'], info['gw']
    centroids = np.array([[j + 0.5, i + 0.5] for i in range(gh) for j in range(gw)])

    # drains = bord
    border = set()
    for i in range(gh):
        border.add(i * gw); border.add(i * gw + gw - 1)
    for j in range(gw):
        border.add(j); border.add((gh - 1) * gw + j)
    sinks = sorted(border)[:min(10, len(border))]

    for _ in range(n_iter):
        p, Q, _ = synaptic_flow(graph, sources, sinks, alpha)
        hebbian_plasticity(graph, p, eta, gamma, beta)

    # pooling final sur le flux synaptique
    z = dendritic_pooling(graph, Q, centroids, n_zones)
    # normaliser
    norm = np.linalg.norm(z) + 1e-8
    return z / norm


# --------------------------------------------------------------------------- #
# Réservoir synaptique (interface)
# --------------------------------------------------------------------------- #
class SynapticReservoir:
    """Réservoir Physarum synaptique : image -> vecteur z compact.

    Chaque image est convertie en une signature z (pooling par zones) via le
    pipeline synaptique (flux tanh + plasticité Hebbienne). Le vecteur z est
    ensuite utilisé par le Predictive Coding / couche lue.
    """

    def __init__(self, alpha: float = 5.0, n_iter: int = 10, n_zones: int = 16,
                 downscale: int = 14, eta: float = 0.1, gamma: float = 0.1,
                 beta: float = 5.0):
        self.alpha = alpha
        self.n_iter = n_iter
        self.n_zones = n_zones
        self.downscale = downscale
        self.eta = eta
        self.gamma = gamma
        self.beta = beta

    def signature(self, img_np: np.ndarray) -> np.ndarray:
        return synaptic_signature(img_np, self.alpha, self.n_iter, self.n_zones,
                                  self.downscale, self.eta, self.gamma, self.beta)
