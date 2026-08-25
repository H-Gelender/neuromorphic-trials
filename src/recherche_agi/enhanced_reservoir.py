"""Réservoir synaptique amélioré — 3 leviers pour augmenter la discrimination de z.

1. RÉTINE ÉLECTRIQUE (champ récepteur multi-échelle) :
   filtre de contraste local (Difference of Gaussians / Laplacien de Gaussienne)
   appliqué AVANT l'injection dans le graphe. Amplifie les transitions
   (bords/traits), annule les aplats uniformes.

2. MULTI-INJECTION TEMPORELLE (saccades visuelles) :
   injection du fluide selon 2-3 axes successifs (haut->bas, gauche->droite,
   diagonales). La signature z = concaténation des réponses sous différentes
   "poussées" géométriques.

3. COMPÉTITION DENDRITIQUE (inhibition latérale) :
   le pooling applique une normalisation compétitive (softmax avec température)
   entre les zones dendritiques → signature z sparse et contrastée.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .physarum import grid_graph_from_image
from .synaptic_physarum import (hebbian_plasticity, synaptic_flow,
                                _zone_assignment)

__all__ = ["retina_filter", "multi_axis_signature", "competitive_pooling",
           "EnhancedReservoir"]


# --------------------------------------------------------------------------- #
# 1. Rétine électrique : Difference of Gaussians (contraste local)
# --------------------------------------------------------------------------- #
def retina_filter(img_np: np.ndarray, sigma_on: float = 1.0,
                  sigma_off: float = 2.5, k: float = 1.0) -> np.ndarray:
    """Difference of Gaussians (DoG) : centre-entourage.

    Retourne le contraste local normalisé. Les bords/traits sont amplifiés,
    les aplats annulés. L'injection dans le graphe utilise |DoG| (activité
    rétinienne) pour que les bords soient toujours des sources.
    """
    img = np.asarray(img_np, dtype=float).squeeze()
    # normaliser en [0,1] (MNIST est normalisé, peut être négatif)
    lo, hi = img.min(), img.max()
    img01 = (img - lo) / (hi - lo + 1e-8)

    center = ndi.gaussian_filter(img01, sigma_on)
    surround = ndi.gaussian_filter(img01, sigma_off)
    dog = center - k * surround
    # activité rétinienne : |DoG| (les deux polarités comptent), normalisée
    activity = np.abs(dog)
    norm = activity.max() + 1e-8
    return activity / norm


# --------------------------------------------------------------------------- #
# 2. Multi-injection temporelle (saccades visuelles)
# --------------------------------------------------------------------------- #
def _axis_sources(gh: int, gw: int, axis: str) -> np.ndarray:
    """Injections selon un axe géométrique (gradient directionnel).

    Crée un profil de pression directionnel sur la grille (haut->bas,
    gauche->droite, etc.) pour orienter le flux.
    """
    y = np.linspace(0, 1, gh).reshape(-1, 1)   # (gh, 1)
    x = np.linspace(0, 1, gw).reshape(1, -1)   # (1, gw)
    if axis == 'top_down':
        prof = np.broadcast_to(y, (gh, gw)).flatten()
    elif axis == 'bottom_up':
        prof = np.broadcast_to(1 - y, (gh, gw)).flatten()
    elif axis == 'left_right':
        prof = np.broadcast_to(x, (gh, gw)).flatten()
    elif axis == 'right_left':
        prof = np.broadcast_to(1 - x, (gh, gw)).flatten()
    elif axis == 'diag_tl_br':
        prof = (np.broadcast_to(y, (gh, gw)) + np.broadcast_to(x, (gh, gw))).flatten()
    elif axis == 'diag_bl_tr':
        prof = (np.broadcast_to(1 - y, (gh, gw)) + np.broadcast_to(x, (gh, gw))).flatten()
    else:
        raise ValueError(f"axe inconnu: {axis}")
    return prof / (prof.sum() + 1e-8)


def _signature_for_axis(img01: np.ndarray, axis: str, alpha: float, n_iter: int,
                        n_zones: int, downscale: int, eta: float, gamma: float,
                        beta: float, use_retina: bool, use_competition: bool,
                        temp: float) -> np.ndarray:
    """Signature z pour UN axe d'injection (avec rétine optionnelle)."""
    graph, sources0, info = grid_graph_from_image(img01, downscale=downscale)
    gh, gw = info['gh'], info['gw']
    n = graph.n_nodes
    centroids = np.array([[j + 0.5, i + 0.5] for i in range(gh) for j in range(gw)])

    # drains = bord
    border = set()
    for i in range(gh):
        border.add(i * gw); border.add(i * gw + gw - 1)
    for j in range(gw):
        border.add(j); border.add((gh - 1) * gw + j)
    sinks = sorted(border)[:min(10, len(border))]

    # injection = rétine (|DoG|) OU intensité brute
    if use_retina:
        # DoG sur l'image complète, puis ré-échantillonner sur la grille (blocs)
        # même formule que grid_graph_from_image : cell = 28//downscale
        retina_img = retina_filter(img01)
        cell_h, cell_w = max(1, 28 // downscale), max(1, 28 // downscale)
        sources = np.zeros(gh * gw)
        for i in range(gh):
            for j in range(gw):
                blk = retina_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                sources[i*gw + j] = blk.mean()
        sources = sources / (sources.sum() + 1e-8)
    else:
        sources = sources0.copy()

    # multi-axe : combiner l'injection d'image avec le gradient directionnel
    axis_prof = _axis_sources(gh, gw, axis)
    combined = sources * axis_prof
    combined = combined / (combined.sum() + 1e-8)

    for _ in range(n_iter):
        p, Q, _ = synaptic_flow(graph, combined, sinks, alpha)
        hebbian_plasticity(graph, p, eta, gamma, beta)

    # pooling (avec ou sans compétition dendritique)
    z = np.zeros(n_zones)
    node_zone = _zone_assignment(centroids, n_zones)
    for k, (i, j, _) in enumerate(graph.edges):
        zone = node_zone[i] if Q[k] >= 0 else node_zone[j]
        z[zone] += max(Q[k], 0.0)

    if use_competition:
        z = competitive_pooling(z, temp)
    else:
        z = z / (np.linalg.norm(z) + 1e-8)
    return z


# --------------------------------------------------------------------------- #
# 3. Compétition dendritique (inhibition latérale)
# --------------------------------------------------------------------------- #
def competitive_pooling(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Inhibition latérale : softmax avec température sur les zones.

    Si une zone est très active, elle supprime le bruit des zones voisines
    → signature sparse et contrastée.
    """
    z = np.asarray(z, dtype=float)
    # softmax : normalisation compétitive (somme à 1)
    ex = np.exp(z / temperature)
    zc = ex / (ex.sum() + 1e-8)
    # normaliser en norme unitaire pour la similarité cosinus
    return zc / (np.linalg.norm(zc) + 1e-8)


# --------------------------------------------------------------------------- #
# Signature multi-axe complète
# --------------------------------------------------------------------------- #
def multi_axis_signature(img_np: np.ndarray, axes=('top_down', 'left_right'),
                         alpha: float = 5.0, n_iter: int = 10, n_zones: int = 32,
                         downscale: int = 8, eta: float = 0.1, gamma: float = 0.1,
                         beta: float = 5.0, use_retina: bool = True,
                         use_competition: bool = True, temp: float = 1.0) -> np.ndarray:
    """Signature z multi-axe : concaténation des réponses sous plusieurs axes.

    Retourne un vecteur z de taille (n_axes × n_zones), normalisé globalement.
    """
    parts = []
    for axis in axes:
        z = _signature_for_axis(img_np, axis, alpha, n_iter, n_zones, downscale,
                                eta, gamma, beta, use_retina, use_competition, temp)
        parts.append(z)
    full = np.concatenate(parts)
    return full / (np.linalg.norm(full) + 1e-8)


# --------------------------------------------------------------------------- #
# Réservoir amélioré (interface compatible)
# --------------------------------------------------------------------------- #
class EnhancedReservoir:
    """Réservoir synaptique amélioré (rétine + multi-axe + compétition)."""

    def __init__(self, axes=('top_down', 'left_right'), alpha: float = 5.0,
                 n_iter: int = 10, n_zones: int = 32, downscale: int = 8,
                 eta: float = 0.1, gamma: float = 0.1, beta: float = 5.0,
                 use_retina: bool = True, use_competition: bool = True,
                 temp: float = 1.0):
        self.axes = list(axes)
        self.alpha = alpha
        self.n_iter = n_iter
        self.n_zones = n_zones
        self.downscale = downscale
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.use_retina = use_retina
        self.use_competition = use_competition
        self.temp = temp

    def signature(self, img_np: np.ndarray) -> np.ndarray:
        return multi_axis_signature(img_np, self.axes, self.alpha, self.n_iter,
                                    self.n_zones, self.downscale, self.eta,
                                    self.gamma, self.beta, self.use_retina,
                                    self.use_competition, self.temp)

    @property
    def n_features(self) -> int:
        return len(self.axes) * self.n_zones
