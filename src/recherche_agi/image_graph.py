"""Conversion image -> graphe par superpixels (Felzenszwalb).

Chaque superpixel devient un nœud du graphe ; les superpixels adjacents sont
reliés par des arêtes. Cette méthode s'adapte naturellement aux formes de
l'image (au lieu d'imposer une grille régulière comme SLIC).

Les images MNIST sont normalisées (valeurs négatives) : on les ramène en [0,1]
avant la segmentation. La luminosité d'un superpixel peut servir d'injection
pour le solveur Physarum.
"""
from __future__ import annotations

import numpy as np
from skimage.segmentation import felzenszwalb

__all__ = ["image_to_graph", "superpixels", "physarum_from_image"]


def superpixels(image: np.ndarray, scale: float = 100, sigma: float = 0.5,
                min_size: int = 5) -> tuple[np.ndarray, int]:
    """Segmente une image en superpixels Felzenszwalb.

    Args:
        image: tableau 2D (H, W) en niveaux de gris (peut être normalisé).
        scale, sigma, min_size: paramètres Felzenszwalb.

    Returns:
        (segments, n_labels) : carte des segments + nombre de superpixels.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        img = img.squeeze()
    # normaliser en [0,1] (nécessaire car MNIST est normalisé, peut être négatif)
    lo, hi = img.min(), img.max()
    img_01 = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    segments = felzenszwalb(img_01, scale=scale, sigma=sigma, min_size=min_size)
    return segments, int(segments.max()) + 1


def image_to_graph(image: np.ndarray, scale: float = 100, sigma: float = 0.5,
                   min_size: int = 5) -> dict:
    """Convertit une image en graphe de superpixels.

    Retourne un dict :
        segments   : carte des segments (H, W)
        n_nodes    : nombre de superpixels (nœuds)
        edges      : liste de paires (i, j) de nœuds adjacents
        centroids  : (n_nodes, 2) positions (x, y) des nœuds
        intensities: (n_nodes,) luminosité moyenne de chaque superpixel
        n_edges    : nombre d'arêtes
    """
    segments, n_nodes = superpixels(image, scale, sigma, min_size)

    # centroïdes + intensité moyenne par superpixel
    centroids = np.zeros((n_nodes, 2))
    intensities = np.zeros(n_nodes)
    img = np.asarray(image).squeeze()
    lo, hi = img.min(), img.max()
    img_01 = (img - lo) / (hi - lo + 1e-8)
    for lab in range(n_nodes):
        ys, xs = np.where(segments == lab)
        centroids[lab] = (xs.mean(), ys.mean())
        intensities[lab] = img_01[ys, xs].mean()

    # arêtes : superpixels adjacents (4-voisinage)
    edges = set()
    h, w = segments.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            l = segments[y, x]
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nl = segments[y + dy, x + dx]
                if nl != l:
                    edges.add(tuple(sorted((int(l), int(nl)))))
    edges = sorted(edges)

    return {
        'segments': segments,
        'n_nodes': n_nodes,
        'n_edges': len(edges),
        'edges': edges,
        'centroids': centroids,
        'intensities': intensities,
    }


def physarum_from_image(image: np.ndarray, scale: float = 100, sigma: float = 0.5,
                        min_size: int = 5, mu: float = 1.0, delta: float = 1.0,
                        D_init: float = 0.5, with_weights: bool = True):
    """Construit un PhysarumGraph à partir d'une image (superpixels = nœuds).

    - Nœuds : superpixels (Felzenszwalb).
    - Arêtes : adjacence spatiale, longueur = distance euclidienne des centroïdes.
    - Injections : intensité lumineuse de chaque superpixel (normalisée).

    Retourne (graph, sources, info).
    """
    from .physarum import PhysarumGraph

    g = image_to_graph(image, scale, sigma, min_size)
    centroids = g['centroids']

    # arêtes pondérées par la distance entre centroïdes
    edges = []
    for (a, b) in g['edges']:
        d = float(np.linalg.norm(centroids[a] - centroids[b]))
        edges.append((a, b, max(d, 1e-3)))

    graph = PhysarumGraph(g['n_nodes'], edges, mu=mu, delta=delta, D_init=D_init)

    # injections = intensités (>= 0), normalisées (conservation de masse)
    sources = np.clip(g['intensities'], 0, None).astype(float)
    total = sources.sum()
    if total > 0:
        sources = sources / total
    else:
        sources = np.full(g['n_nodes'], 1.0 / g['n_nodes'])

    info = {'segments': g['segments'], 'centroids': centroids, 'n_nodes': g['n_nodes']}
    return graph, sources, info
