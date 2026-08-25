"""Physarum solver — réseau de tubes adaptatif sans neurones.

Implémente le modèle de Tero et al. (2007) "A mathematical model for adaptive
transport network in path finding by true slime mold", Journal of Theoretical
Biology, et son interprétation lagrangienne (Solé & Pla-Mauri 2025, arXiv
2511.08531).

Loi de Poiseuille :   Q_ij = (D_ij / L_ij) * (p_i - p_j)
Conservation Kirchhoff : sum_j Q_ij = S_i
Adaptation tubes :   dD_ij/dt = |Q_ij|^mu - delta * D_ij

Pour la classification : les pixels de l'image deviennent des injections de
pression S_i, le flux circule, et la classe prédite est le drain qui récolte
le plus de flux.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PhysarumGraph", "poiseuille_flow", "adapt_conductances",
           "run_physarum", "classify_by_drainage"]


class PhysarumGraph:
    """Un réseau Physarum sur un graphe.

    - nodes: liste d'identifiants de nœuds.
    - edges: liste de (i, j, length) — arêtes avec leur longueur L_ij.
    - Conductances D_ij initialisées et mises à jour par la dynamique.
    """

    def __init__(self, n_nodes: int, edges: list[tuple[int, int, float]],
                 mu: float = 1.0, delta: float = 1.0, D_init: float = 0.5):
        self.n_nodes = n_nodes
        self.mu = mu
        self.delta = delta
        self.edges = []          # (i, j, L_ij)
        self.D = []              # conductance par arête
        for i, j, L in edges:
            self.add_edge(i, j, L, D_init)

    def add_edge(self, i: int, j: int, length: float, D: float = None):
        self.edges.append((int(i), int(j), float(length)))
        self.D.append(float(D) if D is not None else 0.5)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def laplacian(self) -> np.ndarray:
        """Matrice laplacienne pondérée (avec conductances D / L)."""
        L = np.zeros((self.n_nodes, self.n_nodes))
        for (i, j, l), D in zip(self.edges, self.D):
            w = D / l if l > 0 else D
            L[i, j] += -w
            L[j, i] += -w
            L[i, i] += w
            L[j, j] += w
        return L


def poiseuille_flow(graph: PhysarumGraph, sources: np.ndarray,
                    sinks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Résout le flux stationnaire (équations 2-3 de Tero).

    sources: vecteur S_i des injections (sum = 0). Les nœuds de `sinks` sont
    mis à la terre (p = 0) pour fixer la jauge.

    Retourne (pressions p, flux Q sur les arêtes).
    """
    n = graph.n_nodes
    Lap = graph.laplacian()

    # fixer la jauge : forcer p[sink] = 0 pour les nœuds de mise à la terre
    grounded = set(sinks)
    if not grounded:
        grounded = {0}

    free = [i for i in range(n) if i not in grounded]
    b = sources.copy()

    # résoudre sur les nœuds libres
    Lap_ff = Lap[np.ix_(free, free)]
    b_f = b[free]
    # Kirchhoff : pour les nœuds ground, on retire la colonne (p=0)
    p = np.zeros(n)
    try:
        p[free] = np.linalg.solve(Lap_ff, b_f)
    except np.linalg.LinAlgError:
        # matrice singulière (graphe non connexe) → pseudo-inverse
        p[free] = np.linalg.pinv(Lap_ff) @ b_f

    # flux sur chaque arête : Q_ij = D_ij/L_ij * (p_i - p_j)
    Q = np.zeros(graph.n_edges)
    for k, (i, j, l) in enumerate(graph.edges):
        w = graph.D[k] / l if l > 0 else graph.D[k]
        Q[k] = w * (p[i] - p[j])
    return p, Q


def adapt_conductances(graph: PhysarumGraph, Q: np.ndarray,
                       dt: float = 0.1) -> None:
    """Met à jour les conductances selon l'équation 4 de Tero.

    dD_ij/dt = |Q_ij|^mu - delta * D_ij  →  Euler explicite.
    """
    for k in range(graph.n_edges):
        growth = abs(Q[k]) ** graph.mu - graph.delta * graph.D[k]
        graph.D[k] = max(graph.D[k] + dt * growth, 1e-6)


def run_physarum(graph: PhysarumGraph, sources: np.ndarray, sinks: list[int],
                 n_iter: int = 50, dt: float = 0.1, verbose: bool = False
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Itère la dynamique complète : flux stationnaire + adaptation des tubes.

    Retourne (pressions finales, flux finaux). Le réseau "apprend" la structure
    en renforçant les tubes utilisés et en laissant dépérir les autres.
    """
    Q_final = None
    p_final = None
    for it in range(n_iter):
        p, Q = poiseuille_flow(graph, sources, sinks)
        adapt_conductances(graph, Q, dt)
        Q_final, p_final = Q, p
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{n_iter} | flux max {abs(Q).max():.3f} | "
                  f"conductances {np.mean(graph.D):.3f}")
    return p_final, Q_final


def classify_by_drainage(graph: PhysarumGraph, sources: np.ndarray,
                         sinks: list[int], n_iter: int = 50, dt: float = 0.1,
                         ) -> tuple[int, np.ndarray]:
    """Classifie en renvoyant l'indice du drain récoltant le plus de flux.

    Les nœuds `sinks` correspondent à des classes. Chaque drain est mis à la
    terre (p=0) et récolte le flux entrant. La classe prédite = drain avec le
    plus grand débit de flux total.

    Retourne (classe_prédite, flux_par_drain).
    """
    p, Q = run_physarum(graph, sources, sinks, n_iter, dt)
    # flux total entrant dans chaque drain (somme des |Q| sur les arêtes du drain)
    drain_flux = np.zeros(len(sinks))
    for k, (i, j, _) in enumerate(graph.edges):
        for d_idx, s in enumerate(sinks):
            if i == s or j == s:
                drain_flux[d_idx] += abs(Q[k])
    pred = int(np.argmax(drain_flux))
    return pred, drain_flux


# --------------------------------------------------------------------------- #
# Construction du graphe pour MNIST
# --------------------------------------------------------------------------- #
def grid_graph_from_image(image: np.ndarray, downscale: int = 4,
                          ) -> tuple[PhysarumGraph, np.ndarray, dict]:
    """Construit un graphe Physarum à partir d'une image MNIST.

    - L'image (28x28) est réduite en une grille de cellules.
    - Chaque cellule devient un nœud du graphe, connecté à ses voisins.
    - La luminosité du pixel (>= 0) devient l'injection de pression S_i.

    Retourne (graphe, sources, info). """
    img = np.asarray(image).squeeze()
    # downscale par blocs
    h, w = img.shape
    cell_h, cell_w = max(1, h // downscale), max(1, w // downscale)
    gh, gw = h // cell_h, w // cell_w
    # luminance par cellule (moyenne)
    cells = np.zeros((gh, gw))
    for i in range(gh):
        for j in range(gw):
            cells[i, j] = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w].mean()

    n_nodes = gh * gw
    edges = []
    # connexions horizontales et verticales (et diagonales légères)
    for i in range(gh):
        for j in range(gw):
            idx = i * gw + j
            if j + 1 < gw:
                edges.append((idx, idx + 1, 1.0))
            if i + 1 < gh:
                edges.append((idx, idx + gw, 1.0))

    graph = PhysarumGraph(n_nodes, edges)
    # sources = luminosité (>= 0), normalisée
    sources = np.clip(cells.flatten(), 0, None).astype(float)
    sources = sources / (sources.sum() + 1e-8)
    info = {'gh': gh, 'gw': gw, 'cells': cells}
    return graph, sources, info
