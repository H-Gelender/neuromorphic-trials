"""Boucle neuromodulée — plasticité gérée par la surprise (3-Factor Hebbian).

3 étapes biologiques pour fermer la boucle :

ÉTAPE 1 : PLASTICITÉ GÉRÉE PAR LA SURPRISE (Gated Plasticity)
  La plasticité dépend du bain neuromodulateur global S = ||z - ẑ|| :
      η(S) = η_base + β · S
      ΔD_ij = η(S) · (tanh(Δp) · (p_i·p_j)) - γ·D_ij
  - Faible surprise : η chute, le Physarum devient rigide (protège la mémoire).
  - Forte surprise : η augmente, le réseau devient liquide (reconfigure vite).

ÉTAPE 2 : ÉVALUATION MÉTABOLIQUE TEMPORELLE (Gain d'Arousal)
  Le nombre d'itérations de relaxation dépend de la surprise :
      N_iter(S) = N_min + floor(α · S)
  - Chiffre banal : 5-10 itérations (économie d'énergie).
  - Forme nouvelle : 30-50 itérations (équilibre complexe).

ÉTAPE 3 : INHIBITION LATÉRALE INTER-TUYAUX (Compétition corticale)
  Les activations des tuyaux A = [a_1..a_K] sont compétitives :
      Â = softmax(A / τ)   (Winner-Take-All doux)
  - Un tuyau actif étouffe les voisins hésitants → décision nette.

Inspiration : règles Hebbiennes à 3 facteurs (e.g. Fremaux & Gerstner 2016),
GABAergique / compétition corticale, modèle de surprise (Friston free energy).
"""
from __future__ import annotations

import numpy as np

from .enhanced_reservoir import EnhancedReservoir
from .local_ssm import LocalSSM, surprise_to_delta
from .physarum import grid_graph_from_image
from .synaptic_physarum import synaptic_flow

__all__ = ["surprise_eta", "metabolic_n_iter", "lateral_inhibition",
           "NeuromodulatedReservoir", "surprise_gated_plasticity"]


# --------------------------------------------------------------------------- #
# Étape 1 : plasticité gérée par la surprise
# --------------------------------------------------------------------------- #
def surprise_eta(S: float, eta_base: float = 0.1, beta: float = 0.5) -> float:
    """Taux d'apprentissage modulé par la surprise.

        η(S) = η_base + β · S
    """
    return eta_base + beta * S


def surprise_gated_plasticity(graph, pressures: np.ndarray, S: float,
                              eta_base: float = 0.1, gamma: float = 0.1,
                              beta: float = 0.5, beta_sig: float = 5.0) -> None:
    """Règle Hebbienne à 3 facteurs : plasticité gérée par la surprise S.

        η = η_base + β·S
        ΔD_ij = η · (tanh(Δp) · (p_i·p_j)) - γ·D_ij

    Le facteur neuromodulateur global S contrôle le taux d'apprentissage.
    """
    eta = surprise_eta(S, eta_base, beta)

    def sigma(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-beta_sig * x))

    for k, (i, j, _) in enumerate(graph.edges):
        delta_p = pressures[i] - pressures[j]
        co_act = sigma(pressures[i]) * sigma(pressures[j])
        # 3-facteur : η(S) · tanh(Δp) · co_activation
        graph.D[k] = graph.D[k] + eta * (np.tanh(beta_sig * delta_p) * co_act) - gamma * graph.D[k]
        graph.D[k] = max(graph.D[k], 1e-6)


# --------------------------------------------------------------------------- #
# Étape 2 : évaluation métabolique temporelle
# --------------------------------------------------------------------------- #
def metabolic_n_iter(S: float, n_min: int = 5, alpha: float = 40.0,
                     n_max: int = 50) -> int:
    """Nombre d'itérations de relaxation dépendant de la surprise.

        N_iter(S) = N_min + floor(α · S), borné à n_max
    """
    n = n_min + int(np.floor(alpha * S))
    return min(n, n_max)


# --------------------------------------------------------------------------- #
# Étape 3 : inhibition latérale inter-tuyaux
# --------------------------------------------------------------------------- #
def lateral_inhibition(activations: np.ndarray, tau: float = 0.5,
                       method: str = 'softmax') -> np.ndarray:
    """Compétition entre tuyaux (inhibition latérale / WTA doux).

    - method='softmax' : Â = softmax(A / τ) — normalisation compétitive.
    - method='subtractive' : Â_k = A_k - λ·Σ_{j≠k} A_j (inhibition soustractive).
    """
    A = np.asarray(activations, dtype=float)
    if method == 'softmax':
        ex = np.exp(A / tau)
        return ex / (ex.sum() + 1e-8)
    elif method == 'subtractive':
        lam = 1.0 / max(len(A) - 1, 1)
        out = A.copy()
        total = A.sum()
        for k in range(len(A)):
            out[k] = A[k] - lam * (total - A[k])
        return np.clip(out, 0, None)
    else:
        raise ValueError(f"méthode inconnue: {method}")


# --------------------------------------------------------------------------- #
# Réservoir neuromodulé (boucle complète)
# --------------------------------------------------------------------------- #
class NeuromodulatedReservoir(EnhancedReservoir):
    """Réservoir synaptique dont la dynamique est modulée par la surprise S.

    Boucle : signature z → PC calcule l'erreur S → S module η (plasticité)
    et N_iter (relaxation métabolique) → le Physarum s'adapte selon S.

    Pour un usage autonome (sans PC externe), on peut fournir un `predictor`
    (callable z -> ẑ) pour calculer S = ||z - ẑ||. Sans predictor, S=0.
    """

    def __init__(self, axes=('top_down', 'left_right'), alpha: float = 5.0,
                 n_iter: int = 10, n_zones: int = 32, downscale: int = 8,
                 eta_base: float = 0.1, gamma: float = 0.1, beta: float = 5.0,
                 beta_surprise: float = 0.5, n_min: int = 5, alpha_metab: float = 40.0,
                 n_max: int = 50, predictor=None, use_retina: bool = False,
                 use_competition: bool = False, temp: float = 1.0):
        super().__init__(axes=axes, alpha=alpha, n_iter=n_iter, n_zones=n_zones,
                         downscale=downscale, eta=eta_base, gamma=gamma, beta=beta,
                         use_retina=use_retina, use_competition=use_competition,
                         temp=temp)
        self.eta_base = eta_base
        self.beta_surprise = beta_surprise   # β dans η(S) = η_base + β·S
        self.n_min = n_min
        self.alpha_metab = alpha_metab       # α dans N_iter(S)
        self.n_max = n_max
        self.predictor = predictor           # callable z -> ẑ (pour S)
        self.last_surprise = 0.0

    def _surprise(self, z: np.ndarray) -> float:
        """Erreur de prédiction S = ||z - ẑ|| (si un prédicteur est fourni)."""
        if self.predictor is None:
            return 0.0
        z_hat = self.predictor(z)
        return float(np.linalg.norm(z - z_hat))

    def _signature_for_axis_with_surprise(self, img_np, axis, S):
        """Signature z pour un axe, avec plasticité et N_iter modulés par S."""
        graph, sources0, info = grid_graph_from_image(img_np, downscale=self.downscale)
        gh, gw = info['gh'], info['gw']
        centroids = np.array([[j + 0.5, i + 0.5] for i in range(gh) for j in range(gw)])

        border = set()
        for i in range(gh):
            border.add(i * gw); border.add(i * gw + gw - 1)
        for j in range(gw):
            border.add(j); border.add((gh - 1) * gw + j)
        sinks = sorted(border)[:min(10, len(border))]

        sources = sources0.copy()
        from .enhanced_reservoir import _axis_sources
        axis_prof = _axis_sources(gh, gw, axis)
        combined = sources * axis_prof
        combined = combined / (combined.sum() + 1e-8)

        # N_iter modulé par la surprise (étape 2)
        n_iter = metabolic_n_iter(S, self.n_min, self.alpha_metab, self.n_max)
        eta = surprise_eta(S, self.eta_base, self.beta_surprise)

        for _ in range(n_iter):
            p, Q, _ = synaptic_flow(graph, combined, sinks, self.alpha)
            # plasticité gérée par la surprise (étape 1)
            surprise_gated_plasticity(graph, p, S, eta, self.gamma, self.beta_surprise)

        # pooling (étape 3 appliquée au niveau des tuyaux, pas ici)
        z = np.zeros(self.n_zones)
        node_zone = np.zeros(graph.n_nodes, dtype=int)
        n_side = int(np.sqrt(self.n_zones))
        cx = centroids[:, 0]; cy = centroids[:, 1]
        x_bin = np.clip((cx / 28.0 * n_side).astype(int), 0, n_side - 1)
        y_bin = np.clip((cy / 28.0 * n_side).astype(int), 0, n_side - 1)
        node_zone = y_bin * n_side + x_bin
        for k, (i, j, _) in enumerate(graph.edges):
            zone = node_zone[i] if Q[k] >= 0 else node_zone[j]
            z[zone] += max(Q[k], 0.0)
        z = z / (np.linalg.norm(z) + 1e-8)
        return z

    def signature(self, img_np: np.ndarray) -> np.ndarray:
        """Signature z neuromodulée : la surprise module la dynamique du réservoir."""
        # passer 1 : signature rapide pour estimer S (avec n_iter faible)
        # (approximation : on utilise une première passe pour S, puis la vraie)
        z_preview = super().signature(img_np)
        self.last_surprise = self._surprise(z_preview)

        # passer 2 : signature avec plasticité + N_iter modulés par S
        parts = []
        for axis in self.axes:
            z = self._signature_for_axis_with_surprise(img_np, axis, self.last_surprise)
            parts.append(z)
        full = np.concatenate(parts)
        return full / (np.linalg.norm(full) + 1e-8)

    @property
    def n_features(self) -> int:
        return len(self.axes) * self.n_zones


class SSMNeuromodulatedReservoir(NeuromodulatedReservoir):
    """Réservoir neuromodulé + SSM local : fusion espace (Physarum) et temps (SSM).

    À chaque image :
      1. le Physarum synaptique produit la signature z (espace).
      2. la surprise S = ||z - ẑ|| pilote Δ = σ(S).
      3. le SSM local intègre z dans une mémoire récurrente h_t :
             h_t = (1 - Δ)·h_{t-1} + Δ·z
      4. la signature finale = concaténation [h_t, z] (mémoire + nouveauté).

    - S≈0 → Δ≈0 : le SSM garde sa mémoire (contexte stable, inertie).
    - S≫0 → Δ≈1 : la surprise réinitialise la mémoire (capte la nouveauté).

    Ajoute une dimension TEMPORELLE : chaque image est vue à travers le prisme
    du contexte précédent, pas isolément.
    """

    def __init__(self, *args, beta_ssm: float = 3.0, use_memory: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_ssm = beta_ssm
        self.use_memory = use_memory
        self.n_features_ssm = len(self.axes) * self.n_zones   # taille de z
        self.ssm = LocalSSM(self.n_features_ssm, init=0.0)
        self.last_delta = 0.0
        self.last_surprise = 0.0

    def _surprise_from_z(self, z: np.ndarray) -> float:
        if self.predictor is None:
            return 0.0
        z_hat = self.predictor(z)
        return float(np.linalg.norm(z - z_hat))

    def signature(self, img_np: np.ndarray) -> np.ndarray:
        """Signature = [mémoire temporelle h_t, nouveauté z], pilotée par S."""
        # 1) signature neuromodulée z (espace)
        z = super().signature(img_np)
        self.last_surprise = self._surprise_from_z(z)

        # 2) Δ = σ(S) contrôle la vitesse du SSM
        self.last_delta = surprise_to_delta(self.last_surprise, self.beta_ssm)

        # 3) intègre z dans la mémoire temporelle
        h = self.ssm.step(z, self.last_delta)

        # 4) signature finale : mémoire + nouveauté
        if self.use_memory:
            full = np.concatenate([h, z])
            return full / (np.linalg.norm(full) + 1e-8)
        return z

    @property
    def n_features(self) -> int:
        if self.use_memory:
            return 2 * len(self.axes) * self.n_zones
        return len(self.axes) * self.n_zones

    def reset_memory(self):
        self.ssm.reset()
