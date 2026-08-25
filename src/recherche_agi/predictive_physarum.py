"""Système hybride Blob (Physarum) + Predictive Coding.

Le principe (schéma utilisateur) :
  1. PREDICTIVE CODING : un modèle génératif prédit la représentation attendue
     pour une entrée. L'ERREUR de prédiction mesure la nouveauté.
  2. MÉCANIQUE DU BLOB : si l'erreur dépasse un seuil, on crée un nouveau
     "tuyau" (le graphe s'étend). Sinon, on consolide le tuyau le plus proche.
  3. CONSOLIDATION : chaque tuyau a une conductance (force) qui augmente à
     chaque mise à jour — le tuyau "durcit" (mémoire physique stable).

Le réservoir Physarum (flux de tubes) seul ne discrimine pas les chiffres
(intra ≈ inter). On ajoute donc une COUCHE DE LECTURE entraînée qui projette
les signatures de flux dans un espace discriminé. Le predictive coding opère
dans cet espace projeté.

Inspiration :
- Predictive coding : Rao & Ballard (1999) — le cerveau minimise l'erreur de
  prédiction entre top-down et bottom-up.
- Physarum : Tero et al. (2007) — modèle de tubes adaptatif.
- Reservoir computing : la couche lue apprend sur les états du réservoir.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .physarum import grid_graph_from_image, run_physarum

__all__ = ["Tube", "HybridBlobPredictive", "train_readout"]


# --------------------------------------------------------------------------- #
# Couche de lecture entraînée sur les signatures de flux
# --------------------------------------------------------------------------- #
def train_readout(signatures: np.ndarray, labels: np.ndarray,
                  n_classes: int = 10, epochs: int = 50, lr: float = 1e-2,
                  seed: int = 0) -> nn.Linear:
    """Entraîne une couche lue (softmax linéaire) sur les signatures de flux.

    Rend l'espace des signatures discriminant. Retourne le module nn.Linear.
    """
    torch.manual_seed(seed)
    readout = nn.Linear(signatures.shape[1], n_classes)
    opt = torch.optim.Adam(readout.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xs = torch.tensor(signatures, dtype=torch.float32)
    ys = torch.tensor(labels, dtype=torch.long)
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(Xs, ys), batch_size=64, shuffle=True)
    readout.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(readout(xb), yb)
            loss.backward()
            opt.step()
    readout.eval()
    return readout


# --------------------------------------------------------------------------- #
# Tuyaux (mémoire consolidée) + système hybride
# --------------------------------------------------------------------------- #
class Tube:
    """Un 'tuyau' : prototype dans l'espace projeté + conductance de consolidation.

    - `prototype` : vecteur de features (sortie de la couche lue) d'une classe.
    - `conductance` : force du tuyau, augmente à chaque consolidation.
    """

    def __init__(self, prototype: np.ndarray, label: int = None):
        self.prototype = prototype
        self.label = label
        self.conductance = 1.0
        self.n_updates = 0

    def consolidate(self, new_prototype: np.ndarray, learning_rate: float = 0.1):
        self.prototype = (1 - learning_rate) * self.prototype + learning_rate * new_prototype
        self.conductance *= 1.1   # le tuyau durcit
        self.n_updates += 1

    def similarity(self, feat: np.ndarray) -> float:
        a, b = self.prototype, feat
        na, nb = np.linalg.norm(a) + 1e-8, np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (na * nb))


class HybridBlobPredictive:
    """Système hybride : predictive coding (dans l'espace projeté) + blob.

    - Le réservoir Physarum produit une signature de flux.
    - La couche lue (entraînée) projette la signature dans un espace discriminé.
    - Le predictive coding prédit le tuyau attendu ; l'erreur mesure la nouveauté.
    - Le blob crée (nouveauté) ou consolide (connu) les tuyaux.
    """

    def __init__(self, readout: nn.Linear, novelty_threshold: float = 0.5,
                 min_conductance: float = 0.5, downscale: int = 4,
                 n_iter_physarum: int = 10, reservoir=None,
                 adaptive_threshold: bool = False,
                 target_novelty_rate: float = 0.15, homeo_rate: float = 0.1,
                 min_threshold: float = 0.05, max_threshold: float = 0.95):
        self.readout = readout
        self.novelty_threshold = novelty_threshold
        self.min_conductance = min_conductance
        self.downscale = downscale
        self.n_iter_physarum = n_iter_physarum
        # réservoir externe (ex. SynapticReservoir) : callable img -> signature.
        # Si None, on utilise le flux Physarum brut (grille régulière).
        self.reservoir = reservoir
        # --- Régulation homéostatique du seuil de nouveauté ---
        self.adaptive_threshold = adaptive_threshold
        self.target_novelty_rate = target_novelty_rate   # taux de nouveauté désiré
        self.homeo_rate = homeo_rate                     # vitesse de régulation
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.novelty_count = 0.0                         # compteur (lissé)
        self.observed_count = 0.0                        # nb d'observations (lissé)
        self.tubes: list[Tube] = []
        self.history = {'n_tubes': [], 'pred_error': [], 'novelty': [],
                        'threshold': []}

    # -- régulation homéostatique du seuil --
    def _homeostatic_update(self, novelty: bool) -> None:
        """Ajuste le seuil pour maintenir un taux de nouveauté ~ target.

        Si le taux de nouveauté observé est trop ÉLEVÉ (> cible), on AUGMENTE le
        seuil (plus strict, moins de nouveautés créées). S'il est trop BAS, on
        DIMINUE le seuil (plus permissif). C'est un contrôleur homéostatique.
        """
        self.novelty_count = self.novelty_count * (1 - self.homeo_rate) + float(novelty) * self.homeo_rate
        self.observed_count = self.observed_count * (1 - self.homeo_rate) + self.homeo_rate
        if self.observed_count < 1e-6:
            return
        current_rate = self.novelty_count / self.observed_count
        # erreur de régulation : taux observé vs cible
        error = current_rate - self.target_novelty_rate
        # ajuster le seuil dans la direction opposée (homéostasie)
        self.novelty_threshold = np.clip(
            self.novelty_threshold + self.homeo_rate * error,
            self.min_threshold, self.max_threshold)

    # -- réservoir + projection --
    def _signature(self, img_np: np.ndarray) -> np.ndarray:
        if self.reservoir is not None:
            # réservoir externe : signature z synaptique (compact)
            return self.reservoir.signature(img_np)
        # sinon : flux Physarum brut (grille régulière)
        g, src, info = grid_graph_from_image(img_np, downscale=self.downscale)
        gh, gw = info['gh'], info['gw']
        border = set()
        for i in range(gh):
            border.add(i * gw); border.add(i * gw + gw - 1)
        for j in range(gw):
            border.add(j); border.add((gh - 1) * gw + j)
        sinks = sorted(border)[:min(10, len(border))]
        p, Q = run_physarum(g, src, sinks, n_iter=self.n_iter_physarum)
        return np.abs(Q)

    def _features(self, img_np: np.ndarray) -> np.ndarray:
        """Signature de flux -> features projetées par la couche lue."""
        sig = self._signature(img_np).astype(np.float32)
        with torch.no_grad():
            feat = self.readout(torch.tensor(sig).unsqueeze(0)).numpy()[0]
        return feat

    # -- predictive coding : prédire + mesurer l'erreur --
    def predict(self, img_np: np.ndarray) -> tuple[Tube | None, float]:
        """Prédit le tuyau le plus probable + erreur de prédiction (0..2)."""
        feat = self._features(img_np)
        if not self.tubes:
            return None, 1.0
        best = max(self.tubes, key=lambda t: t.similarity(feat))
        error = 1.0 - best.similarity(feat)
        return best, error

    # -- le blob s'adapte --
    def observe(self, img_np: np.ndarray, label: int = None) -> dict:
        feat = self._features(img_np)
        best, error = self.predict(img_np)

        if error > self.novelty_threshold or best is None:
            self.tubes.append(Tube(feat, label))
            novelty = True
        else:
            best.consolidate(feat)
            novelty = False

        # régulation homéostatique du seuil (si activée)
        if self.adaptive_threshold:
            self._homeostatic_update(novelty)

        self.history['n_tubes'].append(len(self.tubes))
        self.history['pred_error'].append(error)
        self.history['novelty'].append(novelty)
        self.history['threshold'].append(self.novelty_threshold)
        return {'error': error, 'novelty': novelty, 'n_tubes': len(self.tubes),
                'threshold': self.novelty_threshold}

    def prune(self):
        kept = [t for t in self.tubes if t.conductance >= self.min_conductance]
        self.tubes = kept

    def classify(self, img_np: np.ndarray) -> tuple[int | None, float]:
        best, _ = self.predict(img_np)
        if best is None:
            return None, 0.0
        return self.tubes.index(best), best.similarity(self._features(img_np))

    def classify_lateral(self, img_np: np.ndarray, tau: float = 0.5,
                         method: str = 'softmax') -> tuple[int | None, np.ndarray]:
        """Classification avec INHIBITION LATÉRALE inter-tuyaux (compétition corticale).

        Calcule les activations de TOUS les tuyaux (similarité cosinus), puis applique
        l'inhibition latérale (softmax ou soustractive) pour rendre la décision nette
        (Winner-Take-All doux). Retourne (index_tuyau_gagnant, activations_compétitives).
        """
        from .neuromodulated import lateral_inhibition
        feat = self._features(img_np)
        if not self.tubes:
            return None, np.array([])
        A = np.array([t.similarity(feat) for t in self.tubes])
        A_comp = lateral_inhibition(A, tau, method)
        winner = int(np.argmax(A_comp))
        return winner, A_comp
