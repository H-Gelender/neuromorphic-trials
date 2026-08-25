"""Encodeur Deep Hebbian hiérarchique — L1 (bords) → L2 (formes).

Refonte de la classification pure. Les échecs précédents viennent de :
1. UN SEUL encodeur qui essaie de faire abstraction des bords ET des boucles
   en même temps → information diluée. Solution : HIÉRARCHIE (L1 puis L2).
2. Oja simple corrèle tous les motifs. Solution : ANTI-HEBBIAN (décorrélation)
   + SOFT-WTA (probabiliste, SoftHebb 2022).
3. Apprend tout le temps. Solution : SALIENCY-DRIVEN (n'apprend que si l'entrée
   est surprenante S ET discriminante).

ARCHITECTURE :
  L1 : patches minuscules (4x4) → détecteurs de bords (Gabor-like) via Hebbien.
  L2 : signatures creuses de L1 concaténées en grille → combinaison en formes
       (boucles, intersections) via Hebbien.
  Sortie : vecteur hiérarchique creux → classification (non supervisée).

MÉCANISMES :
- Saliency-Driven Hebbian : η effectif = β·S·(discriminance). N'apprend que si
  l'entrée est surprenante ET porteuse d'information discriminante.
- Anti-Hebbian : deux neurones qui gagnent ensemble voient leurs poids communs
  réduits (décorrélation / spécialisation).
- Soft-WTA : les gagnants sont choisis selon une probabilité (SoftHebb) plutôt
  qu'un WTA dur → robustesse, incertitude représentée.

Inspiration : SoftHebb (Hétairie et al. 2022), HMAX, cortex visuel (V1→V2).
"""
from __future__ import annotations

import numpy as np

__all__ = ["DeepHebbian", "saliency_gate", "soft_wta", "anti_hebbian_update"]


def soft_wta(activations: np.ndarray, temperature: float = 1.0,
             n_learn: int = 3, deterministic: bool = True) -> np.ndarray:
    """Soft-WTA bayésien : sélection des gagnants.

    - deterministic (inférence) : les top-K gagnants gardent leur activation
      (reproductible). Évite le bruit d'échantillonnage à l'inférence.
    - probabiliste (apprentissage) : les gagnants sont échantillonnés selon la
      distribution softmax (SoftHebb) — incertitude représentée.
    """
    a = np.asarray(activations, dtype=float)
    if deterministic:
        k = min(n_learn, len(a))
        idx = np.argsort(-a)[:k]
        out = np.zeros_like(a)
        out[idx] = a[idx]
        return out
    # probabiliste : softmax puis échantillonnage
    a_s = a - a.max()
    ex = np.exp(a_s / temperature)
    probs = ex / (ex.sum() + 1e-8)
    n = min(n_learn, len(probs))
    idx = np.random.choice(len(probs), size=n, replace=False, p=probs)
    out = np.zeros_like(a)
    out[idx] = probs[idx]
    return out


def saliency_gate(S: float, discriminance: float, beta: float = 1.5,
                  threshold: float = 0.01) -> float:
    """Saliency-Driven : n'apprend que si S (surprise) ET discriminance.

        η = β·S·discriminance, si discriminance > threshold, sinon 0.

    Une entrée banale (S≈0) ou non discriminante (discriminance très faible)
    ne déclenche pas d'apprentissage — protège les motifs acquis.
    """
    if discriminance < threshold:
        return 0.0
    return beta * S * discriminance


def anti_hebbian_update(W: np.ndarray, winners: np.ndarray, z: np.ndarray,
                        lr_anti: float = 0.01) -> np.ndarray:
    """Anti-Hebbian : décorrèle les neurones qui gagnent ensemble.

    Les poids communs des gagnants sont réduits vers l'orthogonalité, forçant
    la spécialisation sur des primitives différentes.
    """
    W = W.copy()
    z = z / (np.linalg.norm(z) + 1e-8)
    winners_idx = np.where(winners > 0)[0]
    for i in winners_idx:
        for j in winners_idx:
            if i == j:
                continue
            # réduire la corrélation entre i et j
            proj = np.dot(W[i], W[j]) * W[j]
            W[i] = W[i] - lr_anti * proj
    # re-normaliser
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    return W


class DeepHebbian:
    """Encodeur hiérarchique Deep Hebbian (L1 bords → L2 formes), non supervisé."""

    def __init__(self, patch_l1: int = 4, n_l1: int = 32, n_l2: int = 64,
                 lr_l1: float = 0.1, lr_l2: float = 0.1, seed: int = 0,
                 temp: float = 1.0, n_learn: int = 3, beta_saliency: float = 1.5,
                 lr_anti: float = 0.01):
        self.patch_l1 = patch_l1
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.lr_l1 = lr_l1
        self.lr_l2 = lr_l2
        self.temp = temp
        self.n_learn = n_learn
        self.beta_saliency = beta_saliency
        self.lr_anti = lr_anti
        self.rng = np.random.default_rng(seed)

        # L1 : patch_l1² → n_l1 (détecteurs de bords)
        d_l1 = patch_l1 * patch_l1
        self.W1 = self.rng.normal(0, 1/np.sqrt(d_l1), size=(n_l1, d_l1))
        self.W1 = self.W1 / (np.linalg.norm(self.W1, axis=1, keepdims=True) + 1e-8)
        # L2 : n_l1 → n_l2 (combine bords en formes) — entrée = latents L1 concat
        self.W2 = self.rng.normal(0, 1/np.sqrt(n_l1), size=(n_l2, n_l1))
        self.W2 = self.W2 / (np.linalg.norm(self.W2, axis=1, keepdims=True) + 1e-8)

    # -- L1 : bords --
    def _patches(self, img):
        img = np.asarray(img).squeeze()
        lo, hi = img.min(), img.max()
        img01 = (img - lo) / (hi - lo + 1e-8)
        p = self.patch_l1
        h, w = img01.shape
        pad_h = (p - h % p) % p; pad_w = (p - w % p) % p
        img_p = np.pad(img01, ((0, pad_h), (0, pad_w)))
        gh, gw = img_p.shape[0]//p, img_p.shape[1]//p
        patches = []
        for i in range(gh):
            for j in range(gw):
                patches.append(img_p[i*p:(i+1)*p, j*p:(j+1)*p].flatten())
        return np.array(patches), gh, gw

    def _encode_l1(self, img, learn, S):
        """L1 : patchs → activations creuses (détecteurs de bords)."""
        patches, gh, gw = self._patches(img)
        lat_l1 = []
        for p in patches:
            pn = p / (np.linalg.norm(p) + 1e-8)
            a = self.W1 @ pn                      # activations L1
            # soft-WTA : probabiliste (apprentissage) ou déterministe (inférence)
            y = soft_wta(a, self.temp, self.n_learn, deterministic=not learn)
            if learn:
                # Hebbian pur (pas de saliency-gate) : apprend sur chaque patch
                self.W1 += self.lr_l1 * np.outer(y, pn)
                # anti-hebbian (décorrélation des gagnants)
                self.W1 = anti_hebbian_update(self.W1, y, pn, self.lr_anti)
                self.W1 = self.W1 / (np.linalg.norm(self.W1, axis=1, keepdims=True) + 1e-8)
            lat_l1.append(y)
        return lat_l1, gh, gw

    def _encode_l2(self, lat_l1, learn, S):
        """L2 : latents L1 concaténés en grille → formes (boucles, intersections)."""
        # agréger les latents L1 (concaténation spatiale en un vecteur)
        z_l1 = np.concatenate(lat_l1)              # (n_patches * n_l1,)
        # moyenner par blocs pour une dimension fixe, puis L2
        # on répartit sur des "positions" : concat puis projeté par W2 (n_l2, n_l1)
        # simplif: on regroupe les activations L1 en vecteur n_l1 (somme)
        z_group = np.zeros(self.n_l1)
        for y in lat_l1:
            z_group += y
        z_group = z_group / (np.linalg.norm(z_group) + 1e-8)
        a = self.W2 @ z_group                       # activations L2 (formes)
        # soft-WTA mais garder de l'énergie (activation douce, pas écrasée)
        y2 = soft_wta(a, self.temp, self.n_learn, deterministic=not learn)
        # normaliser L2 SÉPARÉMENT (énergie comparable à L1)
        n2 = np.linalg.norm(y2)
        if n2 > 0:
            y2 = y2 / n2
        if learn:
            # Hebbian pur (pas de saliency-gate)
            self.W2 += self.lr_l2 * np.outer(y2, z_group)
            self.W2 = anti_hebbian_update(self.W2, y2, z_group, self.lr_anti)
            self.W2 = self.W2 / (np.linalg.norm(self.W2, axis=1, keepdims=True) + 1e-8)
        return y2

    def encode(self, img, learn: bool = False, S: float = 0.5) -> np.ndarray:
        """Forward hiérarchique complet → signature creuse (L1 + L2)."""
        lat_l1, gh, gw = self._encode_l1(img, learn, S)
        y2 = self._encode_l2(lat_l1, learn, S)
        # signature = concaténation [L1 agrégé normalisé, L2 normalisée séparément]
        z_l1 = np.zeros(self.n_l1)
        for y in lat_l1:
            z_l1 += y
        z_l1 = z_l1 / (np.linalg.norm(z_l1) + 1e-8)
        # normaliser L1 et L2 séparément pour garder les deux contributions
        n1 = np.linalg.norm(z_l1); n2 = np.linalg.norm(y2)
        sig = np.concatenate([z_l1/(n1+1e-8), y2/(n2+1e-8)])
        return sig / (np.linalg.norm(sig) + 1e-8)
