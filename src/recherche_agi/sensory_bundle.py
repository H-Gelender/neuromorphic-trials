"""Fourre-tout sensoriel — encodeur à plasticité prédictive, multi-modalités.

Le but : un encodeur qui peut intégrer IMAGES, TEXTE et SIGNAUX dans un espace
latent commun, connecté au réservoir Physarum + Predictive Coding. Aucune
rétropropagation : la plasticité est Hebbienne à 3 facteurs, pilotée par la
surprise S (auto-supervision).

PLASTICITÉ PREDICTIVE (l'encodeur apprend par la surprise) :
    ΔW = η(S) · (x^T · y - Oja_Decay(W, y))

où :
- x : entrée brute (patch d'image, token texte, fenêtre audio)
- y = ReLU(W x) : projection de l'encodeur (primitives latentes)
- η(S) = β·S : taux d'apprentissage proportionnel à la surprise
- Oja_Decay : normalisation (empêche W d'exploser)

Comportement :
- S ≈ 0 : W gelé (consolidation, protège les motifs acquis)
- S ≫ 0 : plasticité libérée, W s'ajuste aux nouvelles primitives

Sources :
- Hebb (1949) ; Oja (1982) ; règles à 3 facteurs (Fremaux & Gerstner 2016)
"""
from __future__ import annotations

import numpy as np

__all__ = ["PredictiveEncoder", "SensoryBundle", "oja_hebbian_update",
           "surprise_rate"]


def surprise_rate(S: float, beta: float = 0.5) -> float:
    """Taux d'apprentissage de l'encodeur : η(S) = β·S."""
    return beta * S


def oja_hebbian_update(W: np.ndarray, x: np.ndarray, y: np.ndarray,
                       eta: float) -> np.ndarray:
    """Règle Hebbienne 3-facteurs avec normalisation Oja.

        ΔW = η · (x^T·y - Oja_Decay(W, y))

    Oja_Decay = y^2 · W (normalisation locale qui évite l'explosion des poids).
    """
    # terme Hebbien : co-activation entrée-sortie
    hebb = np.outer(y, x)                    # (d_out, d_in)
    # terme Oja : normalisation (retire la partie alignée sur y²·W)
    oja = (y ** 2)[:, None] * W
    delta = eta * (hebb - oja)
    return W + delta


class PredictiveEncoder:
    """Encodeur Hebbien à 3 facteurs, piloté par la surprise.

    Projette des patches d'entrée x (d_in) vers un espace latent y (d_out) via
    y = ReLU(W x). W est mis à jour par la règle d'Oja, avec un taux η(S)
    proportionnel à la surprise S (auto-supervision).
    """

    def __init__(self, d_in: int, d_out: int, seed: int = 0, eta_beta: float = 0.5,
                 norm_input: bool = True):
        rng = np.random.default_rng(seed)
        # init Hebbienne normalisée (biologique)
        W = rng.normal(0, 1 / np.sqrt(d_in), size=(d_out, d_in))
        self.W = W
        self.d_in = d_in
        self.d_out = d_out
        self.eta_beta = eta_beta        # β dans η(S) = β·S
        self.norm_input = norm_input

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Projette x (d_in,) -> y (d_out,) = ReLU(W x)."""
        x = np.asarray(x, dtype=float)
        if self.norm_input:
            n = np.linalg.norm(x) + 1e-8
            x = x / n
        y = np.maximum(self.W @ x, 0.0)   # ReLU
        # normaliser le vecteur latent (unit norme pour similarité)
        ny = np.linalg.norm(y) + 1e-8
        return y / ny

    def learn(self, x: np.ndarray, S: float) -> np.ndarray:
        """Encode ET met à jour W selon la surprise (plasticité prédictive).

        - S≈0 : η≈0, W quasi gelé (consolidation).
        - S≫0 : plasticité libérée, W s'ajuste (nouveauté).
        Retourne y (projection post-mise à jour).
        """
        x = np.asarray(x, dtype=float)
        if self.norm_input:
            x = x / (np.linalg.norm(x) + 1e-8)
        y_raw = np.maximum(self.W @ x, 0.0)
        eta = surprise_rate(S, self.eta_beta)
        self.W = oja_hebbian_update(self.W, x, y_raw, eta)
        # re-normaliser W (stabilité)
        self.W = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-8)
        ny = np.linalg.norm(y_raw) + 1e-8
        return y_raw / ny


# --------------------------------------------------------------------------- #
# Modalités : image, texte, signaux
# --------------------------------------------------------------------------- #
def image_to_patches(img: np.ndarray, patch_size: int = 7) -> np.ndarray:
    """Découpe une image en patches (champ récepteur local)."""
    img = np.asarray(img).squeeze()
    h, w = img.shape
    # normaliser
    lo, hi = img.min(), img.max()
    img01 = (img - lo) / (hi - lo + 1e-8)
    ph = pw = patch_size
    # pad pour divisibilité
    pad_h = (ph - h % ph) % ph
    pad_w = (pw - w % pw) % pw
    img_p = np.pad(img01, ((0, pad_h), (0, pad_w)))
    gh, gw = img_p.shape[0] // ph, img_p.shape[1] // pw
    patches = []
    for i in range(gh):
        for j in range(gw):
            patches.append(img_p[i*ph:(i+1)*ph, j*pw:(j+1)*pw].flatten())
    return np.array(patches)


def text_to_embeddings(text: str, vocab_size: int = 128, seq_len: int = 8,
                       d_embed: int = 16, seed: int = 0) -> np.ndarray:
    """Convertit un texte en embeddings (représentation multi-hot + projection).

    Simple : chaque caractère -> one-hot sur vocab, projeté par une matrice
    aléatoire fixe (déterminée). Retourne (seq_len, d_embed).
    """
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.1, size=(vocab_size, d_embed))
    seq = []
    for ch in str(text)[:seq_len].ljust(seq_len, ' '):
        idx = ord(ch) % vocab_size
        v = np.zeros(vocab_size)
        v[idx] = 1.0
        seq.append(v @ E)
    return np.array(seq)


def signal_to_frames(signal: np.ndarray, frame_size: int = 16, hop: int = 8
                     ) -> np.ndarray:
    """Découpe un signal temporel en fenêtres (frames)."""
    sig = np.asarray(signal, dtype=float).flatten()
    if np.std(sig) > 0:
        sig = (sig - sig.mean()) / (np.std(sig) + 1e-8)
    frames = []
    i = 0
    while i + frame_size <= len(sig):
        frames.append(sig[i:i+frame_size])
        i += hop
    if not frames:
        frames.append(np.zeros(frame_size))
    return np.array(frames)


# --------------------------------------------------------------------------- #
# Fourre-tout sensoriel
# --------------------------------------------------------------------------- #
class SensoryBundle:
    """Fourre-tout sensoriel : intègre images, texte, signaux vers un latent commun.

    Chaque modalité est encodée par un PredictiveEncoder. Tous les latents sont
    concaténés en un vecteur commun, connecté au réservoir/PC. La plasticité de
    chaque encodeur est pilotée par la surprise globale S (auto-supervision).

    Scalable : on peut ajouter de nouvelles modalités en ajoutant un encodeur.
    """

    def __init__(self, latent_dim: int = 32, seed: int = 0, eta_beta: float = 0.5):
        self.latent_dim = latent_dim
        self.eta_beta = eta_beta
        self.rng = np.random.default_rng(seed)
        # encodeurs par modalité : créés à la première utilisation (dépend de d_in)
        self.encoders = {}
        self.modalities = []
        self.last_latents = {}
        self.last_surprise = 0.0

    def _get_encoder(self, modality: str, d_in: int) -> PredictiveEncoder:
        if modality not in self.encoders:
            self.encoders[modality] = PredictiveEncoder(
                d_in, self.latent_dim, seed=self.rng.integers(0, 10000),
                eta_beta=self.eta_beta)
            self.modalities.append(modality)
        return self.encoders[modality]

    def _latent(self, modality: str, x: np.ndarray, S: float, learn: bool):
        enc = self._get_encoder(modality, x.shape[0])
        if learn:
            return enc.learn(x, S)
        return enc.encode(x)

    # -- API modalités --
    def encode_image(self, img, patch_size: int = 7, S: float = 0.0,
                     learn: bool = False) -> np.ndarray:
        """Encode une image -> latent commun (agrégation des patches)."""
        patches = image_to_patches(img, patch_size)
        lat = [self._latent('image', p, S, learn) for p in patches]
        self.last_latents['image'] = np.mean(lat, axis=0)
        return self.last_latents['image']

    def encode_text(self, text: str, S: float = 0.0, learn: bool = False,
                    **kwargs) -> np.ndarray:
        """Encode un texte -> latent commun (agrégation des embeddings)."""
        emb = text_to_embeddings(text, **kwargs)
        # concatène les embeddings en un vecteur, puis encode globalement
        flat = emb.flatten()
        # utiliser un encodeur 'text' avec d_in = taille du vecteur
        lat = self._latent('text', flat, S, learn)
        self.last_latents['text'] = lat
        return lat

    def encode_signal(self, signal, frame_size: int = 16, hop: int = 8,
                      S: float = 0.0, learn: bool = False) -> np.ndarray:
        """Encode un signal -> latent commun (agrégation des frames)."""
        frames = signal_to_frames(signal, frame_size, hop)
        lat = [self._latent('signal', f, S, learn) for f in frames]
        self.last_latents['signal'] = np.mean(lat, axis=0)
        return self.last_latents['signal']

    # -- fusion multi-modale --
    def fuse(self, *latents: np.ndarray) -> np.ndarray:
        """Concatène des latents multi-modaux en un vecteur commun normalisé."""
        vecs = [np.asarray(l, dtype=float) for l in latents if l is not None]
        if not vecs:
            return np.zeros(self.latent_dim)
        full = np.concatenate(vecs)
        return full / (np.linalg.norm(full) + 1e-8)

    def surprise_gated_learn(self, S: float, surprise_callback=None):
        """Active l'apprentissage des encodeurs selon la surprise (à appeler après)."""
        self.last_surprise = S
