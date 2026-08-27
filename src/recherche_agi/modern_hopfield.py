"""Modern Hopfield Network (MHN) — remplace le WTA par une projection continue.

Pour toute couche l, la projection de l'entrée x vers l'activation latente z :
    z = softmax(β · W x)          (au lieu d'un argmax brutal)

- W : matrice des motifs/ancrages mémorisés (dictionnaire de poids)
- β (inverse de température) : levier de contrôle
    β → ∞  : retrouve un WTA dur (1 neurone à 100%)
    β modéré : consensus continu et lissé (élimine le bruit sans perdre la compétition)

Reconstruction :
    x_rec = W^T · z               (combinaison convexe des motifs)

Surprise continue et dérivable :
    S_auto = ||x - x_rec||² = ||x - W^T·softmax(βWx)||²

Plasticité Oja pondérée par z continu (au lieu d'une activation binaire).
"""
import numpy as np


def project_hopfield(x, W, beta=5.0):
    """Projection MHN : z = softmax(β · W x).

    x    : (d,) ou (n, d) entrée(s)
    W    : (n_neurons, d) motifs mémorisés
    beta : inverse de température (β→∞ = WTA dur)

    Retourne z : (n_neurons,) ou (n, n_neurons) — distribution continue.
    """
    logits = W @ x if x.ndim == 1 else x @ W.T
    logits = beta * logits
    # softmax stable
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / (exp.sum(axis=-1, keepdims=True) + 1e-12)


def reconstruct(x, W, z):
    """Reconstruction : x_rec = W^T · z (combinaison convexe des motifs)."""
    return W.T @ z if x.ndim == 1 else z @ W


def surprise(x, W, beta=5.0):
    """Surprise continue : S_auto = ||x - W^T·softmax(βWx)||².

    Retourne (S, z, x_rec).
    """
    z = project_hopfield(x, W, beta)
    x_rec = reconstruct(x, W, z)
    S = np.linalg.norm(x - x_rec, axis=-1) ** 2
    return S, z, x_rec


def winner_hopfield(x, W, beta=5.0):
    """Neurone dominant (pour la compatibilité avec le WTA existant)."""
    z = project_hopfield(x, W, beta)
    return int(np.argmax(z))


def oja_hopfield_update(W, x, beta=5.0, lr=0.1, normalize=True):
    """Règle d'Oja pondérée par la distribution continue z (au lieu de binaire).

    ΔW_i = lr · z_i · (x - x_rec) · x^T   (plastique aux attracteurs)
    puis normalisation (Oja : éviter l'explosion).

    W : (n_neurons, d), x : (d,)
    """
    z = project_hopfield(x, W, beta)         # distribution continue
    x_rec = reconstruct(x, W, z)
    # erreur par neurone : z_i * (x - x_rec)
    delta = z[:, None] * (x - x_rec)[None, :]  # (n_neurons, d)
    W_new = W + lr * delta
    if normalize:
        # normalisation Oja par neurone
        norms = np.linalg.norm(W_new, axis=1, keepdims=True) + 1e-8
        W_new = W_new / norms
    return W_new, z
