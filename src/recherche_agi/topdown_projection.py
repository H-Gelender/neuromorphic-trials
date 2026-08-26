"""Projection Top-Down guidée — masque sémantique 1×1.

Mariage entre l'abstraction sémantique des couches profondes (C8) et la
résolution spatiale de la couche basse (C1).

Pipeline :
1. SÉLECTION : on choisit un neurone d'ancrage actif dans une couche profonde
   (ex. le neurone C8 responsable d'un objet).
2. BACK-PROPAGATION TOP-DOWN : on propage un signal unitaire depuis ce neurone
   vers le bas via les skip-connections et les tubes Physarum validés.
3. FILTRE SPATIAL : C1 reçoit la rétro-signal et agit comme un filtre haute
   résolution -> chaque patch 4x4 (ou pixel 1x1) est marqué selon qu'il est
   atteint par le signal top-down.
4. MASQUE FINAL : masque binaire net (0/1) ou carte sémantique où l'objet est
   unifié à l'échelle du patch/pixel.
"""
import numpy as np


def activate_deep_neuron(layers, depth, neuron, activations_by_layer=None):
    """Sélectionne un neurone actif en couche profonde.

    Retourne un vecteur de signal unitaire (1 sur le neurone ciblé, 0 ailleurs).
    """
    n = len(layers[depth].W)
    signal = np.zeros(n)
    if neuron < n:
        signal[neuron] = 1.0
    return signal


def backprop_topdown(signal, layers, skip_connections, current_depth, W_adj=None):
    """Back-propagation top-down : propage le signal de la couche profonde vers C1.

    signal            : (n_current,) signal unitaire sur le neurone ciblé
    layers            : liste des couches (C1..Ck)
    skip_connections  : module SkipConnections (liste de tubes)
    current_depth     : index de la couche profonde (len(layers)-1)
    W_adj             : conductance Physarum (optionnel)

    Retourne la carte d'activation de C1 (n_C1,) après propagation top-down.
    """
    n_layers = len(layers)
    # liste des signaux par couche (bottom-up index)
    sig = [None] * n_layers
    sig[current_depth] = signal

    # propager de la couche profonde vers la couche basse
    for depth in range(current_depth, 0, -1):
        cur_sig = sig[depth]
        lower_sig = np.zeros(len(layers[depth-1].W))
        if cur_sig is None:
            sig[depth-1] = lower_sig
            continue
        # pour chaque connexion (skip ou tube) reliant la couche depth à depth-1
        for c in skip_connections.connections:
            if c['to_layer'] == depth and c['from_layer'] == depth - 1:
                # le signal du neurone cible se propage au neurone source pondéré
                lower_sig[c['from_neuron']] += c['conductance'] * cur_sig[c['to_neuron']]
            elif c['from_layer'] == depth and c['to_layer'] == depth - 1:
                lower_sig[c['to_neuron']] += c['conductance'] * cur_sig[c['from_neuron']]
        # si aucune connexion directe, on propage via le champ récepteur moyen
        if lower_sig.sum() == 0:
            # propagation par répartition uniforme (champ récepteur complet)
            lower_sig += cur_sig.mean() / max(1, len(lower_sig))
        # normaliser pour éviter l'explosion
        m = lower_sig.max()
        if m > 0:
            lower_sig = lower_sig / m
        sig[depth-1] = lower_sig
    return sig[0]   # la carte d'activation de C1


def spatial_mask_c1(c1_signal, c1_activations, threshold=0.5):
    """Utilise C1 comme filtre spatial : masque binaire sur les patches.

    c1_signal      : (n_C1,) rétro-signal top-down reçu par chaque neurone C1
    c1_activations : (n_patches,) neurone C1 gagnant de chaque patch (ou (n_patches, n_C1))
    threshold      : seuil pour le masque binaire

    Retourne un masque (n_patches,) binaire (0/1).
    """
    c1_sig = np.asarray(c1_signal)
    # si c1_activations est 2D, prendre le gagnant
    if c1_activations.ndim == 2:
        winners = np.argmax(c1_activations, axis=1)
    else:
        winners = c1_activations
    # chaque patch est marqué si son neurone gagnant reçoit un fort rétro-signal
    score = c1_sig[winners]
    mask = (score >= threshold).astype(float)
    return mask


def project_to_pixels(mask_patches, gh, gw, patch_size=4):
    """Convertit le masque par patch en masque par pixel (1×1) via kron."""
    m = np.kron(mask_patches.reshape(gh, gw), np.ones((patch_size, patch_size)))
    return m


def topdown_projection(model, image_zn, target_depth, target_neuron,
                       threshold=0.5, patch_size=4):
    """Pipeline complet : projection top-down guidée -> masque sémantique 1×1.

    model         : HierarchicalCOCO (avec .layers, .skips)
    image_zn      : features normalisées des patches de l'image (n, d)
    target_depth  : couche cible (ex. len(model.layers)) — couche profonde
    target_neuron : neurone cible dans cette couche
    """
    layers = model.layers + [model.layer]
    # 1) activations de chaque patch sur la couche profonde
    deep_acts = image_zn @ layers[target_depth].W.T
    deep_winners = np.argmax(deep_acts, axis=1)
    # 2) rétro-signal depuis le neurone cible (borné au nb de neurones de la couche)
    n_neurons = layers[target_depth].W.shape[0]
    signal = np.zeros(n_neurons)
    target_neuron = int(np.clip(target_neuron, 0, n_neurons - 1))
    signal[target_neuron] = 1.0
    c1_sig = backprop_topdown(signal, layers, model.skips, target_depth)
    # 3) C1 gagnants des patches
    c1_acts = image_zn @ layers[0].W.T
    c1_winners = np.argmax(c1_acts, axis=1)
    # 4) masque par patch : les patches dont le neurone C1 reçoit le rétro-signal
    mask_patches = spatial_mask_c1(c1_sig, c1_winners, threshold)
    return mask_patches
