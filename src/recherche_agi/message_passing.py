"""Message Passing sur un graphe de nœuds (patches) — lissage + segmentation.

Deux types de messages :
1. RÉSONANCE (consensus spatial) : les voisins activés envoient un biais positif
   aux neurones d'ancrage similaires du nœud central -> lisse le bruit.
2. SURPRISE / INHIBITION : un nœud à forte erreur diffuse une inhibition à ses
   voisins -> crée une frontière nette (pas de lissage sur les contours).

Formule (à chaque pas de temps) :
   z_i^(t+1) = WTA( x_i·W + α · Σ_{j∈N(i)} F_{j→i} · z_j^(t) )
où F_{j→i} est la conductivité du tube Physarum entre j et i.

Le graphe est une grille (chaque nœud = un patch), N(i) = 4-voisins.
"""
import numpy as np


def build_grid_adjacency(gh, gw, n=None):
    """Construit la liste d'adjacence d'une grille gh x gw (4-voisins).

    n : nombre réel de nœuds (si < gh*gw, les nœuds au-delà sont ignorés).
    """
    if n is None:
        n = gh * gw
    adj = [[] for _ in range(n)]
    for i in range(gh):
        for j in range(gw):
            idx = i * gw + j
            if idx >= n:
                continue
            if i > 0 and idx - gw < n: adj[idx].append(idx - gw)
            if i < gh-1 and idx + gw < n: adj[idx].append(idx + gw)
            if j > 0 and idx - 1 < n and idx % gw > 0: adj[idx].append(idx - 1)
            if j < gw-1 and idx + 1 < n: adj[idx].append(idx + 1)
    return adj


def conductivity_physarum(activations, adjacency, beta=1.0):
    """Conductivité des tubes Physarum entre nœuds voisins.

    Plus deux voisins s'activent ensemble (résonance), plus le tube se renforce.
    F_{j->i} = sigmoid(beta * (A_j * A_i)).
    """
    F = {}
    for i, neigh in enumerate(adjacency):
        for j in neigh:
            # conductivité basée sur la co-activation
            F[(i, j)] = 1.0 / (1.0 + np.exp(-beta * activations[i] * activations[j]))
    return F


def message_passing_step(z, W, x_features, adjacency, F, alpha=0.5,
                         surprise=None, surprise_gain=2.0):
    """Un pas de message passing sur le graphe.

    z          : vecteur d'activations des nœuds (n,) — les "neurones gagnants"
    W          : matrice des prototypes (n_neurons, d)
    x_features : features de chaque nœud (n, d) — l'entrée
    adjacency  : liste d'adjacence
    F          : conductivité Physarum (dict {(i,j): conductivité})
    alpha      : poids du message des voisins
    surprise   : surprise de chaque nœud (n,) — si None, pas d'inhibition
    surprise_gain : force de l'inhibition

    Retourne le nouveau z après lissage/consensus + inhibition.
    """
    n = len(z)
    new_z = z.copy()
    # pour chaque nœud, message de résonance des voisins
    for i in range(n):
        # biais de résonance = somme des activations voisines pondérées par F
        reson = 0.0
        inh = 0.0
        for j in adjacency[i]:
            reson += F.get((j, i), 0.0) * z[j]
            if surprise is not None:
                # un voisin surprenant diffuse de l'inhibition
                inh += surprise[j] * F.get((j, i), 0.0)
        # surprise propre -> inhibition locale (empêche le lissage sur contours)
        own_surprise = surprise[i] if surprise is not None else 0.0
        # la surprise inhibe le lissage : si surprise forte, on garde la valeur
        # brute (frontière), sinon on lisse (consensus)
        smoothness = 1.0 / (1.0 + surprise_gain * own_surprise)
        new_z[i] = z[i] + alpha * smoothness * reson
    return new_z


def spatial_consensus(z, adjacency, F, alpha=0.5, n_iter=1, surprise=None):
    """Applique plusieurs itérations de message passing pour le consensus."""
    for _ in range(n_iter):
        z = message_passing_step(z, None, None, adjacency, F, alpha, surprise)
    return z


def message_passing_train(activations, surprise, adjacency, conductance,
                          alpha=0.5, surprise_gain=2.0):
    """Message passing PENDANT L'ENTRAÎNEMENT (structuration).

    Chaque nœud ajuste son activation en tenant compte de la résonance de ses
    voisins (consensus local) et de l'inhibition des nœuds surprenants
    (maintien de la compétition WTA).

    activations : (n, n_neurons) activations de chaque nœud
    surprise    : (n,) surprise de reconstruction de chaque nœud
    adjacency   : liste d'adjacence
    conductance : (n, n) conductivité Physarum entre nœuds

    Retourne les activations lissées par consensus + inhibition.
    """
    n = activations.shape[0]
    n_neurons = activations.shape[1]
    out = activations.copy()
    for i in range(n):
        reson = np.zeros(n_neurons)
        inh = 0.0
        for j in adjacency[i]:
            g = conductance[i, j]
            reson += g * activations[j]          # résonance des voisins
            inh += surprise[j] * g               # inhibition des voisins surprenants
        own_s = surprise[i]
        # l'inhibition réduit le poids du consensus (garde la frontière)
        smoothness = 1.0 / (1.0 + surprise_gain * own_s)
        out[i] = activations[i] + alpha * smoothness * reson
    return out


def update_physarum_conductance(conductance, activations, adjacency, lr=0.05,
                                decay=0.98, min_g=0.0, max_g=1.0):
    """Met à jour la conductivité Physarum selon le flux de co-activation.

    Les connexions où le flux de messages est utile (co-activation élevée)
    se renforcent, les autres sont élaguées (décroissance).
    """
    for i, neigh in enumerate(adjacency):
        for j in neigh:
            # co-activation des nœuds voisins (résonance)
            co_act = activations[i] @ activations[j]
            # renforcement si co-activation, décroissance sinon (Physarum)
            conductance[i, j] *= decay
            conductance[i, j] += lr * co_act
            conductance[i, j] = min(max_g, max(min_g, conductance[i, j]))
    return conductance
