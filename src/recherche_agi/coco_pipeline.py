"""Pipeline COCO Stuff : entraînement CLASSE PAR CLASSE + détection avec focus.

Le point critique appris : ce type de modèle ne fonctionne PAS si on lui donne
tous les labels d'un coup — l'entraînement doit se faire CLASSE PAR CLASSE.
"""
import numpy as np
from recherche_agi import AnchorNeurons, dynamic_k


def train_class_by_class(patches: dict, n_neurons=200, seed=0,
                         prune_frac=0.15) -> AnchorNeurons:
    """Entraîne le pipeline CLASSE PAR CLASSE sur les patches COCO.

    patches : dict {classe: array(N, d)}
    """
    # dimension d'un patch
    d_in = next(iter(patches.values())).shape[1]
    anchors = AnchorNeurons(d_in=d_in, n_neurons=n_neurons, seed=seed,
                            lr=0.1, use_homeostasis=True)
    for cls in sorted(patches.keys()):      # CLASSE PAR CLASSE
        X = patches[cls]
        print(f"  entraîne classe {cls}: {len(X)} patchs")
        for p in X:
            zn = p / (np.linalg.norm(p) + 1e-8)
            anchors.learn(zn, k=dynamic_k(0.5, 1, 5), label=cls)
        anchors.physarum_prune(prune_frac)   # élagage après chaque classe
    return anchors


def classify_patch(anchors, patch, d_in):
    """Classifie un patch (1D flat)."""
    p = np.asarray(patch, float)
    if p.size != d_in:
        p = p.flatten()[:d_in]
    zn = p / (np.linalg.norm(p) + 1e-8)
    return anchors.predict_label(zn)


def accuracy_per_class(anchors, patches, d_in) -> dict:
    """Accuracy par classe."""
    acc = {}
    for cls, X in patches.items():
        correct = 0
        for p in X:
            pred, _ = classify_patch(anchors, p, d_in)
            if pred == cls:
                correct += 1
        acc[cls] = correct / len(X)
    return acc
