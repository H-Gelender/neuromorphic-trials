"""Service d'analyse de l'espace vectoriel du réseau.

Fournit les outils pour inspecter le modèle et décider quoi retirer :
- activation des couches cachées,
- détection des neurones morts / faibles,
- détection des neurones redondants (poids d'entrée très similaires),
- pruning (construction d'un modèle plus petit).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

__all__ = [
    "denormalize_mnist",
    "compute_activations",
    "analyze_neurons",
    "find_similar_pairs",
    "prune_neurons",
    "summarize_model",
    "analyze_routing",
]


def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """Dénormalise des images MNIST (chargées avec mean=0.1307, std=0.3081)."""
    return x * 0.3081 + 0.1307


def compute_activations(model: nn.Module, X, expert: int = 0,
                        batch_size: int = 256) -> torch.Tensor:
    """Active la couche cachée de l'expert `expert` sur X.

    Retourne (N, hidden) les valeurs post-ReLU.
    """
    model.eval()
    X = torch.as_tensor(X)
    exp = model.experts[expert]
    acts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].flatten(1)
            h = exp.act(exp.fc1(xb))
            acts.append(h)
    return torch.cat(acts, dim=0)


def analyze_neurons(model: nn.Module, X, expert: int = 0,
                    dead_threshold: float = 1e-6,
                    weak_threshold: float = 0.01) -> dict:
    """Analyse les neurones de la couche cachée d'un expert.

    - Neurone "mort" : jamais activé (max d'activation ≈ 0).
    - Neurone "faible" : activé mais avec une moyenne très basse (sous weak_threshold).

    Retourne un dict : activations brutes, moyennes, max, morts, faibles.
    """
    acts = compute_activations(model, X, expert=expert)
    mean_act = acts.mean(dim=0)          # (hidden,)
    max_act = acts.max(dim=0).values     # (hidden,)
    n_neurons = acts.shape[1]

    dead = (max_act <= dead_threshold).nonzero().flatten().tolist()
    weak = [i for i in range(n_neurons)
            if i not in dead and mean_act[i].item() < weak_threshold]

    print(f"[analyse] couche cachée : {n_neurons} neurones")
    print(f"  neurones morts   : {dead if dead else 'aucun'}")
    print(f"  neurones faibles : {weak if weak else 'aucun'}")
    print(f"  activation moyenne par neurone : {mean_act.tolist()}")
    print(f"  activation max    par neurone : {max_act.tolist()}")
    return {'acts': acts, 'mean': mean_act, 'max': max_act,
            'dead': dead, 'weak': weak}


def find_similar_pairs(model: nn.Module, expert: int = 0, threshold: float = 0.9) -> list:
    """Détecte les paires de neurones cachés dont les poids d'entrée sont très similaires.

    Similitude cosinus entre les vecteurs de poids de la première couche
    (784 → hidden). Deux neurones avec cos >= threshold sont considérés
    redondants (un seul suffit). Retourne la liste des paires (i, j, cos).
    """
    W = model.experts[expert].fc1.weight.data  # (hidden, 784)
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    C = Wn @ Wn.T                              # matrice cosinus (hidden, hidden)
    iu, ju = torch.triu_indices(C.shape[0], C.shape[1], offset=1)
    pairs = []
    for a, b in zip(iu.tolist(), ju.tolist()):
        cos = C[a, b].item()
        if cos >= threshold:
            pairs.append((a, b, cos))
    print(f"[similarité] paires de neurones avec cos >= {threshold}: "
          f"{len(pairs)}" + ("" if pairs else " → aucune redondance"))
    for a, b, cos in pairs:
        print(f"  neurones {a} et {b} : cos = {cos:.4f}")
    return pairs


def prune_neurons(model: nn.Module, keep: list, expert: int = 0) -> nn.Module:
    """Retire les neurones cachés (hors de `keep`) de l'expert `expert`.

    Construit un nouveau modèle de la même classe (type(model)), où l'expert
    ciblé a ``hidden = len(keep)`` et les autres experts conservent leur propre
    hidden. Recopie les poids.

    IMPORTANT — honnêteté mathématique :
    - Retirer un neurone **mort** (relu toujours nulle) est **exact et sans
      perte** : sa contribution à la sortie est identiquement nulle.
    - Retirer un neurone **vivant** (même très similaire à un autre) est une
      **compression approximative** : avec ReLU, la contribution est non
      linéaire et ne peut pas être exactement transférée. La perte se mesure
      en comparant `evaluate` avant / après le prune.

    Note : suppose que ``type(model)`` accepte ``d_in``, ``temperature``, et que
    ``add_expert`` accepte un argument ``hidden`` optionnel (le modèle du
    notebook le supporte). Les experts non-prunés conservent leur hidden.
    """
    keep = sorted(keep)
    if not keep:
        raise ValueError("prune_neurons: la liste `keep` ne peut pas être vide.")
    keep_t = torch.tensor(keep, dtype=torch.long)
    e = model.experts[expert]

    # reconstruire un modèle de la même classe
    new_model = type(model)(d_in=model.d_in, temperature=model.temperature)
    # re-ajouter les experts, chacun avec SON hidden (celui d'origine ou le pruné)
    for idx, old_e in enumerate(model.experts):
        h = len(keep) if idx == expert else old_e.hidden
        new_model.add_expert(classes=old_e.classes, hidden=h)

    with torch.no_grad():
        for idx, (old_e, new_e) in enumerate(zip(model.experts, new_model.experts)):
            if idx == expert:
                # pruné : colonnes gardées
                new_e.fc1.weight.data = old_e.fc1.weight.data[keep_t]
                new_e.fc1.bias.data = old_e.fc1.bias.data[keep_t]
                new_e.fc2.weight.data = old_e.fc2.weight.data[:, keep_t]
                new_e.fc2.bias.data = old_e.fc2.bias.data.clone()
            else:
                # non pruné : copie à l'identique (même hidden)
                new_e.fc1.weight.data = old_e.fc1.weight.data.clone()
                new_e.fc1.bias.data = old_e.fc1.bias.data.clone()
                new_e.fc2.weight.data = old_e.fc2.weight.data.clone()
                new_e.fc2.bias.data = old_e.fc2.bias.data.clone()
        # routeur copié
        new_model.router.weight.data = model.router.weight.data.clone()
        new_model.router.bias.data = model.router.bias.data.clone()

    return new_model


def summarize_model(model: nn.Module) -> dict:
    """Retourne un résumé du modèle : nb de params par composant."""
    return {
        'router': sum(p.numel() for p in model.router.parameters()),
        'experts': [sum(p.numel() for p in exp.parameters()) for exp in model.experts],
        'total': model.n_params(),
    }


def analyze_routing(model: nn.Module, datasets: dict, split: str = 'val'):
    """Compte vers quel expert le routeur envoie chaque échantillon d'un split."""
    model.eval()
    X, y = map(torch.as_tensor, datasets[split])
    loader = DataLoader(TensorDataset(X, y), batch_size=256)
    counts = torch.zeros(len(model.experts), dtype=torch.long)
    mean_w = torch.zeros(len(model.experts))
    n = 0
    with torch.no_grad():
        for Xb, _ in loader:
            model(Xb)
            w = model.last_routing
            counts += w.argmax(1).bincount(minlength=len(model.experts))
            mean_w += w.sum(0)
            n += len(Xb)
    mean_w = mean_w / n
    print(f"[routeur] expert choisi par argmax sur {n} échantillons : {counts.tolist()}")
    print(f"[routeur] poids softmax moyen : {mean_w.tolist()}")
    return counts
