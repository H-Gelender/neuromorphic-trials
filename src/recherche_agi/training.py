"""Entraînement — boucle d'entraînement, early stopping, évaluation."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .services.callbacks import MisclassificationCallback

__all__ = ["EarlyStopping", "train_tiny_moe", "evaluate", "train_expert_on_dataset",
           "train_router", "freeze_experts", "unfreeze_experts", "verify_frozen"]


class EarlyStopping:
    """Arrête l'entraînement si la métrique de validation ne progresse plus."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.wait = 0
        self.best_state: dict | None = None

    def update(self, val: float) -> tuple[bool, bool]:
        improved = (self.best is None) or (
            val > self.best + self.min_delta if self.mode == 'max' else val < self.best - self.min_delta
        )
        if improved:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
        return self.wait >= self.patience, improved


def evaluate(model: nn.Module, X, y) -> float:
    """Précision de classification sur (X, y)."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X)).argmax(1)
        return (pred == torch.as_tensor(y)).float().mean().item()


def train_tiny_moe(model: nn.Module, datasets: dict, epochs: int = 30,
                   batch_size: int = 128, lr: float = 1e-3, every_n: int = 5,
                   callback=None, early_stop: EarlyStopping | None = None,
                   fixed_experts: list[int] | None = None,
                   verbose: bool = True) -> dict:
    """Entraîne le TinyMoE. datasets = {'train': (X,y), 'val': (X,y)}.

    - `callback` : appelé tous les `every_n` epochs avec (epoch, metrics, model).
    - `early_stop` : EarlyStopping sur la précision de validation.
    """
    Xtr, ytr = map(torch.as_tensor, datasets['train'])
    Xva, yva = map(torch.as_tensor, datasets['val'])
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256)

    # Exclure les paramètres des experts gelés (si fournis) de l'optimizer
    fixed_set = set(fixed_experts or [])
    params = []
    for name, p in model.named_parameters():
        if name.startswith('experts.'):
            try:
                idx = int(name.split('.')[1])
            except Exception:
                idx = None
            if idx is not None and idx in fixed_set:
                continue
        params.append(p)
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        tl, n, corr = 0.0, 0, 0
        for Xb, yb in train_loader:
            opt.zero_grad()
            out = model(Xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(Xb)
            corr += (out.argmax(1) == yb).sum().item()
            n += len(Xb)
        train_acc = corr / n

        model.eval()
        with torch.no_grad():
            vcorr, vn = 0, 0
            for Xb, yb in val_loader:
                vcorr += (model(Xb).argmax(1) == yb).sum().item()
                vn += len(Xb)
        val_acc = vcorr / vn

        history['epoch'].append(epoch)
        history['train_loss'].append(tl / n)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        metrics = {'epoch': epoch, 'train_loss': tl / n, 'train_acc': train_acc, 'val_acc': val_acc}
        if verbose and (epoch + 1) % every_n == 0:
            print(f"epoch {epoch + 1:3d} | train_loss {metrics['train_loss']:.4f} "
                  f"| train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")
        if callback:
            callback(epoch, metrics, model)
        if early_stop:
            stop, improved = early_stop.update(val_acc)
            if improved:
                early_stop.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if stop:
                print(f"[early stop] arrêt à l'epoch {epoch + 1}")
                break

    if early_stop and early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
    return history


def train_expert_on_dataset(model: nn.Module, expert_idx: int, datasets: dict,
                            split: str = 'train', val_split: str = 'val',
                            epochs: int = 10, batch_size: int = 128, lr: float = 1e-3,
                            route_threshold: float = 0.1,
                            visualize: bool = True) -> dict:
    """Entraîne uniquement l'expert `expert_idx` sur les exemples fournis.

    - Si le routeur n'envoie pas suffisamment d'exemples vers l'expert
      (moyenne des poids < `route_threshold`), on saute l'entraînement.
    - On force le routage vers l'expert pendant l'entraînement.
    - Retourne l'historique simple {'epoch', 'loss'}.
    """
    Xtr, ytr = map(torch.as_tensor, datasets[split])
    Xva, yva = map(torch.as_tensor, datasets[val_split])

    # calculer poids moyens du routeur vers cet expert
    with torch.no_grad():
        w = model.routing_weights(Xtr)
        mean_w = float(w[:, expert_idx].mean().item())
    if mean_w < route_threshold:
        print(f"[train_expert] mean routing weight for expert {expert_idx} = {mean_w:.4f} < {route_threshold} → skip training")
        return {'skipped': True, 'mean_routing': mean_w}

    # optimizer sur les paramètres de l'expert uniquement
    params = [p for p in model.experts[expert_idx].parameters() if p.requires_grad]
    if len(params) == 0:
        print(f"[train_expert] expert {expert_idx} has no trainable params (frozen). Skipping.")
        return {'skipped': True, 'frozen': True}

    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    history = {'epoch': [], 'loss': []}
    prev_forced = model.forced_expert
    model.forced_expert = expert_idx
    for epoch in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for Xb, yb in loader:
            opt.zero_grad()
            out = model(Xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(Xb)
            n += len(Xb)
        history['epoch'].append(epoch)
        history['loss'].append(tl / n if n else 0.0)

    model.forced_expert = prev_forced

    if visualize:
        cb = MisclassificationCallback(Xva, yva, every_n=1, max_examples=8)
        # appeler le callback une fois pour afficher erreurs sous routage forcé
        cb(0, {'epoch': 'final'}, model)

    return history


# --------------------------------------------------------------------------- #
# Gel / dégel d'experts
# --------------------------------------------------------------------------- #
def freeze_experts(model: nn.Module, indices: list[int]) -> None:
    """Gèle les poids des experts listés (requires_grad=False)."""
    for i in indices:
        for p in model.experts[i].parameters():
            p.requires_grad = False


def unfreeze_experts(model: nn.Module, indices: list[int]) -> None:
    """Dégèle les poids des experts listés (requires_grad=True)."""
    for i in indices:
        for p in model.experts[i].parameters():
            p.requires_grad = True


def verify_frozen(model: nn.Module, indices: list[int]) -> bool:
    """Vérifie que les experts listés sont bien gelés (aucun param entraînable)."""
    ok = True
    for i in indices:
        n_train = sum(1 for p in model.experts[i].parameters() if p.requires_grad)
        frozen = n_train == 0
        print(f"  expert {i} : {'gelé ✓' if frozen else f'⚠ {n_train} paramètres entraînables'}")
        ok = ok and frozen
    return ok


# --------------------------------------------------------------------------- #
# Entraînement du routeur
# --------------------------------------------------------------------------- #
def train_router(model: nn.Module, datasets: dict, epochs: int = 20,
                 batch_size: int = 128, lr: float = 1e-3, every_n: int = 5,
                 verbose: bool = True) -> dict:
    """Entraîne UNIQUEMENT les poids du routeur à router vers le bon expert.

    Le label de routage pour chaque échantillon est l'expert dont les `classes`
    contiennent la vraie classe `y`. On utilise la cross-entropy sur les logits
    du routeur (softmax). Les experts sont laissés intacts (non entraînés ici).

    datasets = {'train': (X, y), 'val': (X, y)} — toutes les classes 0-1-2-3.
    """
    Xtr, ytr = map(torch.as_tensor, datasets['train'])
    Xva, yva = map(torch.as_tensor, datasets['val'])

    # label de routage cible : expert qui gère la classe y
    expert_labels = torch.tensor(
        [model.expert_for_class(int(c)) for c in ytr], dtype=torch.long)
    val_expert_labels = torch.tensor(
        [model.expert_for_class(int(c)) for c in yva], dtype=torch.long)

    params = [p for p in model.router.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Le routeur n'a aucun paramètre entraînable.")
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, expert_labels), batch_size=batch_size, shuffle=True)

    history = {'epoch': [], 'loss': [], 'route_acc': []}
    for epoch in range(epochs):
        model.train()
        tl, n, corr = 0.0, 0, 0
        for Xb, eb in loader:
            opt.zero_grad()
            logits = model.router(Xb.flatten(1)) / model.temperature
            loss = lossf(logits, eb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(Xb)
            corr += (logits.argmax(1) == eb).sum().item()
            n += len(Xb)
        route_acc = corr / n
        history['epoch'].append(epoch)
        history['loss'].append(tl / n)
        history['route_acc'].append(route_acc)

        model.eval()
        with torch.no_grad():
            vlogits = model.router(Xva.flatten(1)) / model.temperature
            v_acc = (vlogits.argmax(1) == val_expert_labels).float().mean().item()

        if verbose and (epoch + 1) % every_n == 0:
            print(f"  [router] epoch {epoch + 1:3d} | loss {tl/n:.4f} | "
                  f"train route_acc {route_acc:.4f} | val route_acc {v_acc:.4f}")

    return history
