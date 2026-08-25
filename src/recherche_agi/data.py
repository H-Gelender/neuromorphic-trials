"""Traitement des données — chargement MNIST et filtrage par chiffres."""
from __future__ import annotations

import os
from typing import Iterable

import torch
from torchvision import datasets, transforms

__all__ = ["load_mnist", "filter_by_digits"]


def load_mnist(root: str | None = None, download: bool = True,
               normalize: bool = True):
    """Télécharge / charge MNIST et retourne les splits (train, test).

    Le dossier ``data/`` est ancré à la racine du projet (quel que soit le cwd),
    en remontant depuis ce fichier de module.
    """
    if root is None:
        # <projet>/src/recherche_agi/data.py -> <projet>/data
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'data', 'mnist')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x),
    ])
    train = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    return train, test


def filter_by_digits(train, test, digits: int | Iterable[int],
                     val_fraction: float = 0.1, seed: int = 42) -> dict:
    """Ne garde que les chiffres demandés et découpe en train/val/test.

    Retourne un dict {"train": (X, y), "val": (X, y), "test": (X, y)} avec des
    tenseurs torch (X float32, y int64).
    """
    wanted = {digits} if isinstance(digits, int) else set(digits)
    if not wanted:
        raise ValueError("Au moins un chiffre requis.")

    def _extract(ds):
        y = torch.as_tensor(ds.targets)
        mask = torch.isin(y, torch.tensor(sorted(wanted)))
        idx = torch.nonzero(mask, as_tuple=True)[0]
        X = torch.stack([ds[i][0] for i in idx.tolist()])
        return X, y[idx]

    X_tr, y_tr = _extract(train)
    X_te, y_te = _extract(test)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X_tr), generator=g)
    X_tr, y_tr = X_tr[perm], y_tr[perm]
    n_val = int(len(X_tr) * val_fraction)
    return {"train": (X_tr[n_val:], y_tr[n_val:]),
            "val": (X_tr[:n_val], y_tr[:n_val]),
            "test": (X_te, y_te)}
