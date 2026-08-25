"""Service de callbacks d'entraînement.

Contient les callbacks appelés à intervalle régulier pendant l'entraînement
(logging, affichage des erreurs de validation, etc.).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

__all__ = ["EpochCallback", "MisclassificationCallback"]


class EpochCallback:
    """Appelle ``fn(epoch, metrics, model)`` tous les ``every_n`` epochs."""

    def __init__(self, every_n: int = 5, fn=None):
        self.every_n = every_n
        self.fn = fn

    def __call__(self, epoch: int, metrics: dict, model: nn.Module):
        if self.fn and (epoch + 1) % self.every_n == 0:
            self.fn(epoch, metrics, model)


class MisclassificationCallback(EpochCallback):
    """Affiche, tous les ``every_n`` epochs, les images de validation mal classées.

    Pour chaque erreur on montre le chiffre réel (titre) et le chiffre prédit,
    en rouge si la prédiction est fausse.
    """

    def __init__(self, X_val, y_val, every_n: int = 5, max_examples: int = 8,
                 denorm=None):
        super().__init__(every_n)
        self.X_val = torch.as_tensor(X_val)
        self.y_val = torch.as_tensor(y_val)
        self.max_examples = max_examples
        self.denorm = denorm  # callable pour dénormaliser (None = brut)

    def __call__(self, epoch: int, metrics: dict, model: nn.Module):
        if (epoch + 1) % self.every_n != 0:
            return
        model.eval()
        with torch.no_grad():
            pred = model(self.X_val).argmax(1)
        err_idx = (pred != self.y_val).nonzero().flatten()
        print(f"[callback epoch {epoch + 1}] {len(err_idx)} erreur(s) "
              f"sur {len(self.X_val)} échantillons de validation")
        if len(err_idx) == 0:
            print("  → aucune erreur, rien à afficher")
            return
        idx = err_idx[:self.max_examples].tolist()
        X = self.X_val[idx]
        if self.denorm is not None:
            X = self.denorm(X)
        y_true = self.y_val[idx].tolist()
        y_pred = pred[idx].tolist()

        cols = min(4, len(idx))
        rows = math.ceil(len(idx) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows))
        axes = np.array(axes).reshape(-1)
        for i in range(len(idx)):
            ax = axes[i]
            ax.imshow(X[i].squeeze(), cmap='gray')
            ok = y_true[i] == y_pred[i]
            ax.set_title(f"réel {y_true[i]} / préd {y_pred[i]}", fontsize=8,
                         color='green' if ok else 'red')
            ax.axis('off')
        for i in range(len(idx), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
