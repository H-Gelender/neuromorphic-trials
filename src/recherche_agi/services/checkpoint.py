"""Service de persistance des modèles (checkpoints).

Permet de sauvegarder / recharger l'état d'un modèle MoE. On sauvegarde le
state_dict (poids) + les métadonnées d'architecture et d'entraînement. Les
classes de modèle étant définies dans le notebook, on ne sauvegarde PAS la
classe elle-même : on sauvegarde les poids et une description de l'architecture.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn as nn

__all__ = ["save_model", "load_model_state", "list_checkpoints"]


def save_model(model: nn.Module, path: str, *, architecture: dict | None = None,
               metadata: dict | None = None) -> str:
    """Sauvegarde un modèle dans un fichier .pt (state_dict + infos).

    Args:
        model: le modèle à sauvegarder.
        path: chemin de destination (extension .pt conseillée).
        architecture: dict décrivant l'archi (d_in, n_experts, hidden, ...).
        metadata: infos libres (dataset, accuracy, date, ...).

    Returns:
        le chemin d'écriture effectif (gère le suffixe .pt).
    """
    if not path.endswith('.pt'):
        path = path + '.pt'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # métadonnées par défaut
    arch = dict(architecture or {})
    arch.setdefault('n_params', model.n_params())
    meta = dict(metadata or {})
    meta.setdefault('saved_at', time.strftime('%Y-%m-%d %H:%M:%S'))

    torch.save({
        'state_dict': model.state_dict(),
        'architecture': arch,
        'metadata': meta,
    }, path)
    return path


def load_model_state(path: str):
    """Charge un checkpoint .pt et retourne un dict.

    Returns:
        dict avec les clés 'state_dict', 'architecture', 'metadata'.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'state_dict' not in ckpt:
        raise ValueError(f"Checkpoint invalide (pas de state_dict): {path}")
    return ckpt


def list_checkpoints(directory: str) -> list[str]:
    """Liste les fichiers .pt dans un répertoire (triés par date)."""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return [os.path.join(directory, f) for f in files]
