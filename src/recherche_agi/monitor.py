"""Monitoring visuel de l'architecture évolutive.

Dessine le modèle comme un graphe multi-couches :
- chaque COUCHE = une rangée de neurones (taille = nb de neurones)
- connexions BOTTOM-UP (C1 -> C2) : traits gris/bleus
- connexions de RÉTROACTION (C2 -> C1) : traits rouges/pointillés (top-down)
- affiche le nb de poids par couche + total
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import imageio.v2 as imageio


def draw_architecture(model, ax=None, title=None):
    """Dessine l'architecture du modèle (AnchorNeurons) en graphe multi-couches.

    model : DynamicAnchorNeurons (une couche) avec .W et .co_act.
    Pour l'instant on dessine la couche C1 + la couche C2 (si présente via le
    contrôleur) : ici on représente C1 et ses connexions internes, et on place
    une éventuelle C2 comme colonne supérieure.
    """
    W = model.W
    n = len(W)                      # nb de neurones
    d = W.shape[1]                  # dim des poids
    co = model.co_act               # connectivité interne (bottom-up entre neurones C1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure

    # --- Couche 1 (bottom) : rangée de neurones ---
    x1 = np.linspace(0.1, 0.9, n)
    y1 = 0.2
    # taille des neurones ~ nombre d'activations
    acts = model.activations if hasattr(model, 'activations') else np.ones(n)
    sizes = 60 + 300 * acts / (acts.max() + 1e-8)
    ax.scatter(x1, np.full(n, y1), s=sizes, c='teal', edgecolors='black', zorder=3)
    ax.text(0.5, y1-0.12, f"C1 : {n} neurones", ha='center', fontsize=12, fontweight='bold')

    # --- connexions BOTTOM-UP internes C1 (co_act) ---
    maxc = co.max() if co.max() > 0 else 1
    n_bu = 0
    for i in range(n):
        for j in range(i+1, n):
            if co[i, j] > 0.05 * maxc:
                wgt = co[i, j] / maxc
                ax.plot([x1[i], x1[j]], [y1, y1], color='steelblue',
                        alpha=0.2+0.5*wgt, lw=0.3+1.5*wgt, zorder=1)
                n_bu += 1

    # --- Couche 2 (haut) : la couche abstraite / méta ---
    # on simule une C2 de n2 neurones reliés (bottom-up depuis C1)
    n2 = min(n, max(5, n//4))       # C2 plus compacte
    x2 = np.linspace(0.1, 0.9, n2)
    y2 = 0.75
    ax.scatter(x2, np.full(n2, y2), s=150, c='orange', edgecolors='black', zorder=3)
    ax.text(0.5, y2+0.08, f"C2 : {n2} neurones (abstraite/méta)", ha='center',
            fontsize=12, fontweight='bold')

    # --- connexions BOTTOM-UP C1 -> C2 (feedforward) ---
    # chaque neurone C2 reçoit de quelques C1 (échantillonnés)
    n_fw = 0
    for j in range(n2):
        sources = np.linspace(0, n-1, n//n2 if n//n2 > 0 else 1).astype(int)
        for i in sources:
            ax.plot([x1[i], x2[j]], [y1, y2], color='steelblue', alpha=0.25,
                    lw=0.4, zorder=2)
            n_fw += 1

    # --- connexions de RÉTROACTION C2 -> C1 (top-down) ---
    # quelques retours depuis C2 vers C1 (rouge pointillé)
    n_fb = 0
    for j in range(0, n2, max(1, n2//8)):
        i = int(np.linspace(0, n-1, n2)[j])
        ax.plot([x2[j], x1[i]], [y2, y1], color='crimson', alpha=0.6,
                lw=1.2, ls='--', zorder=2)
        n_fb += 1

    # --- infos poids ---
    ax.text(0.02, 0.95, f"Poids C1 : {n}x{d} = {n*d:,}\n"
                        f"Connexions bottom-up: {n_bu}\n"
                        f"Feedforward C1→C2: {n_fw}\n"
                        f"Rétroaction C2→C1: {n_fb}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13)
    # légende
    ax.plot([], [], color='steelblue', lw=2, label='Bottom-up (feedforward)')
    ax.plot([], [], color='crimson', lw=2, ls='--', label='Rétroaction (top-down)')
    ax.legend(loc='lower right', fontsize=9)
    return fig


def architecture_frame(model, title=None, figsize=(12, 7)):
    """Génère un frame (array RGB) pour le GIF."""
    fig = draw_architecture(model, title=title)
    fig.set_size_inches(*figsize)
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=70)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def save_architecture_gif(model_history, out='arch_evolution.gif', duration=0.6,
                          every=1, titles=None):
    """Assemble un GIF de l'évolution de l'architecture.

    model_history : liste de modèles (snapshots) au fil de l'entraînement.
    """
    frames = []
    for k, model in enumerate(model_history[::every]):
        title = titles[k*every] if titles else f"Étape {k*every}"
        frames.append(architecture_frame(model, title=title))
    imageio.mimsave(out, frames, duration=duration)
    return out
