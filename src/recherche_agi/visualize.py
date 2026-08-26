"""Visualisation de l'entraînement COCO : ce que voit le système + architecture."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_input(patches_by_class, class_names=None, n_cols=10, out='input_viz.png'):
    """Ce que le système voit : échantillon de patches par classe.

    patches_by_class : {classe: array(N, H, W)} (pixels gris) ou (N,H,W,3).
    """
    n_cls = len(patches_by_class)
    n_rows = n_cls
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]
    for r, (cls, X) in enumerate(sorted(patches_by_class.items())):
        X = np.asarray(X)
        # X peut être (N,H,W), (N,H,W,3), ou un seul patch (H,W)/(H,W,3)
        if X.ndim == 3 and X.shape[0] == 1 and X.shape[1] == X.shape[2]:
            samples = [X[0]]
        elif X.ndim == 4 and X.shape[0] == 1:
            samples = [X[0]]
        elif X.ndim >= 3:
            samples = [X[idx] for idx in range(min(n_cols, len(X)))]
        else:
            samples = [X] * n_cols
        for c in range(n_cols):
            p = samples[c % len(samples)]
            ax = axes[r, c]
            ax.imshow(p, cmap='gray' if p.ndim == 2 else None)
            ax.axis('off')
        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        axes[r, 0].set_ylabel(f"{cls} ({name})", fontsize=8)
    plt.suptitle("Ce que le système voit (échantillon de patches par classe)")
    plt.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out


def visualize_architecture(anchors, out='arch_viz.png'):
    """Architecture du modèle : nombre de neurones, connectivité (co_act).

    anchors : AnchorNeurons avec .W (n, d) et .co_act (n, n).
    """
    n = len(anchors.W)
    co = anchors.co_act
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) taille / neurones
    axes[0].bar(['neurones', 'dims/poids'], [n, anchors.W.shape[1]])
    axes[0].set_title(f"Architecture : {n} neurones\n({n}x{anchors.W.shape[1]} poids)")
    axes[0].set_ylabel("nombre")

    # 2) matrice de connectivité (co_act)
    im = axes[1].imshow(co, cmap='viridis')
    axes[1].set_title(f"Connectivité (co-activation)\n{int((co>0).sum())} connexions")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 3) graphe des neurones (réseau) — disposition circulaire
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    axes[2].scatter(pos[:,0], pos[:,1], s=20, c='teal')
    maxc = co.max() if co.max() > 0 else 1
    n_edges = 0
    for i in range(n):
        for j in range(i+1, n):
            if co[i,j] > 0.1*maxc:
                axes[2].plot([pos[i,0],pos[j,0]],[pos[i,1],pos[j,1]], color='gray', alpha=0.4, lw=0.5)
                n_edges += 1
    axes[2].set_title(f"Graphe des neurones\n{n_edges} connexions")
    axes[2].set_aspect('equal'); axes[2].axis('off')

    plt.suptitle(f"Architecture du modèle — {n} neurones, {int((co>0).sum())} connexions")
    plt.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out


def visualize_evolution(cb, out='evolution_viz.png'):
    """Évolution des 3 critères d'équilibre au cours de l'entraînement."""
    h = cb.history
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, key, title in [
        (axes[0], 'dW', "A. Variation des poids ||ΔW|| (équilibre si → 0)"),
        (axes[1], 'S', "B. Surprise de reconstruction S (plateau)"),
        (axes[2], 'D', "C. Variation du flux Physarum ||ΔD|| (stabilisé)"),
    ]:
        vals = h[key]
        ax.plot(h['time'][:len(vals)], vals, color='teal')
        ax.set_ylabel(key); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("temps (s)")
    plt.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out
