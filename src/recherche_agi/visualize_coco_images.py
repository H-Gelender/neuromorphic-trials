"""Visualiser les VRAIES images COCO Stuff en clair (image complète + masque)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# palette simple pour les classes stuff (indice -> couleur)
STUFF_PALETTE = {
    118: (135, 206, 235),   # sky bleu clair
    113: (128, 128, 128),   # road gris
    127: (34, 139, 34),     # tree vert
    129: (160, 82, 45),     # wall-brick marron
    145: (124, 252, 0),     # grass vert clair
    133: (70, 130, 180),    # water bleu acier
    116: (0, 105, 148),     # sea bleu foncé
    120: (255, 255, 255),   # snow blanc
    105: (128, 0, 128),     # house violet
    115: (238, 232, 170),   # sand sable
    114: (139, 69, 19),     # roof brun
    146: (139, 90, 43),     # dirt terre
}


def colormap_mask(stuff_map):
    """Convertit un masque stuff en image RGB colorée."""
    H, W = stuff_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in STUFF_PALETTE.items():
        mask = stuff_map == cls
        rgb[mask] = color
    # les classes non mappées en gris sombre
    known = np.zeros_like(stuff_map, dtype=bool)
    for cls in STUFF_PALETTE:
        known |= (stuff_map == cls)
    rgb[~known] = 40
    return rgb


def visualize_coco_sample(image_rgb, stuff_map, objects=None, out='coco_sample.png',
                          figsize=(14, 5)):
    """Affiche une image COCO complète + son masque stuff coloré + boxes objets.

    image_rgb : (H, W, 3) uint8  (l'image originale, claire et précise)
    stuff_map : (H, W) int       (masque de segmentation stuff)
    objects   : liste de dict {x,y,w,h,name} optionnel
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # 1) image réelle en couleur (claire)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image COCO réelle (couleur)")
    axes[0].axis('off')
    # 2) masque stuff coloré
    mask_rgb = colormap_mask(stuff_map)
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Masque stuff (segmentation)")
    axes[1].axis('off')
    # boxes objets éventuelles
    if objects:
        for o in objects:
            if 'x' in o:
                rect = Rectangle((o['x'], o['y']), o['w'], o['h'],
                                 linewidth=1.5, edgecolor='red', facecolor='none')
                axes[0].add_patch(rect)
                axes[0].text(o['x'], o['y']-2, o.get('name',''), color='red', fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def visualize_patches_rgb(patches_by_class, class_names=None, n_per=4, out='patches_rgb.png'):
    """Affiche les patches en COULEUR, à taille lisible (plus grande).

    patches_by_class : {classe: array(N, H, W, 3)} patches RGB
    """
    n_cls = len(patches_by_class)
    n_cols = n_per
    n_rows = n_cls
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*2.2, n_rows*2.2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]
    for r, (cls, X) in enumerate(sorted(patches_by_class.items())):
        X = np.asarray(X)
        for c in range(n_cols):
            idx = int(np.random.default_rng(r*10+c).integers(0, len(X)))
            p = X[idx].astype(float)
            ax = axes[r, c]
            ax.imshow(p)
            ax.axis('off')
        name = class_names.get(cls, str(cls)) if class_names else str(cls)
        axes[r, 0].set_ylabel(f"{cls} ({name})", fontsize=9)
    plt.suptitle("Patches en COULEUR (agrandis) par classe")
    plt.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out
