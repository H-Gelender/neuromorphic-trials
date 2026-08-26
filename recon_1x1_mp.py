# Reconstruction 1x1 avec MESSAGE PASSING (consensus spatial pixel par pixel)
import numpy as np
from datasets import load_dataset
from recherche_agi import color_texture_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

m = np.load('data/coco_stuff/live_model_fine.npz')
W = m['W']  # (800, 35)
rng = np.random.default_rng(0)
palette = rng.integers(0, 255, (len(W), 3))

ds = load_dataset('shunk031/cocostuff','stuff-thing',split='validation',streaming=True,trust_remote_code=True)

def per_pixel_features(img):
    """Features couleur par pixel (vectorisé, sans gradient) -> (H,W,35)."""
    p = img.astype(float)
    H, Wimg = p.shape[0], p.shape[1]
    R, G, B = p[...,0], p[...,1], p[...,2]
    out = np.zeros((H, Wimg, 35))
    for c, ch in enumerate([R,G,B]):
        out[..., c*3] = ch/255.0                     # moyenne (=valeur)
        out[..., c*3+1] = 0.0                        # std (0 pour 1 pixel)
        hist_bin = (ch/255.0*(8)).astype(int).clip(0,7)  # histogramme simplifié
        # one-hot simplifié : placer le pixel dans son bin
        out[..., c*3+2 + hist_bin] = 1.0
    # contraste global 0, gradients 0 (patch 1x1)
    out[..., 27:] = 0.0
    return out

def winner_map_pixel(feats):
    """Carte des neurones gagnants par pixel (H,W)."""
    H, Wimg, D = feats.shape
    flat = feats.reshape(-1, D)
    zn = flat / (np.linalg.norm(flat, axis=1, keepdims=True)+1e-8)
    winners = np.argmax(zn @ W.T, axis=1).reshape(H, Wimg)
    return winners

def message_passing_1x1(winners, n_iter=2):
    """Consensus spatial : mode majoritaire des 4-voisins (filtrage vectorisé)."""
    H, Wimg = winners.shape
    out = winners.copy()
    # empiler les 5 valeurs (soi + 4 voisins) le long d'un axe
    for _ in range(n_iter):
        stack = [out]
        s = np.zeros_like(out)
        s[1:,:] = out[:-1,:]; stack.append(s)   # haut
        s = np.zeros_like(out)
        s[:-1,:] = out[1:,:]; stack.append(s)   # bas
        s = np.zeros_like(out)
        s[:,1:] = out[:,:-1]; stack.append(s)   # gauche
        s = np.zeros_like(out)
        s[:,:-1] = out[:,1:]; stack.append(s)   # droite
        stacked = np.stack(stack, axis=-1)      # (H,W,5)
        # mode le long du dernier axe via bincount vectorisé
        flat = stacked.reshape(-1, 5)
        maxv = flat.max() + 1
        counts = np.zeros((flat.shape[0], maxv))
        for k in range(5):
            np.add.at(counts, (np.arange(flat.shape[0]), flat[:,k]), 1)
        out = counts.argmax(axis=1).reshape(H, Wimg)
    return out

for n, x in enumerate(ds):
    if n >= 1: break
    img = np.array(x['image'])
    feats = per_pixel_features(img)
    print(f"features pixel: {feats.shape}")
    w_raw = winner_map_pixel(feats)
    w_mp = message_passing_1x1(w_raw, n_iter=2)

    def color_map(lab):
        ci = np.zeros((lab.shape[0], lab.shape[1], 3), dtype=np.uint8)
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                ci[i,j] = palette[lab[i,j]]
        return ci

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title("Réelle"); axes[0].axis('off')
    axes[1].imshow(color_map(w_raw)); axes[1].set_title("1x1 brut"); axes[1].axis('off')
    axes[2].imshow(color_map(w_mp)); axes[2].set_title("1x1 + message passing"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('notebooks/figs/coco_1x1_message_passing.png', dpi=80)
    plt.close()
    print("Reconstruction 1x1 + message passing générée")
print("Terminé")
