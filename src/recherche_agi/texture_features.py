"""Caractéristiques de texture + couleur pour les patches COCO stuff."""
import numpy as np


def color_texture_features(patch_rgb, bins=8):
    """Extrait des caractéristiques couleur + texture d'un patch RGB.

    - Couleur : moyennes + écarts R,G,B ; histogrammes R,G,B (bins)
    - Texture : gradient moyen, variance, contraste local (LBP simplifié)
    """
    p = np.asarray(patch_rgb, float)
    if p.ndim == 2:
        p = np.stack([p]*3, axis=-1)
    R, G, B = p[...,0], p[...,1], p[...,2]

    feats = []
    # --- couleur ---
    for c in (R, G, B):
        feats.append(c.mean()/255.0)
        feats.append(c.std()/255.0)
        hist, _ = np.histogram(c, bins=bins, range=(0,255))
        feats.extend(hist/hist.sum())   # histogramme normalisé
    # --- texture ---
    gray = p.mean(axis=-1)
    feats.append(gray.std()/255.0)                    # contraste global
    # si le patch est trop petit pour le gradient (1x1), on met 0
    if gray.shape[0] >= 2 and gray.shape[1] >= 2:
        gy, gx = np.gradient(gray)
        feats.append(np.sqrt(gx**2 + gy**2).mean()/255.0) # énergie du gradient
        feats.append(np.abs(gx).mean()/255.0)             # gradient horizontal
        feats.append(np.abs(gy).mean()/255.0)             # gradient vertical
        feats.append((gray[:,1:]*gray[:,:-1]).mean()/255.0**2)  # co-occurrence
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])
    return np.array(feats, float)


def extract_all_features(patches_by_class_rgb, bins=8):
    """Applique color_texture_features à chaque patch de chaque classe.

    patches_by_class_rgb : {classe: array(N, H, W, 3)}
    """
    out = {}
    for cls, X in patches_by_class_rgb.items():
        feats = np.array([color_texture_features(x, bins) for x in X])
        out[cls] = feats
    return out
