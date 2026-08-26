"""Rapport de monitoring de l'entraînement COCO.

Génère un compte rendu complet :
- nombre d'images passées, patches, classes vues
- classes par image
- temps d'entraînement (CPU) + throughput
- neurones/couches créés, taille des poids
- répartition des classes
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TrainingTracker:
    """Suit les statistiques de l'entraînement."""

    def __init__(self):
        self.start = time.time()
        self.n_images = 0
        self.n_patches = 0
        self.classes_seen = {}        # classe -> nb de patches
        self.classes_per_image = []   # nb de classes par image
        self.patches_per_image = []   # nb de patches par image
        self.time_per_image = []      # temps par image (s)
        self.last_img_t = None

    def begin_image(self):
        """Appelé au début de chaque image."""
        self.last_img_t = time.time()

    def end_image(self, classes_in_image, n_patches_in_image):
        """Appelé à la fin de chaque image."""
        self.n_images += 1
        self.n_patches += n_patches_in_image
        for c in classes_in_image:
            self.classes_seen[c] = self.classes_seen.get(c, 0) + 1
        self.classes_per_image.append(len(classes_in_image))
        self.patches_per_image.append(n_patches_in_image)
        if self.last_img_t is not None:
            self.time_per_image.append(time.time() - self.last_img_t)

    def elapsed(self):
        return time.time() - self.start

    def summary(self):
        n = max(1, self.n_images)
        return {
            'n_images': self.n_images,
            'n_patches': self.n_patches,
            'n_classes': len(self.classes_seen),
            'classes_per_image_mean': np.mean(self.classes_per_image) if self.classes_per_image else 0,
            'patches_per_image_mean': np.mean(self.patches_per_image) if self.patches_per_image else 0,
            'time_per_image_mean': np.mean(self.time_per_image) if self.time_per_image else 0,
            'elapsed_s': self.elapsed(),
            'throughput_img_s': n / self.elapsed() if self.elapsed() > 0 else 0,
            'throughput_patch_s': self.n_patches / self.elapsed() if self.elapsed() > 0 else 0,
        }


def generate_report(tracker, model, features, out='report.png'):
    """Génère le rapport visuel de monitoring.

    tracker : TrainingTracker
    model   : EvolutiveCOCO (avec .layer, .controller, .history)
    features: dict {classe: array} (pour la distribution)
    """
    s = tracker.summary()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) temps par image
    if tracker.time_per_image:
        axes[0,0].plot(tracker.time_per_image, color='steelblue')
        axes[0,0].set_title(f"Temps par image (moy {s['time_per_image_mean']:.2f}s)")
        axes[0,0].set_xlabel("image"); axes[0,0].set_ylabel("s"); axes[0,0].grid(alpha=0.3)
    else:
        axes[0,0].text(0.5, 0.5, "pas de données", ha='center'); axes[0,0].axis('off')

    # 2) répartition des classes (top 20)
    if features:
        counts = sorted(((len(v), k) for k, v in features.items()), reverse=True)[:20]
        names = [str(k) for _, k in counts][::-1]
        vals = [c for c, _ in counts][::-1]
        axes[0,1].barh(names, vals, color='teal')
        axes[0,1].set_title("Top classes (nb features)")
        axes[0,1].set_xlabel("features")
    else:
        axes[0,1].text(0.5, 0.5, "n/a", ha='center'); axes[0,1].axis('off')

    # 3) évolution des neurones (archi)
    h = model.history
    if h['n_neurons']:
        axes[1,0].plot(h['n_patches'], h['n_neurons'], color='orange')
        axes[1,0].set_title(f"Neurogenèse : {h['n_neurons'][0]} → {h['n_neurons'][-1]} neurones")
        axes[1,0].set_xlabel("patches"); axes[1,0].set_ylabel("neurones"); axes[1,0].grid(alpha=0.3)
    else:
        axes[1,0].text(0.5,0.5,"n/a",ha='center'); axes[1,0].axis('off')

    # 4) résumé texte
    txt = (
        f"COMPTE RENDU D'ENTRAÎNEMENT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Images passées        : {s['n_images']:,}\n"
        f"Patches traités       : {s['n_patches']:,}\n"
        f"Classes vues          : {s['n_classes']}\n"
        f"Classes/image (moy)   : {s['classes_per_image_mean']:.1f}\n"
        f"Patches/image (moy)   : {s['patches_per_image_mean']:.1f}\n"
        f"Temps/image (moy)     : {s['time_per_image_mean']:.2f}s\n"
        f"Temps total (CPU)     : {s['elapsed_s']:.1f}s ({s['elapsed_s']/60:.1f} min)\n"
        f"Débit                 : {s['throughput_img_s']:.1f} img/s\n"
        f"                      : {s['throughput_patch_s']:.1f} patches/s\n"
        f"Neurones finaux       : {model.layer.n_neurons_current}\n"
        f"Couches               : {model.controller.layer_count} (+{model.summary()['archived_layers']} archivée)\n"
        f"Poids/couche          : {model.layer.W.shape[0]}×{model.layer.W.shape[1]} = {model.layer.W.shape[0]*model.layer.W.shape[1]:,}\n"
        f"Poids totaux          : {model.layer.W.size:,}\n"
        f"Connexions (co_act>0) : {int((model.layer.co_act>0).sum())}"
    )
    axes[1,1].text(0.05, 0.95, txt, va='top', ha='left', fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    axes[1,1].axis('off')

    plt.suptitle("Compte rendu d'entraînement COCO (CPU)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out, dpi=85)
    plt.close(fig)
    return out
