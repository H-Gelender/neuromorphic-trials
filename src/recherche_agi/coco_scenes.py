"""Scènes COCO Stuff — classes et palette pour le scale.

Classes stuff les plus fréquentes de COCO Stuff (indices -> noms).
"""
import numpy as np

# Classes stuff COCO les plus fréquentes (indice -> nom)
STUFF_CLASSES = {
    0: "unlabeled",
    1: "person",
    2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter",
    15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
    25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
    # stuff
    92: "banner", 93: "blanket", 94: "bridge", 95: "cardboard",
    96: "counter", 97: "curtain", 98: "door-stuff", 99: "floor-wood",
    100: "flower", 101: "fruit", 102: "gravel", 103: "hair",
    104: "hill", 105: "house", 106: "light", 107: "mirror-stuff",
    108: "net", 109: "pillow", 110: "platform", 111: "playingfield",
    112: "railing", 113: "road", 114: "roof", 115: "sand", 116: "sea",
    117: "shelf", 118: "sky-other", 119: "skyscraper", 120: "snow",
    121: "solid-other", 122: "stairs", 123: "streetlight", 124: "swimming pool",
    125: "tent", 126: "towel", 127: "tree", 128: "truck", 129: "wall-brick",
    130: "wall-stone", 131: "wall-tile", 132: "wall-wood", 133: "water-other",
    134: "window-blind", 135: "window-other", 136: "tree-merged",
    137: "fence-merged", 138: "ceiling-merged", 139: "sky-other-merged",
    140: "cabinet-merged", 141: "table-merged", 142: "floor-other-merged",
    143: "pavement-merged", 144: "mountain-merged", 145: "grass-merged",
    146: "dirt-merged", 147: "paper-merged", 148: "food-other-merged",
    149: "building-other-merged", 150: "rock-merged", 151: "wall-other-merged",
    152: "rug-merged", 153: "wall-merged", 154: "cliff-merged",
    155: "plant-other-merged", 156: "plastic-merged", 157: "glass-merged",
    158: "ceiling-tile-merged", 159: "metal-merged", 160: "wood-merged",
    161: "misc-merged", 162: "sky-other-fused", 163: "ground-other-merged",
    164: "water-other-merged", 165: "earth-merged", 166: "tree-other-merged",
}

# palette de couleurs (indice -> (r,g,b)) pour visualiser les masques
PALETTE = {
    113: (128, 128, 128),   # road (gris)
    118: (128, 128, 0),     # sky (olive)
    127: (0, 128, 0),       # tree (vert)
    129: (128, 0, 0),       # wall-brick (marron)
    145: (0, 128, 0),       # grass
    105: (128, 0, 128),     # house
    133: (0, 0, 128),       # water
    116: (0, 0, 128),       # sea
    123: (255, 255, 0),     # streetlight
}


class CocoScenes:
    """Utilitaire pour construire des scènes COCO à partir des patches."""

    def __init__(self, patches_by_class: dict, patch_size: int = 32):
        self.patches = patches_by_class   # {classe: array(N, d)}
        self.patch_size = patch_size
        # dimension = patch_size*patch_size
        d = next(iter(patches_by_class.values())).shape[1]
        self.side = int(round(d ** 0.5))

    def build_scene(self, composition: dict, width: int, gap: int = 8):
        """Construit une scène : composition {classe: nb_instances} + fond noir.

        Retourne (image, liste des vraies boxes (x0,x1)).
        """
        n_channels = 1
        img = np.zeros((self.side, width), dtype=np.float32)
        true_boxes = []
        x = gap
        for cls, count in composition.items():
            for _ in range(count):
                # prendre un patch aléatoire de cette classe
                X = self.patches[cls]
                idx = int(np.random.default_rng(len(true_boxes)).integers(0, len(X)))
                patch = X[idx].reshape(self.side, self.side)
                img[:, x:x+self.side] = patch
                true_boxes.append((x, x+self.side))
                x += self.side + gap
        return img, true_boxes

    def build_scene_from_image(self, image, stuff_map, classes, width=None):
        """Construit une scène à partir d'une vraie image COCO (les patches
        extraits d'une image réelle)."""
        # conversion image -> gris
        if image.ndim == 3:
            img_g = image.mean(axis=2)
        else:
            img_g = image
        # masque binaire par classe -> on garde les pixels de chaque classe
        scene = np.zeros_like(img_g, dtype=np.float32)
        true_boxes = []
        for cls in classes:
            mask = stuff_map == cls
            if mask.sum() == 0: continue
            ys, xs = np.where(mask)
            if len(xs) == 0: continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            # crop de la région de la classe
            region = img_g[y0:y1, x0:x1]
            scene[y0:y1, x0:x1] = region
            true_boxes.append((x0, x1))
        return scene, true_boxes
