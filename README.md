# Module Adaptatif — Recherche AGI (scale vers COCO Stuff)

Système **100% non supervisé** d'apprentissage incrémental et **hiérarchique
profond**, entraîné sur **COCO Stuff** (images de scènes réelles). Architecture
basée sur des **neurones d'ancrage** (SOM Hebbien), la **fatigue homéostatique**,
l'**élagage Physarum**, la **neurogenèse**, le **message passing** sur graphe et
la **création dynamique de couches**.

## 🧠 Architecture

```
[ Image COCO ] → patches (4×4) → features (couleur + texture)
      │
      ▼
[ Couche C1 ]  AnchorNeurons (max 2000) — neurogenèse progressive
      │  + fatigue homéostatique + élagage Physarum
      ▼  (spawn par plateau de surprise)
[ Couche C2 ]  neurones /2
      ▼
[ Couche C3 ]  neurones /4 ... (hiérarchie profonde)
      │
      ▼
[ Message Passing ]  consensus spatial + inhibition de surprise (conductivité Physarum)
      │
      ▼
[ Segmentation / Reconstruction ]
```

- **AnchorNeurons** : neurones qui concourent sur les features, le gagnant
  s'ajuste vers l'entrée (Oja/Kohonen). Clusters émergents sans étiquettes.
- **Neurogenèse sélective** : ajoute un neurone seulement si le patch est mal
  représenté (surprise élevée).
- **Hiérarchie profonde** : quand la surprise ne descend plus (plateau), une
  **nouvelle couche** est créée (neurones divisés par 2).
- **Message passing** : les nœuds du graphe s'échangent des messages
  (résonance = consensus spatial, inhibition = frontières sur les contours).
- **Élagage Physarum** : atrophie les neurones sous-utilisés, la conductivité
  pilote l'adjacence du graphe.

## 📁 Structure

```
recherche-agi/
├── notebooks/
│   ├── mnist_coco_evolutive.ipynb     # hiérarchie profonde (création de couches)
│   ├── mnist_coco_monitor.ipynb       # monitoring (perf + reconstruction + archi)
│   ├── mnist_coco_report.ipynb        # compte rendu (images, classes, temps CPU)
│   └── mnist_coco_texture.ipynb       # texture + couleur (features stuff)
├── src/recherche_agi/
│   ├── data.py                        # chargement MNIST (référence)
│   ├── unsupervised.py                # AnchorNeurons, WTA, fatigue, Physarum
│   ├── evolutive_coco.py              # HierarchicalCOCO (création de couches)
│   ├── texture_features.py            # features couleur + texture
│   ├── message_passing.py             # message passing sur graphe
│   ├── online_training.py             # callback d'équilibre (ΔW/ΔS/ΔD)
│   ├── monitor.py                     # visuels architecture évolutive
│   ├── report.py                      # compte rendu (TrainingTracker)
│   ├── visualize.py                   # visuels architecture + évolution
│   ├── visualize_coco_images.py       # visuels images COCO claires
│   ├── coco_pipeline.py               # entraînement classe par classe
│   ├── coco_scenes.py                 # scènes COCO (classes, palette)
│   └── stable_layers.py               # cycle respiratoire (référence)
└── README.md
```

## 📊 Résultats clés

### Hiérarchie profonde (création de couches)
Le modèle **crée ses couches lui-même** via un **spawn par plateau de surprise** :
- 7 couches créées, tailles `[54, 82, 73, 53, 50, 62, 31]` (neurones divisés)
- Abstraction croissante : C1 (détails) → C7 (concepts abstraits), comme un CNN.

### Message passing (segmentation émergente)
- **Résonance** : les voisins activés s'envoient un biais → lisse le bruit de
  quantification (consensus local).
- **Inhibition** : les nœuds surprenants (contours) créent des frontières nettes.
- Reconstruction 4×4 : le message passing **lisse visiblement** l'image.

### Compte rendu d'entraînement (CPU)
| Métrique | Valeur (171 classes) |
|---|---|
| Images passées | 171 |
| Patches traités | 1 174 563 |
| Temps total CPU | ~8-20 min |
| Débit | 980-2300 patches/s |
| Neurones finaux | 400-800 |
| Couches | 2-7 |

## 🚀 Exécution

```bash
cd C:/Users/henry/Desktop/workspace/recherche-agi
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/mnist_coco_evolutive.ipynb
```

## 🔑 Leçons

1. **Entraînement classe par classe** (jamais tous les labels d'un coup).
2. **Caractéristiques = la nature du problème** : texture/couleur pour les scènes,
   pixels pour les formes.
3. **Hiérarchie profonde > couches larges** : le modèle crée des couches
   (neurones divisés) au lieu de grossir.
4. **Message passing** : lissage par consensus spatial + frontières par inhibition.
5. Le goulot COCO est le **téléchargement réseau** (pas le calcul) — cacher en
   local les données.
