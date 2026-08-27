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
│   ├── mnist_coco_texture.ipynb       # texture + couleur (features stuff)
│   ├── mnist_coco_message_passing.ipynb # message passing (structuration + inférence)
│   ├── mnist_coco_topdown.ipynb       # projection top-down guidée (masque 1×1)
│   ├── mnist_coco_multiinst.ipynb     # extraction multi-instances (masques multiples)
│   └── mnist_coco_hopfield.ipynb      # modern hopfield network (remplace le WTA)
├── src/recherche_agi/
│   ├── data.py                        # chargement MNIST (référence)
│   ├── unsupervised.py                # AnchorNeurons, WTA, fatigue, Physarum
│   ├── evolutive_coco.py              # HierarchicalCOCO (création de couches)
│   ├── texture_features.py            # features couleur + texture
│   ├── message_passing.py             # message passing sur graphe (structuration + inférence)
│   ├── skip_connections.py            # skip connections transversales auto-régulées
│   ├── topdown_projection.py          # projection top-down guidée (masque 1×1)
│   ├── modern_hopfield.py             # MHN : softmax(βWx) remplace le WTA
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

---

# 🔀 Skip Connections auto-régulées + Message Passing (structuration + inférence)

## Skip connections (module `skip_connections.py`)
Le réseau tisse des **connexions transversales** auto-régulées (graphe petit-monde) :
1. **Pousse synaptique** : connexion candidate C1 → C_{k+m} (conductance faible)
2. **Validation par la surprise** : si la connexion réduit la surprise, le tube
   se renforce (flux de résonance)
3. **Élagage Physarum** : les connexions sans flux sont asséchées et supprimées

Résultat : ~500 connexions inter-couches, préservant les détails fins (bords)
pour les couches profondes (concept).

## Message passing PENDANT l'entraînement (structuration, `step_image`)
Pendant l'apprentissage, les nœuds s'échangent des messages pour former des
représentations stables :
- **Consensus local** : un nœud apprend en tenant compte de la résonance de ses
  voisins (pas isolément)
- **Inhibition** : les nœuds à forte erreur empêchent l'apprentissage uniforme
  (maintien de la compétition WTA)
- **Tubes Physarum** : la conductance W_adj est mise à jour par la co-activation

Résultat : la surprise descend (1.6 → 0.2), le modèle apprend des structures
spatiales stables, pas du bruit isolé.

## Message passing d'INFÉRENCE (reconstruction)
À la reconstruction, les poids sont gelés, le MP nettoie l'image :
- **Harmonisation spatiale** : les nœuds voisins diffusent leur activation →
  une région homogène opte pour la même décision (sous-pixels bruités éliminés)
- **Propagation top-down** : le méta-neurone de C8 diffuse son signal via les
  skip connections vers C1 → masque continu et unifié

## Validation sur PLUSIEURS images (3 exemples)
Le notebook `mnist_coco_message_passing.ipynb` montre la reconstruction par
couche sur **3 images différentes** (8 couches, ~500 skip connections) :
| Couches | Effet |
|---|---|
| C1-C3 | bruitées (détails bas niveau) |
| C5-C8 | zones grandes et homogènes (lissage) — murs/sol uniformes |

=> **Abstraction + lissage** : les couches profondes produisent des catégories
sémantiques (mur, sol) au lieu des couleurs réelles, et le message passing
harmonise spatialement chaque couche.

---

# ⬇️ Projection Top-Down guidée (masque sémantique 1×1)

Le module `topdown_projection.py` **marie l'abstraction sémantique des couches
profondes (C8) avec la résolution spatiale de C1**.

## Pipeline
1. **Sélection** : un neurone d'ancrage actif en couche profonde (ex. le neurone
   C8 responsable d'un objet)
2. **Back-propagation top-down** : on propage le signal unitaire vers le bas via
   les skip-connections et les tubes Physarum validés
3. **Filtre spatial** : C1 agit comme un filtre haute résolution (chaque patch
   est marqué s'il reçoit le rétro-signal)
4. **Masque final** : masque binaire net (0/1) où l'objet est unifié à l'échelle
   du pixel (1×1)

## Résultat (notebook `mnist_coco_topdown.ipynb`, 2 exemples)
| Exemple | Neurone C8 | Masque 1×1 |
|---|---|---|
| 1 (salon) | #7 | 10009/16960 patches (~59%) — isole télévision, fenêtres, meubles |
| 2 | #2 | 8989/23360 (~38%) |

=> Le masque top-down est **net et binaire**, isolant des structures de l'image
(meubles, fenêtres, étagères) via la back-propagation depuis le neurone C8.

## Correction clé
Bug d'adjacence : quand le nb de patches ne forme pas une grille carrée parfaite
(`gh*gw > n`), l'adjacence référençait des indices hors bornes. Corrigé :
`build_grid_adjacency(gh, gw, n)` borne les indices au nombre réel de patches.

---

# 🗂️ Extraction multi-instances (masques d'objets multiples)

Le module `topdown_projection.py` (fonction `multi_instance_topdown`) effectue
la projection **séparément pour chaque neurone actif en C8** → un **dictionnaire
de masques distincts** (une instance par neurone).

## Résultat (notebook `mnist_coco_multiinst.ipynb`)
Sur une image de salon (8 couches, C8=20 neurones) :
| Instance (neurone C8) | % image | Élément de la scène |
|---|---|---|
| #7 | 24% | fenêtres + murs clairs |
| #11 | 21% | téléviseur |
| #0 | 19% | chaises + table + sol |
| #19 | 7% | cuisine |
| #15 | 7% | plafond |

=> **Chaque neurone C8 code une instance distincte et cohérente** (mobilier,
fenêtres, sol, cuisine), comme demandé : Masque 1 = Mobilier, Masque 2 =
Fenêtres/Lumière, Masque 3 = Sol/Arrière-plan.

## Correction clé
La back-propagation top-down produisait des signaux C1 **identiques** pour tous
les neurones C8 (les skip connections étaient trop redondantes). Corrigé : le
masque d'une instance = les patches dont le neurone C8 correspondant est le
gagnant (activation directe), naturellement distinct par neurone.

---

# 🧲 Modern Hopfield Network (remplace le WTA)

Le module `modern_hopfield.py` remplace le **WTA (argmax brutal)** par la
**dynamique de rappel de Modern Hopfield** :
$$z = \\text{softmax}(\\beta \\cdot W x)$$

- **β (inverse de température)** : levier de contrôle ultime
  - β → ∞ : retrouve un **WTA dur** (1 neurone à 100%)
  - β modéré : **consensus continu et lissé** (élimine le bruit sans perdre la compétition)
- **Reconstruction** : $x_{rec} = W^T z$ (combinaison convexe des motifs)
- **Surprise continue** : $\\mathcal{S}_{auto} = \\|x - W^T \\cdot \\text{softmax}(\\beta W x)\\|^2$ — dérivable, sans faux pics de quantification
- **Plasticité Oja** : pondérée par $z$ continu (converge vers les attracteurs)

## Effet de β (notebook `mnist_coco_hopfield.ipynb`)
| β | Distribution z |
|---|---|
| 0.1 | consensus lissé (toutes les activations contribuent) |
| 1-5 | compromis |
| 50-500 | WTA dur (1 neurone dominant) |

## Intégration (le reste de l'architecture inchangé)
- Surprise continue pour déclencher la création de couches (plateau propre)
- Oja pondérée par z continu
- Message passing régularise les vecteurs d'activations lissés z
- Le pipeline s'entraîne : surprise 1.559 → 0.614

---

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
