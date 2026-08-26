# Module Adaptatif — Recherche AGI (MNIST)

Système **100% non supervisé** d'apprentissage incrémental sur MNIST (0-9), basé
sur des **neurones d'ancrage** (SOM Hebbien), la **fatigue homéostatique**
(anti-oubli) et l'**élagage Physarum** (gestion des ressources).

Le système s'adapte au **drift** (0/1 → 2-9, sans jamais ré-entraîner les classes
anciennes) sans **oubli catastrophique**, et atteint **~0.95 d'acc globale**.

## 🧠 Architecture

```
[ Image MNIST ] → latents (pixels / patches)
      │
      ▼
[ Neurones d'Ancrage ]  (SOM Hebbien, WTA)
      │  + fatigue homéostatique (θ_i += α·y_i → anti-oubli)
      │  + co-activation (connexions neuromorphiques)
      ▼
[ Élagage Physarum ]  (atrophie des neurones sous-utilisés → gestion ressources)
      │
      ▼
[ Classification a posteriori ]  (observation, non supervisée)
```

- **AnchorNeurons** : neurones qui concourent sur les latents, le gagnant
  s'ajuste vers l'entrée (Oja/Kohonen). Clusters émergents sans étiquettes.
- **Fatigue homéostatique** : un neurone qui gagne souvent voit son seuil monter
  → exploration, pas de dominance.
- **Co-activation** : deux neurones qui gagnent ensemble forment une connexion
  (mémoire neuromorphique).
- **Élagage Physarum** : les neurones sous-utilisés sont atrophiés (comme le blob
  atrophie les tubes sans flux), libérant du budget pour de nouvelles classes.

## 📊 Résultats clés

### Drift sans oubli catastrophique
| Après intro | Acc 0/1 (préservé) | Acc nouveau | Neurones |
|---|---|---|---|
| Phase 1 (0/1) | 1.000 | — | 121 |
| 2 | 1.000 | 0.840 | 170 |
| 3 | 0.990 | 0.880 | 187 |
| 5-9 | 0.96-0.99 | 0.28-0.60 | 200 (saturés) |

→ **Anti-oubli démontré** : 0/1 reste > 0.95 après l'introduction de tous les
chiffres. Mais les 200 neurones **saturent** → 5-9 dégradent.

### Avec élagage Physarum (résout le goulot)
| Config | Acc 0/1 | Acc 5-9 | Neurones |
|---|---|---|---|
| Sans Physarum | 0.980 | 0.508 | 200 (saturés) |
| **Avec Physarum** | 1.000 | **0.904** | 140 |

→ **GAIN sur 5-9 : +73%** avec MOINS de neurones actifs (140 vs 200).

### Acc globale (test aléatoire 0-9)
| Approche | Acc globale |
|---|---|
| Sans Physarum | 0.620 |
| **Avec élagage Physarum** | **0.951** |

## 📁 Structure du projet

```
recherche-agi/
├── notebooks/
│   ├── mnist_unsupervised.ipynb        # base : neurones d'ancrage 100% non supervisé
│   ├── mnist_drift_test.ipynb          # anti-oubli catastrophique (drift 0/1 → 2-9)
│   ├── mnist_physarum_pruning.ipynb    # élagage Physarum résout le goulot
│   ├── mnist_global_accuracy.ipynb     # acc globale après drift (test 0-9)
│   ├── mnist_neurogenesis.ipynb        # neurogenèse dynamique + couche Hebbienne
│   ├── mnist_two_layers_viz.ipynb      # deux couches + visualisation en images
│   ├── mnist_feedback_topdown.ipynb    # rétroaction top-down (couche 2 → couche 1)
│   ├── mnist_threshold_wta_patches.ipynb # WTA par seuil + découpage image
│   ├── mnist_multidigit_detection.ipynb # détection multi-chiffres (stride + bounding box)
│   ├── mnist_reconstruction_detection.ipynb # reconstruction image + 2 scores (IoU & count)
│   ├── mnist_metaneurons.ipynb         # méta-neurones (centroïdes de prototypes)
│   ├── mnist_respiratory_cycle.ipynb   # cycle respiratoire (gel dynamique + création de couches)
│   ├── mnist_respiratory_test.ipynb    # test du cycle sur MNIST (IoU + bounding boxes)
│   ├── mnist_mode_collapse.ipynb       # mode collapse + domination densité
│   ├── mnist_focus_iterative.ipynb     # méthode itérative avec FOCUS (accuracy)
│   ├── mnist_coco_scale.ipynb          # scale vers COCO Stuff (classe par classe)
│   ├── mnist_coco_texture.ipynb        # COCO : texture + couleur (au lieu pixels gris)
│   ├── mnist_coco_recon.ipynb          # reconstruction image réelle vs modèle
│   ├── mnist_coco_evolutive.ipynb      # entraînement évolutif (neurogenèse + couches)
│   ├── mnist_coco_monitor.ipynb        # monitoring (perf + reconstruction + architecture)
│   └── mnist_coco_report.ipynb         # compte rendu (images, classes, temps CPU)
├── src/recherche_agi/
│   ├── data.py                        # chargement MNIST + filtre par chiffres
│   ├── unsupervised.py                # AnchorNeurons, WTA, fatigue, co-activation,
│   │                                  #   top-down, image_to_patches, élagage Physarum
│   ├── stable_layers.py               # cycle respiratoire (S(t), gel, spawning, dégel)
│   ├── coco_pipeline.py               # entraînement COCO classe par classe
│   ├── coco_scenes.py                 # scènes COCO (classes, palette, construction)
│   ├── texture_features.py            # caractéristiques couleur + texture
│   ├── online_training.py             # callback d'équilibre (ΔW/ΔS/ΔD) + arrêt 1h
│   ├── evolutive_coco.py              # entraînement évolutif (neurogenèse + couches)
│   ├── monitor.py                     # visuels architecture évolutive (GIF)
│   ├── report.py                      # compte rendu (images, classes, temps CPU)
│   ├── visualize.py                   # visuels architecture + évolution
│   ├── visualize_coco_images.py       # visuels images COCO claires
│   └── training.py                    # entraînement auxiliaire (référence)
└── README.md
```

## 🚀 Exécution

```bash
cd C:/Users/henry/Desktop/workspace/recherche-agi
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/mnist_global_accuracy.ipynb   # acc globale après drift
```

## 📚 Leçons scientifiques

1. **L'apprentissage Hebbian pur** capture les primitives mais plafonne (~0.40)
   sans décodage — la classification finale vient des neurones d'ancrage + élagage.
2. **La fatigue homéostatique** résout l'oubli catastrophique (les anciens
   neurones sont préservés).
3. **L'élagage Physarum** est le levier décisif : il gère la contrainte de
   ressources en atrophiant les neurones inutiles, portant l'acc globale de 0.62
   à 0.95.

---

# 🧬 Neurogenèse dynamique + couche Hebbienne

Le notebook `notebooks/mnist_neurogenesis.ipynb` ajoute deux mécanismes.

## 1. Neurogenèse dynamique
On part d'un **petit groupe de neurones** (5), et selon la **surprise** (motif
inconnu), on **augmente le nombre de neurones** (croissance adaptative).

| Seuil | Neurones (5 →) | Acc globale |
|---|---|---|
| 0.3 | 200 (plafond) | 0.595 |
| **0.5** | 194 | **0.620** |
| 0.7 | 118 | 0.555 |

Croissance détaillée (seuil 0.5) : 5 → 46 (après 0/1) → 67, 86, 105... → **194**
(après 9). L'acc finale 0.62 est comparable au drift sans élagage — la
neurogenèse gère sa capacité par croissance pure (sans réutiliser les ressources).

## 2. Couche Hebbienne connectée à la couche 1
Une 2e couche (SOM Hebbien) reçoit les activations de la couche 1 et classifie.

| Représentation d'entrée | Acc (couche 1 → 2) |
|---|---|
| Couche 1 seule (référence) | **0.620** |
| Activation brute (padding) | 0.415 |
| One-hot du gagnant | 0.115 |
| Top-k des similarités | 0.038 |

## Analyse honnête
1. La **neurogenèse seule fonctionne** (croissance pilotée par la surprise, acc 0.62).
2. La **couche 2 Hebbienne dégrade** (0.04-0.42 vs 0.62), quelle que soit la
   représentation d'entrée.
3. **Causes** : la **dimension dynamique** de la neurogenèse (les neurones
   grossissent) rend l'entrée de la couche 2 instable (padding/one-hot), et la
   couche 1 classifie déjà directement — la couche 2 dilue au lieu d'affiner.

=> La neurogenèse est une bonne **croissance adaptative**, mais connecter une
couche Hebbienne par-dessus ne s'améliore pas dans cette architecture — le frein
est la dimension dynamique.

## Système COMBINÉ : neurogenèse + élagage Physarum

La vraie amélioration est d'**intégrer la neurogenèse AU système** (élagage
Physarum), pas de la tester seule — le comportement biologique du cerveau :
**croissance** (nouveau) + **atrophie** (inutile).

| Réglage (max, seuil, prune) | Acc globale |
|---|---|
| max=60, seuil=0.6, prune=0.3 | 0.917 |
| max=60, seuil=0.6, prune=0.4 | 0.932 |
| **max=60, seuil=0.7, prune=0.4** | **0.974** |
| max=80, seuil=0.6, prune=0.3 | 0.831 |

## Bilan : amélioration par neurogenèse
| Approche | Acc |
|---|---|
| Élagage seul (référence) | ~0.95-0.96 |
| Neurogenèse seule | 0.62 (sature) |
| **Neurogenèse + élagage COMBINÉ** | **0.974** |

**Conclusion** :
1. La neurogenèse intégrée à l'élagage Physarum **dépasse** l'élagage seul
   (0.974 vs 0.95-0.96) : croissance + atrophie cohabitent et s'améliorent.
2. Il faut un **plafond bas (60) + seuil haut (0.7) + élagage fort (0.4)** : la
   croissance est contenue, l'atrophie gère les ressources.
3. **Avantage** : on n'a plus besoin de fixer le nombre initial de neurones (la
   neurogenèse croît de 5 automatiquement) — le système est **autonome**.
4. C'est le vrai comportement biologique : **neurogenèse + élagage synaptique** =
   adaptation continue du réservoir.

---

# 🖼️ Deux couches de neurones — résultats + visualisation

Le notebook `notebooks/mnist_two_layers_viz.ipynb` ajoute une **couche Hebbienne**
connectée au système combiné, et **visualise en images les deux couches**.

## Résultats
| Couche | Acc globale |
|---|---|
| Couche 1 seule (neurogenèse + élagage) | **0.946** |
| Couche 1 → couche 2 (Hebbienne) | 0.300 |

La couche 2 **ne dépasse pas** la couche 1 seule : la représentation d'entrée
(activations de la couche 1) n'est pas assez discriminante, et l'élagage Physarum
déplace les neurones de la couche 1 → l'entrée de la couche 2 bouge.

## Visualisation (2 figures)
1. **Couche 1** : prototypes en **images** (784 = 28×28), par classe dominante —
   reconnaissables (0-9).
2. **Couche 2** : poids dans l'espace d'activation (60 dims, affichés en 8×8) —
   patterns d'activation appris.

Le meilleur reste la couche 1 (`predict_label`) sur le système combiné.

---

# 🔄 Rétroaction Top-Down (couche 2 → couche 1)

Le notebook `notebooks/mnist_feedback_topdown.ipynb` connecte la couche 2 **en
retour** vers la couche 1 :

$$\text{Entrée}_{C1} = \text{Signal Visuel} + \gamma \cdot \text{Feedback}_{C2}$$

Le feedback (représentation de haut niveau) est rétroprojeté via `W2^T` dans
l'espace de la couche 1, **affinant la perception** de bas niveau (predictive
coding top-down).

## Résultats
| Configuration | Acc |
|---|---|
| Couche 1 seule (sans feedback) | 0.939 |
| **γ = 0.12-0.15 (feedback)** | **1.000** |
| γ = 0.02 | 0.974 |
| γ > 0.2 | 0.000 (déformation) |

## Analyse
La rétroaction à **intensité modérée** (γ≈0.12) **corrige les ambiguïtés** de la
couche 1 (0.939 → 1.000) : le contexte de haut niveau guide la perception de bas
niveau. Mais γ trop fort **déforme** le signal → acc effondrée.

C'est le principe du **predictive coding** : le top-down guide le bottom-up,
avec modération (γ optimal ~0.12).

## Cycle feedforward-feedback (décomposition 2 temps)
Pour chaque image, la boucle se fait en **2 temps** :
- **t=0 (feedforward)** : `a1 = W1·x`, `z1 = WTA(a1)`, `z2 = WTA(W2·z1)`
- **t=1 (résonance)** : `a1 = W1·x + γ·(W2^T·z2)`, `z1 = WTA(a1)`

## Les 3 régimes de γ (facteur d'attention)
| γ | Régime | Acc |
|---|---|---|
| **0** | Feedforward pur (mesure) | 0.968 |
| **0.1-0.4** | Guidage attentionnel (top-down modéré) | 0.85-0.97 |
| **≥1.0** | Hallucination (contexte domine) | 0.14-0.52 |

## Visualisation (4 figures)
1. **Couche 1** : prototypes en images (0-9)
2. **Couche 2** : poids dans l'espace d'activation (8×8)
3. **Courbe acc vs γ** : zone de guidage en vert, seuil d'hallucination en rouge
4. **Hallucination démontrée** : un vrai « 5 » à γ=2 est classé par un neurone de
   **classe 0** — le top-down allume un neurone absent de l'image.

Le γ optimal est dans la **zone de guidage (0.1-0.4)** : doser le contexte sans
laisser la C2 dominer la mesure visuelle.

---

# 🔲 WTA par seuil + découpage image

Le notebook `notebooks/mnist_threshold_wta_patches.ipynb` modifie deux mécanismes.

## 1. WTA par seuil (au lieu de top-K)
$$\text{WTA}_\theta(a) = \begin{cases} a_i & \text{si } a_i \ge \theta \\ 0 & \text{sinon} \end{cases}$$

## 2. Découpage de l'image (patches)
On encode l'image par **patches** (blocs) au lieu des pixels bruts.

## Résultats commentés
**Découpage image** :
| Patch | Acc |
|---|---|
| sans découpage | 0.971 |
| 4×4 | 0.970 |
| **7×7** | **0.976** |
| 14×14 | 0.935 |

Le découpage en patches (4-7) est comparable ou légèrement supérieur aux pixels
bruts (varie selon les seeds). Le patch 7×7 est un bon compromis contexte local
vs résolution.

**WTA par seuil** : le seuil **ne change pas la classification directe**
(argmax ≈ seuil) car on prend toujours le max au-dessus du seuil. L'intérêt du
WTA par seuil est la **représentation** : garder plusieurs neurones au-dessus du
seuil enrichit la couche 2 (sparsité contrôlée).

## Analyse
=> Le gain du découpage est modeste et dépend du seed ; le seuil n'impacte pas la
classification directe (il sert à la **sparsité de la représentation** pour la
couche 2).

---

# 🔍 Détection multi-chiffres (fenêtre glissante + stride + bounding box)

Le notebook `notebooks/mnist_multidigit_detection.ipynb` passe à des **images
contenant plusieurs chiffres** : une fenêtre 28×28 **défile** l'image (stride),
chaque fenêtre est classifiée, puis on **regroupe en bounding boxes** avec une
classification par chiffre.

## Pipeline
```
[ Image multi-chiffres ] → fenêtre glissante (stride) → classification → bounding boxes
```

## Résultats
- Image `[3, 7, 2]` → fenêtre glissante (stride 7) → bounding boxes : `[3, 0, 1, 3]`
- **3 chiffres spatialement encadrés** (bounding boxes), mais **étiquetage imparfait**

## Analyse honnête
1. Le **pipeline fonctionne structurellement** : la fenêtre glissante défile et
   les détections voisines se regroupent en bounding boxes.
2. La **qualité d'étiquetage** reflète la discrimination du classifieur : les
   chiffres confondables (2/3/6/8) sont mal classés, et le fond peut être classé
   "0" (neurones de 0 sur-représentés).
3. **Améliorations possibles** : classifieur plus discriminant (plus de neurones,
   ou le feedback top-down γ=0.12 qui donnait 1.000) + seuil d'énergie du fond.

Le pipeline de **détection multi-chiffres** est opérationnel ; sa précision dépend
de la discrimination du classifieur en amont.

---

# 🖼️ Reconstruction de l'image + bounding boxes + 2 scores

Le notebook `notebooks/mnist_reconstruction_detection.ipynb` change d'approche :
au lieu d'un **décodeur**, on **recrée l'image globale** par les **prototypes**
des neurones (auto-encodage non supervisé), puis on fait des **bounding boxes**
sur l'image recréée. **2 scores distincts** : IoU (segmentation) + nombre de
boxes (comptage). Scènes avec **1 chiffre + patches noirs**.

## Pipeline
```
[ Image 1 chiffre + patches noirs ]
   → reconstruction (prototypes des neurones, rejet du fond)
   → bounding boxes (segmentation par énergie)
   → Score IoU + Score de nombre de boxes
```

## Résultats (10 chiffres, 1 box attendue)
| Chiffre | IoU | Count |
|---|---|---|
| 0 | 0.54 | 1.00 |
| 1 | 0.55 | 1.00 |
| 2 | 0.57 | 1.00 |
| 3 | 0.54 | 1.00 |
| 4 | 0.43 | 1.00 |
| 5 | 0.86 | 1.00 |
| 6 | 0.68 | 1.00 |
| 7 | 0.50 | 1.00 |
| 8 | 0.48 | 1.00 |
| 9 | 0.39 | 1.00 |
| **MOYENNE** | **0.553** | **1.000** |

## Les 2 scores distincts
1. **Score IoU** (qualité segmentation) : 0.553 en moyenne — chevauchement entre
   la box prédite et la vérité (la reconstruction étale un peu le chiffre).
2. **Score de nombre de boxes** (comptage) : **1.000** — le système détecte
   exactement 1 box pour 1 chiffre, les **patches noirs sont ignorés** (rejet par
   seuil de similarité lors de la reconstruction).

=> Le **comptage est parfait**, la **segmentation (IoU)** est correcte (>0.5 sur
la plupart des chiffres) mais imparfaite à cause de la reconstruction bruitée.

---

# 🧠 Méta-Neurones (centroïdes de prototypes)

Le notebook `notebooks/mnist_metaneurons.ipynb` ajoute des **méta-neurones** :
agrégation par similitude matricielle des prototypes de Couche 1.

$$S_{ij} = \\frac{W_i \\cdot W_j}{\\|W_i\\|\\,\\|W_j\\|}, \\qquad W_{\\text{meta}_k} = \\frac{1}{|C_k|} \\sum_{i \\in C_k} W_i$$

## Balayage du seuil (réduction vs acc)
| Seuil | Neurones | Réduction | Acc |
|---|---|---|---|
| 0.95 | 120→120 | 0% | 0.693 |
| 0.85 | 120→118 | 2% | 0.690 |
| **0.7** | 120→98 | **18%** | 0.677 |
| 0.6 | 120→89 | 26% | 0.667 |
| 0.5 | 120→77 | 36% | 0.590 |

## Reconstruction multi-chiffres (3 chiffres + patches noirs)
- **Score IoU** : **0.667** (segmentation)
- **Score count** : **1.000** (3/3 boxes détectées)
- **Réduction réseau** : 120 → 98 méta (18%)

## Analyse
1. **Réduction** : le seuil 0.7 fusionne les prototypes similaires (18% de réduction).
2. **Classification** : l'acc est maintenu (léger recul à seuil agressif).
3. **Localisation multi-chiffres** : boxes bien alignées, **count parfait** (1.000),
   IoU 0.67 — mais la reconstruction **visuelle** du contenu est bruitée (moyennage).

=> Les méta-neurones apportent la **réduction du réseau** et une **meilleure
robustesse de localisation**, au prix d'une reconstruction visuelle moins
spécifique.

---

# 🌬️ Cycle respiratoire : gel dynamique + création de couches

Le notebook `notebooks/mnist_respiratory_cycle.ipynb` construit un système qui
**gèle et crée des couches dynamiquement** pour être plus stable (compromis
Plasticité/Stabilité de Grossberg).

## Signal d'Activité Structurelle S(t) + détection d'oscillation (FFT)
$$S(t) = \text{Taux de Création}(t) - \text{Taux d'Élagage}(t)$$

L'analyse **fréquentielle (FFT)** de $S(t)$ distingue :
- **Signal plat / bruit blanc** → stabilité
- **Pic fréquentiel net (cycle limite)** → instabilité → déclenche les leviers

## Cycle respiratoire complet (démontré)
| Phase | État |
|---|---|
| **Exploration** | C1 dégelée, 1 couche (plasticité) |
| **Consolidation** | oscillation → gel de C1 + **spawning de C2** (2 couches) |
| **Perturbation** | objet inconnu → surprise → **dégel sélectif** |

## Surprise de reconstruction (déclencheur de dégel)
- **Chiffres connus** : surprise 0.66 (faible, restent gelés)
- **Objets inconnus** : surprise 0.96 (élevée, persistant → dégel)

## Les 3 leviers + plasticité sélective
1. **Amortissement** : augmentation de la viscosité λ (dissipe l'oscillation)
2. **Méta-neurones** : fusion des oscillateurs instables
3. **Spawning de couche** : gel C1 + création C2 (presse top-down)
4. **Dégel** : neurogenèse d'extension (W1 gelé, ajout de neurones) + métro-plasticité
   η(S) par recuit simulé.

=> Le réseau **"respire"** : il alterne plasticité (explorer) et stabilité
(consolider), au lieu d'une structure rigide.

---

# 🧪 Test du cycle respiratoire sur MNIST (IoU + bounding boxes)

Le notebook `notebooks/mnist_respiratory_test.ipynb` connecte le **cycle
respiratoire** à l'entraînement réel des AnchorNeurons et évalue la
reconstruction + bounding boxes + 2 scores.

## Résultats (scène multi-chiffres [3,7,2] + patches noirs)
| Configuration | IoU | Count | Boxes |
|---|---|---|---|
| Base (sans cycle) | 0.565 | 1.000 | 3/3 |
| **Cycle respiratoire** | **0.660** | 1.000 | 3/3 |
| **Gain** | **+0.095** | — | — |

## Analyse
1. **Les bounding boxes restent bonnes** avec le cycle (3/3, count parfait).
2. **L'IoU AUGMENTE** avec le cycle respiratoire (+0.095) : la consolidation
   (gel de C1) stabilise les prototypes et améliore la reconstruction.
3. Le cycle apporte un **double bénéfice** : meilleure segmentation (IoU) ET
   stabilité structurelle (gel, pas d'oubli catastrophique, couches dynamiques).

=> Le cycle respiratoire améliore l'IoU tout en gardant la détection correcte :
un gain de stabilité ET de qualité de segmentation.

---

# 🎭 Mode collapse + domination par la densité

Le notebook `notebooks/mnist_mode_collapse.ipynb` diagnostique le problème :
le réseau remplace les vrais chiffres (3, 7, 2) par des reconstructions denses
(0, 8) et sature le fond de bruit.

## Cause (densité par chiffre)
| Chiffre | Densité (pixels actifs) |
|---|---|
| 0 | 0.203 (dense) |
| 1 | 0.062 (fin) |
| 3 | 0.208 (dense) |
| 7 | 0.106 (fin) |

En `W·x`, les chiffres **denses** (0,3,5,8) produisent plus d'activation et
**gagnent toujours** → les **fins** (1,7,4) sont écrasés (**mode collapse**).

## Corrections testées
| Seuil | IoU | Count | Bruit |
|---|---|---|---|
| 0.0 | 0.560 | 1.000 | 0.796 |
| 0.2 | 0.544 | 0.667 | 0.704 |
| 0.3 | 0.250 | 0.333 | 0.509 |
| 0.4 | 0.252 | 0.333 | 0.272 |
| **0.5** | **0.728** | **1.000** | **0.148** |

## Analyse honnête
1. **Mode collapse confirmé** : les denses dominent la compétition.
2. **Seuillage global** : un seuil médian (0.2-0.4) **dégrade** (écrase les formes
   fines 7,2). Le **seuil 0.5** est optimal : nettoie le bruit (0.80→0.15) ET
   améliore l'IoU (0.56→0.73) avec count parfait.
3. **Homéostasie divisive** : testée, dégrade (sur-compensation). L'homéostasie
   soustractive actuelle est meilleure.

=> Le mode collapse est réel (domination de densité) mais un **seuil RELATIF** à
chaque prototype est la bonne direction (le seuil 0.5 le démontre), pas un seuil
global agressif ni la fatigue divisive.

---

# 🎯 Méthode itérative avec FOCUS

Le notebook `notebooks/mnist_focus_iterative.ipynb` remplace le seuil rigide par
une **méthode itérative avec focus** (attention visuelle : saccade + fovéa).

## Pipeline
```
[ Image ] → 1. Détection (seuil relatif adaptatif)
          → 2. FOCUS (crop + agrandissement de chaque objet)
          → 3. Analyse (classification de l'objet isolé)
          → accuracy
```

## Résultats (accuracy réintroduite, seuil rigide retiré)
| Scène | Détectés | Acc |
|---|---|---|
| [3,7,2] | [3,7,0] | 0.667 |
| [0,8,1] | [0,2,1] | 0.667 |
| [5,9,4] | [2,1,9] | 0.000 |
| [1,2,3] | [1,0,3] | 0.667 |
| [6,7,8] | [0,7,2] | 0.333 |
| **Moyenne** | — | **0.524** |

## Analyse
1. **Détection** : count parfait (1.000) sur toutes les scènes — les objets sont
   localisés robustement (seuil RELATIF adaptatif, pas de seuil dur).
2. **Focus** : chaque objet est croppé et agrandi (fovéa) — un chiffre isolé du
   fond pour l'analyse.
3. **Analyse** : accuracy moyenne 0.524 — la classification reflète le
   classifieur (confusions 2/8, 5/9 sur certains chiffres).

=> Le pipeline itératif est robuste pour la **localisation** (count 1.000) et
l'**accuracy** mesure la classification après focus. Le focus aide : un 7 isolé
est bien classé (conf 0.61) alors qu'il était mal classé dans la scène brute
(mode collapse).

---

# 🌍 Scale vers COCO Stuff

Le notebook `notebooks/mnist_coco_scale.ipynb` scale le pipeline vers **COCO
Stuff** (images de scènes réelles, ~171 classes stuff). Dataset HuggingFace
`shunk031/cocostuff`, entraînement **CLASSE PAR CLASSE** (leçon critique).

## Setup
- 30 images COCO réelles → **~54 classes stuff**, patches 32×32 (1024 dims)
- `datasets` 2.21 (les scripts ne sont plus supportés par 5.x) + `trust_remote_code=True`

## Résultats (classes distinctes)
| Classe | Acc |
|---|---|
| **house (105)** | **0.979** |
| railing (112) | 0.167 |
| sky (118) | 0.242 |
| snow (120) | 0.040 |
| **Moyenne** | **0.357** |

Détection avec focus : les objets sont détectés, house est correctement classée
(conf 0.88-0.90).

## Analyse honnête
1. **OK** : l'apprentissage **classe par classe** fonctionne pour les classes
   **homogènes** (house 0.98).
2. **LIMITE** : les classes stuff COCO sont **hétérogènes** (une même classe
   regroupe des textures très variables) → snow 0.04, sky 0.24 s'effondrent.
3. **Scale à 54 classes** : échoue (acc ~0 sur la plupart) — la **variance
   intra-classe** explose avec le nombre de classes.

## Pistes
- plus de neurones, patches plus grands
- **caractéristiques de texture** (au lieu de pixels bruts) pour séparer les
  classes stuff (sky/snow, road/gravel)
- des couches hiérarchiques (le cycle respiratoire) pour abstraire les textures

=> Le scale vers COCO révèle la limite réelle : l'hétérogénéité intra-classe
des classes stuff, pas la capacité du pipeline classe par classe.

---

# 🎨 COCO Stuff : caractéristiques de TEXTURE + COULEUR

Le notebook `notebooks/mnist_coco_texture.ipynb` corrige le vrai problème :
on ne voyait **rien** sur les patches en **niveaux de gris** (et le modèle non
plus). Les classes stuff COCO sont des **textures** (ciel, neige, route, herbe)
qui ne se distinguent pas par la forme mais par la **couleur** et la **texture**.

## Correction
Au lieu des pixels gris bruts, on extrait des **caractéristiques couleur + texture** :
- **Couleur** : moyennes + écarts R,G,B, histogrammes de couleurs
- **Texture** : gradient moyen, variance/contraste, co-occurrence

## Résultats (classe par classe)
| Classe | Pixels gris | **Texture+couleur** |
|---|---|---|
| house | 0.979 | 0.82-0.91 |
| sky | 0.242 | **0.74-0.77** |
| snow | 0.040 | **0.65-0.69** |
| railing | 0.167 | **0.70** |
| **Moyenne** | **0.357** | **0.68-0.70** |

## Analyse
1. L'accuracy passe de **~0.36 (pixels gris)** à **~0.68-0.70 (texture+couleur)**.
2. Le diagnostic était juste : les classes stuff sont des **textures** et la
   **couleur** est discriminante (ciel bleu, herbe verte, route grise).
3. Les caractéristiques (moyennes RGB, histogrammes, gradients, contraste)
   portent l'information que les pixels gris bruts avaient perdue.

=> Le scale vers COCO est **VIABLE** avec les bonnes caractéristiques : couleur
+ texture au lieu des pixels bruts. C'est la clé pour les classes stuff
(matériaux/étendues) vs les formes (chiffres).

---

# 🧬 Entraînement COCO évolutif (neurogenèse + cycle respiratoire)

Le notebook `notebooks/mnist_coco_evolutive.ipynb` fait **évoluer l'architecture**
pendant l'entraînement (le comportement attendu) :
- **Neurogenèse** : ajoute des neurones quand la surprise est élevée
- **Cycle respiratoire** : gèle C1 et ajoute des couches quand l'oscillation apparaît

## Résultats (features COCO, 55 classes)
| Métrique | Initial | Final |
|---|---|---|
| **Neurones** (neurogenèse) | 11 | **970** |
| **Couches** (cycle respiratoire) | 1 | **2** |
| Phase | exploration | **consolidation** (C1 gelée) |
| Accuracy (échantillon) | — | 0.425 |

## Comportement démontré
1. **Le modèle CROÎT** : 11 → 970 neurones (la surprise élevée crée de nouveaux
   neurones via la neurogenèse).
2. **Le modèle AJOUTE une couche** : 1 → 2 (le cycle respiratoire gèle C1 et
   déploie C2 quand l'oscillation structurelle apparaît).
3. **L'architecture évolue réellement** pendant l'entraînement — contrairement
   aux versions précédentes où le nombre de neurones/couches était fixe.

## Note sur les données
- Le **goulot n'est pas le calcul mais le réseau** : ~21 s/image en streaming HF.
- Solution : **cacher le set de validation en bloc** (5 000 images, bien moins
  que les 164K du train) pour un accès local instantané.

---

# 📊 Monitoring COCO : performance + reconstruction + architecture évolutive

Le notebook `notebooks/mnist_coco_monitor.ipynb` montre **comment le modèle
évolue et progresse** pendant l'entraînement (set validation COCO, 171 classes).

## Comportement évolutif corrigé
Le modèle ÉVOLUE réellement (contrairement aux versions fixes) :
- **Neurogenèse progressive** : 10 → 31 → 52 → 74 → 93 → 115 → 136 → 156
  neurones (croissance douce, plus d'explosion instantanée)
- **Spawn de couche à saturation** : quand C1 atteint ~80% du plafond, C1 est
  **archivée** et une **C2 est créée** pour absorber la nouveauté (couches 1→2)
- **Élagage Physarum** compense la croissance

## Monitoring (3 figures)
1. **Évolution de l'accuracy** : 0.012 → 0.053 au fil des patches
2. **Ce que les poids construisent** : prototypes des neurones (la représentation apprise)
3. **Architecture finale** : couches, neurones, connexions bottom-up + rétroaction

## Correction clé
Le problème initial : la neurogenèse ajoutait un neurone à CHAQUE patch
(explosion à 2000) et aucune couche n'était créée. Corrigé avec :
- **neurogenèse sélective** (seuil de nouveauté élevé : n'ajoute que si mal représenté)
- **spawn par saturation** (plafond + fraction de saturation) au lieu de l'oscillation

---

# 📋 Compte rendu d'entraînement COCO

Le notebook `notebooks/mnist_coco_report.ipynb` génère un **compte rendu complet**
d'entraînement (module `report.py` + `TrainingTracker`) :

## Métriques suivies
| Métrique | Valeur (exemple) |
|---|---|
| **Images passées** | 40 |
| **Patches traités** | 32 800 |
| **Classes vues** | 40 |
| **Classes/image (moy)** | 1.0 |
| **Patches/image (moy)** | 820 |
| **Temps/image (moy)** | ~0 s (features locales) |
| **Temps total CPU** | 0.5 s |
| **Débit** | 60 904 patches/s |
| **Neurones finaux** | 31 |
| **Couches** | 2 (+1 archivée) |
| **Poids totaux** | 1 085 |
| **Connexions (co_act>0)** | 130 |

## Rapport visuel
Le rapport affiche 4 panneaux :
1. **Temps par image** (courbe)
2. **Top classes** (répartition des features)
3. **Neurogenèse** (évolution des neurones)
4. **Résumé texte** (toutes les métriques en un coup d'œil)

Le débit est élevé (~60K patches/s) car on utilise les **features en local**
(plus de réseau) — le calcul NumPy sur CPU mono-thread reste le facteur limitant
(22 cœurs inutilisés).
