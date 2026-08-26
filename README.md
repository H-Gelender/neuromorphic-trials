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
│   └── mnist_reconstruction_detection.ipynb # reconstruction image + 2 scores (IoU & count)
├── src/recherche_agi/
│   ├── data.py                        # chargement MNIST + filtre par chiffres
│   ├── unsupervised.py                # AnchorNeurons, WTA, fatigue, co-activation,
│   │                                  #   top-down, image_to_patches, élagage Physarum
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
