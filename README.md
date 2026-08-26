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
│   └── mnist_two_layers_viz.ipynb      # deux couches + visualisation en images
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
