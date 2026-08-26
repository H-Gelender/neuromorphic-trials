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
│   └── mnist_global_accuracy.ipynb     # acc globale après drift (test 0-9)
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
