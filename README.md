# Module Adaptatif — Recherche AGI (MNIST 0-9)

Recherche sur un **module adaptatif** type MoE (Mixture of Experts) appliqué à la
classification de chiffres manuscrits MNIST (0-9). Le notebook
[`notebooks/mnist_adaptive_results.ipynb`](notebooks/mnist_adaptive_results.ipynb)
reproduit l'intégralité du pipeline. Ce README explique la méthode, les résultats
et les conclusions.

---

## 📌 Objectif

Construire un **système de réseau de neurones autonome** qui :
1. Apprend les chiffres **progressivement** (un expert par chiffre, les anciens
   sont gelés).
2. **Découvre tout seul** les couches redondantes entre experts (bridges).
3. **Compresse** le modèle en fusionnant les couches partagées, en validant que la
   perte de précision reste sous un seuil acceptable.
4. Se compare honnêtement à un réseau dense classique entraîné sur toutes les
   classes d'un coup.

---

## 🧱 Architecture

### MoE (Mixture of Experts)
- **Routeur** au début : `Linear(784 → n_experts)` + softmax, choisit l'expert.
- **10 experts** indépendants, un par chiffre. Chaque expert = MLP `784 → 16 → 1`.
- Le routeur peut être **forcé** vers un expert (`forced_expert`).
- Sortie = somme pondérée des experts, chaque expert contribuant sur ses classes.

### Couches partagées (compression)
`SharedMoE` : les experts peuvent **partager une même couche fc1** (les doublons
de features disparaissent). `n_params()` compte les paramètres **uniques**
(une couche partagée n'est comptée qu'une fois, même si plusieurs experts
l'utilisent).

### MLP dense (référence)
Un réseau classique `784 → hidden → 10` entraîné sur les 10 chiffres d'un coup.

---

## 🔬 Méthode

1. **Entraînement progressif** : pour chaque chiffre d de 0 à 9, on ajoute son
   expert, on gèle les experts précédents, on entraîne le routeur + le nouvel
   expert sur `[0..d]`. Enfin, on entraîne le routeur sur toutes les classes.

2. **Détection des bridges** : on mesure la **similarité de la couche fc1** entre
   chaque paire d'experts, via le **cosinus des activations** sur un lot commun.
   Deux experts ont un *bridge* si beaucoup de leurs neurones fc1 produisent des
   activations quasi identiques → features redondantes, fusionnables.

3. **Compression autonome** : pour chaque paire découverte (≥ 4 neurones
   fusionnables) :
   - fusion de la couche (l'expert B réutilise la couche de A),
   - **réentraînement court** de la paire fusionnée sur ses propres données,
   - validation : si `Δacc ≥ -seuil (0.02)`, on garde ; sinon **rollback**.

4. **Comparaison** : MoE compressé vs MLP denses de plusieurs tailles
   (courbe params vs précision).

---

## 📊 Résultats

### 1. Entraînement progressif
| Modèle | Params | Acc (0-9) |
|---|---|---|
| MoE 10 experts (non compressé) | 133 620 | 0.8421 |

### 2. Bridges (similarité de couche fc1)
| Paire | Similarité | Interprétation |
|---|---|---|
| **4-9** | 0.971 | très redondantes → fusionnables |
| 8-9 | 0.941 | fortement fusionnables |
| 1-7 | 0.900 | modéré |
| **8-0** | 0.847 | plus distinctes → peu de fusion |

### 3. Compression autonome
| | Avant | Après |
|---|---|---|
| Params | 133 620 | 70 820 |
| Réduction | — | **-47.0 %** |
| Couches partagées | 10 | 5 |
| Acc | 0.8391 | 0.8230 |
| Fusions conservées | — | **11 / 30** |

Le système a découvert 30 paires fusionnables, testé les fusions, et conservé
11 d'entre elles (perte < 0.02). 19 ont été rejetées (perte trop grande → rollback).

### 4. Comparaison MoE vs MLP dense
| Modèle | Params | Test acc | Efficacité (acc/10k params) |
|---|---|---|---|
| **MoE compressé** | 70 820 | 0.8230 | 0.116 |
| MLP h=16 | 12 730 | 0.9502 | **0.746** |
| MLP h=32 | 25 450 | 0.9681 | 0.380 |
| MLP h=64 | 50 890 | 0.9740 | 0.191 |
| MLP h=89 | 70 765 | 0.9775 | 0.138 |
| MLP h=128 | 101 770 | 0.9779 | 0.096 |

---

## 🎯 Conclusion

**Le MLP dense entraîné sur les 10 chiffres d'un coup bat nettement le MoE
compressé :**

- À **paramètres égaux** (~70 800) : MLP **0.9775** vs MoE **0.8230** → +15 points.
- Un **MLP minuscule** (h=16, 12 730 params, 5.5× plus petit) atteint **0.9502**,
  soit 6.4× plus efficace (acc/param) que le MoE compressé.

### Pourquoi le MoE perd-il ?
1. **La compression est agressive** : -47% de params mais le réentraînement court
   post-fusion ne récupère pas toute la performance.
2. **L'entraînement progressif** (un chiffre à la fois) est sous-optimal vs
   l'entraînement **conjoint** sur les 10 chiffres (le MLP voit toutes les classes
   simultanément → meilleure discrimination).
3. **Les experts sont trop petits/partagés** pour que la spécialisation compense.

### Quand le MoE a-t-il du sens ?
Le MoE n'est **pas** meilleur pour la précision brute. Il a de l'intérêt pour
**l'extensibilité adaptative** :
- ajouter une nouvelle classe/expertise **sans réentraîner tout** le modèle,
- **gel** des experts existants (apprentissage sans oubli catastrophique),
- **compression** par partage de features redondantes.

Si l'objectif est la précision pure, un MLP dense est supérieur. Si l'objectif
est un **système adaptatif incrémental** (ajouter des capacités en continu), le
MoE autonome avec découverte/fusion/validation est la bonne direction — mais il
faudrait un **entraînement conjoint** ou des **experts plus gros** pour qu'il soit
compétitif.

---

## 📂 Structure du projet

```
recherche-agi/
├── notebooks/
│   ├── mnist_adaptive_results.ipynb   # notebook de synthèse MoE (pipeline complet)
│   ├── mnist_physarum.ipynb           # architecture Physarum (flux sans neurones)
│   ├── mnist_predictive_physarum.ipynb # Blob + Predictive Coding (système hybride)
│   ├── mnist_superpixels_graph.ipynb  # image -> graphe par superpixels + flux
│   ├── mnist_synaptic_physarum.ipynb  # arêtes actives (synapses) + pooling
│   ├── mnist_synaptic_predictive.ipynb # réservoir synaptique + Predictive Coding
│   ├── mnist_enhanced_reservoir.ipynb  # 3 leviers de discrimination
│   ├── mnist_neuromodulated.ipynb      # boucle neuromodulée (plasticité par surprise)
│   ├── mnist_neuromodulated_loop.ipynb # boucle neuromodulée COMPLÈTE branchée
│   ├── mnist_local_ssm.ipynb           # SSM local (intégration temporelle sous chaque nœud)
│   ├── mnist_sensory_bundle.ipynb      # fourre-tout sensoriel (image, texte, signal)
│   ├── mnist_multimodal_context.ipynb  # couplage vision + langage (image + label texte)
│   ├── mnist_fewshot_pretrain.ipynb    # pré-entraînement few-shot + gel anti-oubli
│   ├── mnist_unsupervised.ipynb        # classification 100% non supervisée (neurones d'ancrage)
│   ├── mnist_topdown_recall.ipynb      # rappel top-down (régénération + résonance)
│   ├── mnist_self_eval_loop.ipynb      # boucle d'auto-évaluation de la surprise (S_auto)
│   ├── mnist_visualize_accuracy.ipynb  # visualisation (accuracy + convergence S_auto)
│   ├── mnist_deep_hebbian.ipynb        # Deep Hebbian hiérarchique (L1 bords → L2 formes)
│   ├── mnist_drift_test.ipynb          # test du modèle drift (anti-oubli catastrophique)
│   └── mnist_physarum_pruning.ipynb    # élagage Physarum des AnchorNeurons sous drift
├── src/recherche_agi/
│   ├── data.py                        # chargement MNIST + filtre par chiffres
│   ├── deep_hebbian.py                # encodeur hiérarchique (L1→L2, anti-Hebbian, soft-WTA)
│   ├── enhanced_reservoir.py          # réservoir amélioré (rétine, multi-axe, compétition)
│   ├── image_graph.py                 # image -> graphe superpixels (Felzenszwalb)
│   ├── local_ssm.py                   # SSM local (h_t = (1-Δ)h + Δ·x)
│   ├── neuromodulated.py              # boucle neuromodulée (plasticité gérée par surprise)
│   ├── sensory_bundle.py              # fourre-tout sensoriel (encodeur multi-modalités)
│   ├── unsupervised.py                # classification non supervisée (SOM + WTA dynamique + fatigue)
│   ├── physarum.py                    # solveur Physarum (Tero 2007)
│   ├── predictive_physarum.py         # hybride Blob + Predictive Coding
│   ├── synaptic_physarum.py           # Physarum synaptique (tanh + Hebbien + pooling)
│   ├── training.py                    # entraînement, early stopping, routeur, gel d'experts
│   └── services/
│       ├── callbacks.py               # callbacks d'entraînement
│       ├── checkpoint.py              # sauvegarde/chargement de modèles
│       ├── llm.py                     # service LLM local (pydantic-ai)
│       └── network_analysis.py        # analyse de neurones, similarité, pruning
├── data/                              # MNIST (téléchargé, gitignoré)
├── pyproject.toml
└── README.md
```

## 🚀 Exécution

```bash
cd C:/Users/henry/Desktop/workspace/recherche-agi
.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/mnist_adaptive_results.ipynb   # exécute tout
.venv/Scripts/jupyter-lab notebooks/mnist_adaptive_results.ipynb  # pour explorer
```

Le notebook de synthèse reproduit tout le pipeline de bout en bout.

---

# 🦠 Physarum — Intelligence sans neurones

Le notebook `notebooks/mnist_physarum.ipynb` teste une **architecture Physarum** :
le changement de paradigme est radical — **on ne calcule pas des matrices de
poids, on modélise des dynamiques de flux.**

## Références scientifiques
- **Tero, Kobayashi, Nakagaki (2007)** — *A mathematical model for adaptive
  transport network in path finding by true slime mold*, J. Theoretical Biology.
  Modèle de référence : loi de Poiseuille + conservation de Kirchhoff + adaptation
  des tubes.
- **Solé & Pla-Mauri (2025)** — *Cognition as least action: the Physarum
  Lagrangian*, arXiv:2511.08531. Formulation lagrangienne : les états stationnaires
  minimisent la dissipation d'énergie.

## Modèle mathématique
| Équation | Sens |
|---|---|
| $Q_{ij} = \frac{D_{ij}}{L_{ij}}(p_i - p_j)$ | **Poiseuille** : flux ∝ conductance × gradient de pression |
| $\sum_j Q_{ij} = S_i$ | **Kirchhoff** : conservation de la masse |
| $\frac{dD_{ij}}{dt} = |Q_{ij}|^\mu - \delta D_{ij}$ | **Adaptation** : tubes utilisés élargis, autres dépéris |
| $P = \sum \frac{L_{ij}}{D_{ij}} Q_{ij}^2$ | **Dissipation** (Lagrangien) : flux optimal = moindre action |

## Idée de classification
Les pixels de l'image → **injections de pression** $S_i$ sur un graphe. Le flux
circule, les tubes s'adaptent selon la forme du chiffre. On extrait une
**signature de flux** (réservoir) et on classifie par similarité avec des
prototypes par classe.

## Résultat
| Architecture | Params | Acc (0-9) |
|---|---|---|
| **Physarum (réservoir)** | **0** | **~0.36** |
| MLP dense h=16 | 12 730 | 0.95 |
| MoE compressé | 70 820 | 0.82 |

**Lecture honnête** : le Physarum sans poids **capture la forme spatiale** (le flux
suit le tracé du chiffre) mais a une **capacité discriminative limitée** sur 10
classes (~0.36). C'est un **réservoir**, pas un classifieur : il faudrait un
décodage plus puissant (couche de lecture entraînée sur les signatures) pour
rivaliser avec un réseau entraîné. L'intérêt du Physarum est ailleurs : calcul
**sans poids entraînés**, robustesse morphologique, et parallélisme naturel.

---

# 🧠 Blob + Predictive Coding — système hybride

Le notebook `notebooks/mnist_predictive_physarum.ipynb` implémente le **système
hybride** que tu as décrit : le blob et le predictive coding partagent le travail.

```
[ Nouvelle Donnée ] ──> 1. Predictive Coding ──> Erreur de Prédiction (Nouveauté)
                                                        │
                                                        ▼
                       2. Mécanique du Blob ──> Crée/Consolide un tuyau
                                                        │
                                                        ▼
                       3. Consolidation    ──> Durcit le tuyau (Mémoire physique)
```

## Pipeline
1. **Predictive Coding** (Rao & Ballard 1999) : un modèle prédit la représentation
   attendue ; l'**erreur de prédiction** mesure la nouveauté.
2. **Blob** (Tero et al. 2007) : si l'erreur dépasse un seuil, le blob **crée un
   nouveau tuyau** (le graphe s'étend) ; sinon il **consolide** le tuyau le plus
   proche.
3. **Consolidation** : chaque tuyau a une **conductance** qui augmente à chaque
   mise à jour → le tuyau **durcit** (mémoire physique stable, apprentissage
   incrémental sans oubli).

## Point technique clé
Le **flux Physarum brut ne discrimine pas** les chiffres (intra ≈ inter ≈ 0.97).
On ajoute donc une **couche de lecture entraînée** (softmax linéaire) qui projette
les signatures de flux dans un espace discriminé. Le predictive coding opère dans
cet espace projeté.

## Résultats mesurés
- Couche lue seule sur flux : **0.46** (vs 0.10 au hasard)
- **Détection de nouveauté** : certaines classes (5, 0, 6) déclenchent la création
  de tuyaux (erreur > seuil)
- **Consolidation** : la conductance d'un tuyau revu passe de 1.0 à 2.85
  (le tuyau durcit, la mémoire se consolide)
- Classification hybride : ~0.19-0.23

## Lecture honnête
Le pipeline **démontre le mécanisme** (détection de nouveauté + création de tuyaux
+ consolidation/durcissement) mais la **précision reste limitée** car le réservoir
Physarum discrimine mal les chiffres. L'intérêt du système hybride est le
**mécanisme d'apprentissage incrémental sans réentraîner tout le modèle**, pas la
précision brute. Pour améliorer : une représentation plus discriminante du réservoir,
ou un décodage plus puissant sur les signatures.

---

# 🕸️ Image → Graphe par superpixels

Le notebook `notebooks/mnist_superpixels_graph.ipynb` convertit l'image MNIST en
**graphe** via l'algorithme **Felzenszwalb** (`skimage.segmentation.felzenszwalb`)
qui s'adapte naturellement aux formes (au lieu d'une grille régulière).

## Principe
- **Chaque superpixel = un nœud** du graphe
- **Superpixels adjacents = arêtes**
- L'intensité de chaque superpixel devient l'**injection de pression** pour le Physarum
- Les arêtes sont pondérées par la distance entre centroïdes

Ce graphe est ensuite **connecté au solveur Physarum** (`physarum_from_image`) :
le flux circule le long des superpixels, suivant la morphologie du chiffre.

## Résultat (honnête)
| Représentation | Acc couche lue |
|---|---|
| Grille régulière | **0.450** |
| Superpixels | 0.245 |

**Lecture** : les superpixels améliorent la **visualisation** et la **fidélité
morphologique** (le graphe et le flux épousent la forme du chiffre), mais pour la
**classification**, la **grille régulière reste meilleure** car elle a une
dimension fixe et un ordre stable (reproductible d'une image à l'autre), tandis
que les superpixels varient en nombre/ordre (le padding introduit du bruit).

=> Les superpixels sont le bon choix pour la **représentation** et l'**injection**
   (flux qui suit la forme) ; la grille régulière reste le bon choix pour la
   **couche lue de classification** (reproductibilité).

---

# 🧬 Physarum synaptique — arêtes actives (synapses)

Le notebook `notebooks/mnist_synaptic_physarum.ipynb` transforme les **arêtes
passives** (loi d'Ohm) en **synapses actives** qui font du traitement local.

## Les 3 étapes
1. **Flux non-linéaire** : `Q_ij = D_ij · tanh(α·(p_i - p_j))`
   — le seuil synaptique étouffe le bruit et amplifie les vrais traits.
2. **Plasticité Hebbienne locale** (STDP-like) : `D_ij += η·(σ(p_i)·σ(p_j)) - γ·D_ij`
   — la co-activation des nœuds renforce le tuyau (extraction de corrélations).
3. **Intégration dendritique** (pooling) : `z_k = Σ_{(i,j)∈Zone_k} ReLU(Q_ij)`
   — vecteur compact `z` par sous-zones spatiales, envoyé au Predictive Coding.

```
[ Graphe 14x14 ] ──> 1. Flux Non Linéaire ──> Q_ij = D_ij * tanh(alpha * Δp)
                          │
                          ▼
                   2. Plasticité Hebbienne ──> D_ij += η·(σ(p_i)·σ(p_j)) - γ·D_ij
                          │
                          ▼
                   3. Intégration Dendritique ──> vecteur z (somme ReLU par zone)
                          │
                          ▼
              [ Predictive Coding + Couche Lue ]
```

## Résultats mesurés
- **Flux tanh** : max passe de 0.14 (linéaire) à 0.44 (synaptique) — amplification.
- **Plasticité Hebbienne** : les conductances divergent (max 0.55 → 0.72) —
  le réseau se spécialise localement.
- **Vecteur z** (64 dims, compact) → **couche lue acc 0.393**.

## Comparaison des représentations
| Représentation | Signature | Acc couche lue |
|---|---|---|
| Grille régulière (flux linéaire) | 24 dims | 0.45 |
| Superpixels | 50 dims | 0.245 |
| **Physarum synaptique** | **64 dims** | **~0.40** |

Le réservoir synaptique est **comparable à la grille brute** mais avec des arêtes
**actives** qui font du traitement local (seuil, corrélations, pooling dendritique),
produisant une signature compacte prête pour le Predictive Coding.

---

# 🔗 Réservoir synaptique + Predictive Coding

Le notebook `notebooks/mnist_synaptic_predictive.ipynb` **connecte le réservoir
synaptique au Predictive Coding**.

## Pipeline complet
```
[ Image MNIST ] → 1. Flux Non Linéaire (tanh) → 2. Plasticité Hebbienne
              → 3. Pooling Dendritique (vecteur z) → 4. Predictive Coding
              → 5. Blob (crée/consolide des tuyaux)
```

Le réservoir synaptique produit un vecteur **z compact** (64 zones) qui alimente
le PC et la couche lue.

## Seuil de nouveauté adaptatif (régulation homéostatique)

Le seuil de nouveauté peut être **adaptatif** avec **régulation homéostatique** :
- trop de nouveautés (> cible) → le seuil **monte** (plus strict)
- pas assez → le seuil **descend** (plus permissif)

C'est un contrôleur homéostatique qui maintient un **taux de nouveauté** cible.

| Config | Acc | Tuyaux | Seuil final |
|---|---|---|---|
| Seuil fixe (0.5) | 0.187 | 4 | — |
| Adaptatif (cible 0.10) | 0.160 | 3 | 0.86 |
| Adaptatif (cible 0.15) | 0.160 | 3 | 0.75 |
| Adaptatif (cible 0.20) | 0.160 | 3 | 0.60 |
| Adaptatif (cible 0.30) | 0.193 | 4 | 0.38 |

**Impact honnête** : l'homéostasie **régule automatiquement le seuil** (pas de
calibrage manuel, le nombre de tuyaux reste contrôlé), mais l'impact sur la
classification est **marginal** (0.187 → 0.193 au mieux) — la précision est
limitée par la **discrimination du réservoir**, pas par le seuil.

---

# 🚀 Réservoir amélioré — 3 leviers de discrimination

Le notebook `notebooks/mnist_enhanced_reservoir.ipynb` teste 3 leviers pour
augmenter la discrimination de la signature `z` :

1. **Rétine électrique** (Difference of Gaussians) : amplifie les bords, annule les aplats.
2. **Multi-injection temporelle** (saccades visuelles) : injection selon plusieurs axes.
3. **Compétition dendritique** (inhibition latérale softmax) : signature sparse.

## Résultats mesurés (honnêtes)
| Réservoir | Dims | Acc |
|---|---|---|
| **BASE** (synaptique 64 zones) | 64 | **0.400** |
| L1 : Rétine (DoG) | 64 | 0.270 |
| **L2 : Multi-axe 2×64** | 128 | **0.412** |
| L3 : Compétition (softmax) | 64 | 0.220 |
| L1+L2+L3 combinés | 64 | 0.190 |

## Conclusion
- **L2 (multi-axe)** est le **seul levier utile** — plus de résolution spatiale
  aide légèrement (0.400 → 0.412).
- **L1 (rétine DoG)** dégrade : `|DoG|` perd la forme globale (annule les aplats).
- **L3 (compétition softmax)** est fortement **nuisible** : le softmax écrase la
  magnitude et détruit l'information discriminante.

=> Sur ce réservoir MNIST, l'**intensité brute + le multi-axe spatial** sont les
meilleures injections. Les transformations bio-inspirées (DoG, softmax compétitif)
ne payent pas à cette échelle de graphe. C'est un résultat scientifique important :
il guide le choix des injections vers la simplicité et la résolution spatiale.

## Connexion du réservoir amélioré au PC
Le réservoir **multi-axe** (meilleur) est connecté au Predictive Coding :

| Réservoir | Dims | Couche lue | Hybride PC |
|---|---|---|---|
| Base synaptique 64 zones | 64 | 0.453 | 0.190 |
| **Amélioré multi-axe 2×64** | 128 | **0.490** | 0.150 |

**Lecture honnête** : le multi-axe améliore la **couche lue** (discrimination,
0.453 → 0.490) mais pas la **classification par tuyaux du PC** (0.190 → 0.150) —
le seuil de nouveauté regroupe toujours plusieurs classes. Le réservoir amélioré
est prêt ; la limite reste la **décision par tuyaux** du mécanisme de nouveauté.

---

# 🧠 Boucle neuromodulée — plasticité gérée par la surprise

Le notebook `notebooks/mnist_neuromodulated.ipynb` ferme la boucle neuromodulée
avec 3 étapes biologiques.

## Les 3 étapes
1. **Plasticité gérée par la surprise** (3-Factor Hebbian) :
   `η(S) = η_base + β·S` — la surprise module le taux d'apprentissage du Physarum.
   - S≈0 → η≈0.1 (rigide, protège la mémoire)
   - S≫0 → η≈0.6 (liquide, reconfigure vite)
2. **Évaluation métabolique temporelle** (arousal) :
   `N_iter(S) = N_min + ⌊α·S⌋` — plus de surprise = plus d'itérations de relaxation.
   - banal → ~5 itérations (économie d'énergie) ; nouveau → ~45 itérations.
3. **Inhibition latérale inter-tuyaux** (compétition corticale) :
   `softmax(A/τ)` ou soustraction — un tuyau dominant écrase les hésitants (WTA doux).

## Résultats mesurés
| Étape | Comportement mesuré |
|---|---|
| η(S) | 0.10 → 0.60 quand S passe de 0 à 1 |
| N_iter(S) | 5 → 45 quand S passe de 0 à 1 |
| Inhibition | softmax [0.52, 0.19, 0.16, 0.13] ; soustractif [0.6, 0, 0, 0] |
| Couche lue neuromodulée | 0.520 |

## Boucle complète
```
[ Chiffre ] → Physarum Synaptique (Flux) ◄──┐
                   │                          │ Surprise S
                   ▼                          │ - plasticité η(S)
             [ Signature z ]                  │ - relaxation N_iter(S)
                   │                          │
                   ▼                          │
        [ Predictive Coding ε = z - ẑ ] ──────┘
                   │
                   ▼
   [ Tuyaux + Inhibition Latérale ] → Décision
```

Le système adapte **sa propre dynamique** selon l'inattendu : connu → rigide et
économe (protège la mémoire contre l'oubli) ; nouveau → plastique et profond
(reconfigure rapidement).

---

# 🔄 Boucle neuromodulée COMPLÈTE

Le notebook `notebooks/mnist_neuromodulated_loop.ipynb` branche toute la boucle :
**réservoir neuromodulé → predictive coding → tuyaux → inhibition latérale → décision**.

```
[ Chiffre ] → Physarum Neuromodulé ◄──┐
                   │                    │ Surprise S (module η et N_iter)
                   ▼                    │
             [ Signature z ]            │
                   │                    │
                   ▼                    │
        [ Predictive Coding ε = z - ẑ ]─┘
                   │
                   ▼
   [ Tuyaux + Inhibition Latérale ] → Décision
```

## Branchement
1. **Prédicteur** : prototypes (moyenne des signatures par classe) pour calculer S = ‖z − ẑ‖.
2. **`NeuromodulatedReservoir`** : S module la plasticité η(S) et la relaxation N_iter(S).
3. **`HybridBlobPredictive`** branché sur le réservoir : le PC opère sur les signatures neuromodulées.
4. **`classify_lateral`** : inhibition latérale (softmax/soustraction) sur les activations des tuyaux → décision WTA doux.

## Résultats mesurés
| Élément | Valeur |
|---|---|
| Surprise S (1er exemple) | 1.261 |
| Couche lue sur z neuromodulée | 0.425 |
| Tuyaux (blob) | 3 (labels 5, 4, 9) |
| Classification standard | 0.190 |
| Classification inhibition latérale | 0.190 |

L'inhibition latérale (softmax sur les activations) donne les activations
compétitives [0.559, 0.311, 0.129] — le tuyau dominant est amplifié. Ici, avec 3
tuyaux, le softmax ne change pas le gagnant (d'où l'égalité standard/latérale).
L'inhibition devient décisive quand **plusieurs tuyaux hésitent** (elle tranche).

---

# ⏱️ SSM local — l'intégration temporelle sous chaque nœud

Le notebook `notebooks/mnist_local_ssm.ipynb` implémente le **SSM local** : chaque
nœud/tuyau du Physarum porte une **mémoire récurrente** $h_t$.

## L'équation
$$h_t = (1 - \Delta_t) \cdot h_{t-1} + \Delta_t \cdot x_t$$

La **surprise** $S_t$ du Predictive Coding pilote la constante de temps :
$$\Delta_t = \sigma(S_t) = \sigma(\Vert z_t - \hat{z}_t \Vert)$$

- $S \approx 0$ (prévisible) → $\Delta \to 0$ : le SSM **garde sa mémoire** (inertie).
- $S \gg 0$ (rupture) → $\Delta \to 1$ : la surprise **réinitialise** la mémoire.

## Résultats mesurés
| Comportement | Valeur |
|---|---|
| Δ = σ(S) | S=0 → 0.5 ; S=3 → 1.0 |
| Mémoire (S faible, 5 pas) | h = 0.98 (accumulation lente) |
| Mémoire (S forte, 5 pas) | h = 1.00 (capture rapide) |
| Rupture (saut x=10) | h : 0 → 9.999 (réinitialisation) |
| Signature [h, z] | 128 dims (64 mémoire + 64 z) |
| Couche lue (mémoire SSM) | 0.467 |

## Fusion espace + temps
- **Physarum** → structure et espace (corrélations spatiales)
- **Micro-SSM sous le nœud** → temps et contexte (inertie temporelle)
- **Surprise (PC)** → vitesse à laquelle le temps s'écoule et se réinitialise

Le `SSMNeuromodulatedReservoir` combine Physarum (espace) + SSM (temps) : la
signature finale est **[h_t, z]** = mémoire temporelle + nouveauté.

---

# 🧩 Fourre-tout sensoriel — intégration multi-modalités

Le notebook `notebooks/mnist_sensory_bundle.ipynb` implémente un **encodeur
multi-modalités** (images, texte, signaux) vers un espace latent commun, connecté
au système.

## Plasticité prédictive (l'encodeur apprend par la surprise)
L'encodeur n'utilise **pas la backprop** mais une règle **Hebbienne à 3 facteurs**
pilotée par la surprise S :

$$\Delta W = \eta(S) \cdot \Big( x^T y - \text{Oja\_Decay}(W, y) \Big), \quad \eta(S) = \beta \cdot S$$

- $S \approx 0$ → $\eta \approx 0$ : W **gelé** (consolidation).
- $S \gg 0$ → plasticité libérée : W s'ajuste aux nouvelles primitives.

## Modalités
| Modalité | Pré-traitement | Latent |
|---|---|---|
| Image | patches (champ récepteur local) | 32 dims |
| Texte | embeddings par caractère | 32 dims |
| Signal | frames temporelles | 32 dims |

Fusion multi-modale : concaténation des latents → vecteur commun
(image seule 32, image+texte 64, +signal 96).

## Scalabilité
Chaque modalité a son `PredictiveEncoder` (Hebbien 3-facteurs). **Ajouter une
modalité = ajouter un encodeur** — le système est conçu pour être étendu aux
autres modalités (audio, vidéo, capteurs...).

## Résultats mesurés
| Élément | Valeur |
|---|---|
| η(S) | 0 → 0.5 quand S va de 0 à 1 |
| Latents par modalité | 32 dims chacun (image, texte, signal) |
| Fusion | 32 / 64 / 96 dims selon nb de modalités |
| Encodage | patches (16×49), embeddings (8×16), frames (11×16) |

Le latent fusionné alimente le **Predictive Coding** : le PC prédit le latent,
l'erreur S pilote la plasticité de l'encodeur (auto-supervision). L'encodeur
doit apprendre (via la plasticité prédictive) pour discriminer les nouveautés.

## Entraînement de l'encodeur — résultats commentés

On entraîne l'encodeur Hebbien sur MNIST et on mesure la **discrimination** des
latents (intra − inter classe). Deux variantes :

| Variante | Discr avant | Discr après | Commentaire |
|---|---|---|---|
| **Oja naïf** | 0.018 | **−0.000** (intra≈inter≈1.0) | **Effondrement** : les latents dégénèrent vers une seule direction |
| **WTA** (winner-take-all) | 0.087 | 0.075 | **Stable** : pas d'effondrement, les filtres se spécialisent |

### Commentaire (3 leçons)
1. **Le Oja naïf s'effondre** : la règle d'Oja pure (sans compétition) fait
   converger tous les latents vers la même représentation — un piège connu de la
   plasticité Hebbienne sans inhibition.
2. **Le WTA stabilise** : l'inhibition latérale interne (seul le filtre max
   répond) force la spécialisation des filtres → représentation stable (discr
   ~0.08), pas d'effondrement.
3. **L'entraînement n'améliore que marginalement** (Δ≈−0.01) : la plasticité
   Hebbienne seule ne crée pas de discrimination multi-classes. La **compétition
   (WTA)** fournit un réservoir stable, mais la **discrimination vient du
   décodage** (couche lue), pas de l'encodeur.

**Conclusion** : l'encodeur Hebbien est un bon **réservoir** (stable grâce au WTA,
apprentissage non supervisé par la surprise), mais pour la classification il faut
le décodage entraîné en aval — c'est le rôle de la couche lue / du PC.

## Classification finale : encodeur WTA + couche lue

On couple l'encodeur WTA (fourre-tout) à une couche lue entraînée pour MNIST.

| Représentation | Dims | Test acc |
|---|---|---|
| Moyenne des latents WTA | 32 | 0.333 |
| **Concaténation des latents WTA** | 512 | **0.653** |
| Réservoir Physarum + lue (base) | 64 | 0.375 |

### Commentaire
1. **La concaténation bat largement la moyenne** (0.653 vs 0.333) : elle préserve
   l'**information spatiale** (chaque patch contribue séparément), rendant la
   représentation discriminante.
2. **L'encodeur WTA + couche lue (0.653) dépasse le réservoir Physarum (0.375)** —
   le fourre-tout sensoriel est maintenant **compétitif** pour la classification.
3. Le pipeline complet : **encodeur Hebbien WTA** (apprentissage non supervisé par
   la surprise) → **latents concaténés** → **couche lue** (décodage entraîné).

Le fourre-tout sensoriel remplit son rôle : il encode les modalités (image, texte,
signal) vers des latents que le décodage peut classer, de façon scalable et sans
backprop sur l'encodeur.

## Pipeline complet : fourre-tout dans le système

On coule l'encodeur WTA dans le pipeline complet — les latents deviennent la
signature du système :

```
[ Image ] → Encodeur WTA (latents concat 512) → [Couche lue | PC + tuyaux]
```

| Élément | Test acc |
|---|---|
| **Couche lue sur l'encodeur** | **0.747** |
| Hybride (PC + tuyaux) | 0.253 |

### Commentaire
1. La **couche lue sur l'encodeur (0.747)** est excellente : le fourre-tout
   produit des latents très discriminants (concaténation spatiale).
2. Le **PC hybride (0.253)** perd comme toujours à cause du mécanisme de tuyaux
   (le seuil de nouveauté regroupe plusieurs classes).
3. Le fourre-tout est **intégré** : image → encodeur WTA → système (réservoir/PC/
   décodage). Il est **scalable** aux autres modalités (texte, signal) via le même
   mécanisme d'encodeur Hebbien.

## Effet de la taille de patch

On teste l'encodeur seul avec différentes tailles de patch (compromis **résolution
spatiale vs primitives locales**) :

| Patch | d_in | # patches | Signature | Test acc |
|---|---|---|---|---|
| **7×7** | 49 | 16 | 512 dims | **0.747** |
| 14×14 | 196 | 4 | 128 dims | 0.320 |
| 28×28 | 784 | 1 | 32 dims | 0.100 |

### Commentaire
1. **Plus le patch est petit, mieux c'est** : patch 7 → 0.747, patch 28 → 0.100.
2. **Petit patch** = beaucoup de patches (16) → la concaténation préserve la
   **structure spatiale** + l'encodeur capture des **primitives locales fines**
   (traits, courbes).
3. **Grand patch** (28 = image entière) = 1 seul patch → pas de résolution
   spatiale, et le WTA sur l'image entière éteint tout sauf 1 filtre → 0.100.
4. Cohérent avec la **vision biologique** : le champ récepteur local (petit) est
   essentiel (les V1 ont de petits champs récepteurs).
5. **Patch 7×7 = meilleur compromis** (0.747).

## Confirmation : encodeur seul vs système complet

| Approche | Test acc |
|---|---|
| **Encodeur seul + couche lue** | **0.747** |
| Système complet (PC + tuyaux) | ~0.25 |

**Confirmé** : l'encodeur seul bat nettement le système complet. Le maillon faible
est le **mécanisme de tuyaux** du PC (seuil de nouveauté qui regroupe les classes).
Le décodage direct (couche lue) est bien supérieur — c'est une conclusion
scientifique importante du projet.

---

# 🧠 Couplage multi-sensoriel — vision + langage

Le notebook `notebooks/mnist_multimodal_context.ipynb` couple **l'image MNIST** et
son **label en texte** (`zero`, `one`...) pour démontrer le **couplage de plusieurs
sens** dans le fourre-tout sensoriel.

```
[ Image ] → Encodeur WTA → latent image ──┐
                                          ├── fusion → couche lue → classification
[ texte ] → Embeddings   → latent texte ──┘
```

## Résultats mesurés (test acc)
| Modalité | Sig | Test acc |
|---|---|---|
| **Image seule** | 512 dims | 0.640 |
| Texte seul (label) | 128 dims | 1.000 |
| **Fusion image+texte** | 640 dims | **1.000** |

## Le texte contextuel corrige la vision
- **Vision seule** : 0.640 (72 erreurs sur 200)
- **Fusion image+texte** : 1.000
- **72/72 erreurs de vision corrigées** par le texte contextuel (figure de vérification)

## Commentaire
1. Le **texte du label** donne une classification parfaite (1.000) : c'est attendu,
   il contient la réponse — c'est une **démonstration du couplage** vision+langage.
2. Le texte **contextuel corrige les ambiguïtés** de la vision : 72/72 erreurs
   visuelles résolues par la fusion.
3. Le fourre-tout couple bien les sens : image → encodeur WTA, texte → embeddings,
   **fusion → décodage**.
4. **Limite méthodologique honnête** : le label exact dans le texte rend la
   classification triviale. Pour un vrai gain, il faudrait un texte contextuel
   **partiel** (ex. "chiffre rond" pour 0/6/8) qui aide sans trancher — une piste
   pour la suite.

---

# 🎯 Pré-entraînement Few-Shot — création de champs récepteurs

Le notebook `notebooks/mnist_fewshot_pretrain.ipynb` implémente la phase de
**pré-entraînement few-shot** de l'encodeur WTA.

## Le paradigme (3 étapes)
1. **Création de champs récepteurs** : la règle d'Oja modulée par la surprise
   `η(S)` aligne les filtres de W sur les gradients réels des données.
2. **Inhibition / spécialisation (WTA)** : un canal qui se spécialise (ex. boucle
   du 0) interdit à ses voisins d'apprendre le même motif → répartition du
   vocabulaire visuel.
3. **Gel des tuyaux (anti-oubli)** : après 10-50 images, on gèle W (`η=0`). Les
   canaux sont stabilisés, prêts pour la classification **Few-Shot**.

## Effet du nombre d'images de pré-entraînement (few-shot, 5 ex/cls)
| Pré-entraînement | Test acc |
|---|---|
| 10 images | 0.400 |
| 20 images | 0.480 |
| 30 images | 0.460 |
| 50 images | 0.555 |
| 100 images | 0.490 |
| 300 images | 0.545 |

## Trade-off few-shot (exemples étiquetés, encodeur gelé sur 30 images)
| Exemples/classe | Test acc |
|---|---|
| 1 | 0.300 |
| 3 | 0.370 |
| 5 | 0.395 |
| 10 | 0.430 |
| 20 | 0.505 |
| 40 | 0.530 |

## Commentaire
1. Le pré-entraînement few-shot **crée des champs récepteurs dès 10-50 images** :
   l'acc est stable (0.40-0.56) quelle que soit la quantité → **peu de
   pré-entraînement suffit**.
2. Le **gel anti-oubli** (η=0) protège les canaux appris.
3. Le trade-off est **dominé par les exemples étiquetés** (1 → 0.30, 40 → 0.53).
4. **Limite** : l'encodeur pré-entraîné sur 30 images (0.59 max) est moins bon
   que sur 300 (0.747) — compromis few-shot : rapidité vs qualité des champs
   récepteurs.

=> Le paradigme few-shot fonctionne : pré-entraînement minimal, gel, puis
classification avec peu d'exemples étiquetés.

---

# 🧠 Classification 100% non supervisée — neurones d'ancrage

Le notebook `notebooks/mnist_unsupervised.ipynb` **retire toute partie supervisée**
(pas de couche lue, pas de régression) et utilise 3 mécanismes Hebbiens.

## Les 3 mécanismes
1. **Neurones d'ancrage** (SOM / K-Means Hebbien) : des neurones concourent pour
   s'activer sur z ; le plus proche ajuste ses poids vers z (Oja). Les clusters
   émergent **sans jamais voir une étiquette**.
2. **WTA dynamique** : `K(S)` varie avec la surprise — S élevée → K augmente
   (analyse détaillée), S faible → K diminue (compact).
3. **Fatigue synaptique / homéostasie** : `θ_i += α·y_i` — un neurone qui gagne
   souvent voit son seuil monter → exploration des primitives.
4. **Top-down predictive feedback** : la prédiction ẑ est renvoyée à l'encodeur
   comme inhibition → seule l'erreur résiduelle passe.

L'**étiquetage est une observation a posteriori** : on observe quel neurone répond
à quelle classe.

## Résultats mesurés
| Mécanisme | Comportement |
|---|---|
| WTA dynamique | K : 2 → 4 quand S passe de 0 à 1 |
| Fatigue | θ du neurone actif : 0 → 0.80 (seuil monte) |
| Top-down | z (norm 2.83) → z_résiduel (norm 1.0) |
| **Classification non supervisée** | **0.490** |
| Pureté moyenne des clusters | 0.505 |

## Commentaire honnête
1. **AUCUNE supervision** : pas de couche lue ni de régression. Les clusters
   émergent des données via Oja/Hebbian (SOM).
2. Le WTA dynamique adapte la sparsité à la surprise ; la fatigue empêche la
   dominance d'un canal ; le top-down transmet l'erreur résiduelle.
3. **Coût** : la non-supervision coûte ~25 points (0.75 supervisé → 0.49), mais
   le système ne voit **aucune étiquette** pendant l'apprentissage — c'est le
   **paradigme autonome** recherché.

---

# 🔄 Rappel Top-Down — régénérer l'image + écrire le mot

Le notebook `notebooks/mnist_topdown_recall.ipynb` fait l'**inverse de la
perception** : projeter un état abstrait vers le bas pour régénérer la forme
brute, sans réseau complexe.

## 1. Régénérer l'image (décodage top-down)
```
[ Neurone d'Ancrage du 1 ] (activation forcée 1.0)
      │  × W_anchor
      ▼
[ Vecteur Latent Reconstruit ẑ ] (512 dims)
      │  × W_encoder^T (rétro-projection)
      ▼
[ Image Synthétique Régénérée (28x28) ]
```

## 2. Écrire le mot (association croisée / résonance)
```
Phase apprentissage : [image] + [texte "un"] → neurone lie les deux
Phase rappel      : [image seule] → neurone s'allume → sort le texte "un"
```

## Résultats mesurés
**Régénération d'image** : la rétro-projection $W_{enc}^T$ régénère une forme
**reconnaissable** (structure préservée, bruitée) — figure de vérification.

**Association visuel→texte** (résonance, mots en français) :
| Chiffre | Texte rappelé | Correct ? |
|---|---|---|
| 0 | zéro | ✓ |
| 1 | un | ✓ |
| 2 | cinq | ✗ |
| 3 | trois | ✓ |
| 4 | quatre | ✓ |
| 5 | trois | ✗ |
| 6 | zéro | ✗ |
| 7 | sept | ✓ |
| 8 | un | ✗ |
| 9 | neuf | ✓ |

**6/10 rappels corrects** — la résonance fonctionne. Les erreurs (2↔5, 6↔0)
viennent de neurones d'ancrage qui regroupent des chiffres visuellement proches
(pureté des clusters ~0.5).

## Commentaire honnête
1. Le système fait l'**inverse de la perception** sans réseau complexe :
   activation d'un neurone → régénération (W_enc^T) + résonance texte.
2. Le décodage est **imparfait** (image bruitée, certains rappels faux) car la
   rétro-projection transposée n'est pas un vrai inverse génératif, et les
   clusters sont imparfaits.
3. C'est la boucle **perception → abstraction → régénération**, entièrement
   non supervisée.

---

# 🎯 Boucle d'Auto-Évaluation de la Surprise (S_auto)

Le notebook `notebooks/mnist_self_eval_loop.ipynb` implémente la **boucle
itérative d'auto-évaluation** qui juge la fidélité de l'image générée au concept
assimilé.

## La boucle
```
Génération (Action) : Neurone i → image prototype x̂
      │
      ▼
Ré-injection (Perception) : x̂ ré-injecté dans le SensoryBundle
      │
      ▼
Comparaison (Jugement) : Surprise d'Auto-Évaluation S_auto = ||z_re - z_cur||
      │
      ▼
Affinement (Correction) :
  - S_auto faible → image fidèle → BOUCLE S'ARRÊTE
  - S_auto élevée → correction du latent (Oja inversée / point fixe) → repart
```

## Résultats mesurés
| Métrique | Valeur |
|---|---|
| S_auto initiale (moyenne, 10 chiffres) | 3.371 |
| S_auto finale | 3.000 |
| Réduction moyenne | 0.371 (**11%**) |
| Convergence (ex. neurone du 7) | 3.44 → 3.00 en 6 itérations |

La figure montre la convergence de S_auto sur les 10 chiffres.

## Commentaire honnête
1. La boucle **converge** : l'affinement itératif réduit la surprise
   d'auto-évaluation (point fixe).
2. **S_auto élevée → correction** du latent (Oja inversée / point fixe) ;
   **S_auto faible → image fidèle** → boucle s'arrête.
3. C'est l'**auto-évaluation** : le système juge lui-même la qualité de sa
   génération, **sans aucune étiquette ni superviseur externe**.
4. La convergence vers 3.0 (pas 0) reflète le **plancher de reconstruction**
   de la rétro-projection transposée — mais la boucle affine l'image vers le
   concept le plus cohérent.

---

# 📊 Visualisation des résultats

Le notebook `notebooks/mnist_visualize_accuracy.ipynb` **visualise** l'évolution
du système non supervisé.

## 1. Évolution de l'accuracy pendant l'apprentissage
| Étape | Accuracy |
|---|---|
| Début (50 images) | 0.354 |
| Milieu (250 images) | 0.475 |
| Fin (400 images) | 0.440-0.500 |

L'accuracy **monte** avec les images vues (0.38 → 0.44+ en moyenne) : les neurones
d'ancrage se spécialisent progressivement (WTA + fatigue). Les oscillations
reflètent la redistribution des gagnants pendant l'apprentissage.

## 2. Convergence S_auto (images avant/après)
La figure montre pour 3 chiffres : l'image réelle, l'image **avant** la boucle
(prototype), l'image **après** la boucle (affinée), et la différence.

| Métrique | Avant | Après |
|---|---|---|
| S_auto (moyenne, 10 chiffres) | 3.368 | **3.000** (−10.9%) |

## 3. Les 3 figures
1. Courbe d'accuracy (montée pendant l'apprentissage)
2. Images avant/après la boucle d'auto-évaluation (3 chiffres)
3. Courbe de convergence S_auto (10 chiffres)

## Conclusion
Les visualisations montrent un système **autonome** qui apprend (accuracy monte),
**juge** (S_auto) et **se corrige** (affinement itératif), sans aucune supervision.

---

# 🏗️ Deep Hebbian hiérarchique (L1 bords → L2 formes)

Le notebook `notebooks/mnist_deep_hebbian.ipynb` refond la classification avec une
**hiérarchie** : L1 apprend des détecteurs de bords (petits patches), L2 combine
ces bords en formes (boucles, intersections).

## Les mécanismes
1. **Hiérarchie Deep Hebbian** : L1 (patches 4-7×4-7 → 32-64 bords) → L2 (→ 64-128 formes).
2. **Anti-Hebbian** : décorrèle les neurones qui gagnent ensemble (spécialisation).
3. **Soft-WTA** (SoftHebb) : sélection probabiliste des gagnants à l'apprentissage,
   déterministe à l'inférence.

Le **saliency-gate a été retiré** : il bloquait trop l'apprentissage (62% des
mises à jour à eta=0). On apprend sur chaque entrée (Deep Hebbian pur).

## Résultats mesurés
| Réglage | Dims | Test acc (supervisé) |
|---|---|---|
| Base (n_learn=3) | 96 | 0.230 |
| n_learn=8 | 96 | 0.235 |
| Plus large (64/128) | 192 | 0.340 |
| patch 7, plus large | 192 | **0.395** |

Classification non supervisée (ancres) : 0.170.

## Analyse honnête
Le Deep Hebbian hiérarchique **ne dépasse pas l'encodeur simple (0.747)**. Le
plafond est ~0.40 même avec un encodeur plus large. **Causes probables** :
1. Le **soft-WTA probabiliste + l'agrégation L1→L2 par somme détruisent la
   structure spatiale** (la position des bords est perdue).
2. La littérature (HMAX, PCANet) garde des **feature maps positionnées** entre
   les couches — pas une somme globale.
3. L'anti-Hebbian et la sparsité réduisent encore la capacité.

=> La hiérarchie est le bon concept mais nécessite de **préserver l'espace**
(feature maps) entre L1 et L2, pas une agrégation par somme — une piste pour la
suite.

## Feature maps SPATIALES (SpatialDeepHebbian)

On préserve la **structure spatiale** entre L1 et L2 (la cause n°1 de l'échec) :
- **L1** : `W_l1` PARTAGÉ (détecteurs de bords invariants par translation) →
  feature map positionnée F ∈ (H,W,n_l1).
- **L2** : **convolution Hebbienne** sur voisinages 3×3 de F → combine les bords
  en formes à leur position (équivalent CNN non supervisé).

## Bilan final : Hebbian pur plafonne

| Approche | Test acc (supervisé) |
|---|---|
| DeepHebbian (somme) | 0.395 |
| **SpatialDeepHebbian (feature maps)** | 0.355 |
| Encodeur simple + couche lue | **0.747** |

**Conclusion honnête** :
1. Les feature maps spatiales ne battent **pas** l'encodeur simple.
2. Même en préservant l'espace, l'**Hebbian pur plafonne à ~0.39**.
3. Le vrai goulot = la **supervision**, pas l'architecture : le décodage
   supervisé (couche lue) sur l'encodeur simple atteint 0.747.
4. L'apprentissage Hebbian (sans régression) **capture les primitives** (bords,
   formes) mais **pas la séparation des classes**.
5. La hiérarchie spatiale est le bon concept pour les **primitives** ; la
   **classification** nécessite un décodage plus puissant.

---

# 🔁 Test du modèle drift — anti-oubli catastrophique

Le notebook `notebooks/mnist_drift_test.ipynb` observe comment les **AnchorNeurons
(avec fatigue homéostatique)** s'adaptent à un **drift** :
1. **Phase 1** : présenter uniquement des **0 et des 1**.
2. **Phase 2** : introduire **progressivement 2-9**, sans jamais ré-entraîner sur 0/1.
3. **Observation** : allouer de **nouveaux neurones** pour 2-9 **sans détruire**
   ceux de 0/1 (résolution de l'oubli catastrophique).

## Résultats mesurés (200 neurones)
| Après intro | Acc 0/1 (préservé) | Acc nouveau | Neurones actifs |
|---|---|---|---|
| Phase 1 | 1.000 | — | 121 |
| 2 | 1.000 | 0.840 | 170 |
| 3 | 0.990 | 0.880 | 187 |
| 4 | 0.970 | 0.780 | 199 |
| 5 | 0.970 | 0.600 | 200 |
| 6 | 0.980 | 0.380 | 200 |
| 7 | 0.960 | 0.440 | 200 |
| 8 | 0.990 | 0.280 | 200 |
| 9 | 0.980 | 0.380 | 200 |

## Conclusion
1. **Oubli catastrophique RÉSOLU** : acc 0/1 = 1.000 → **0.98** (préservé), les
   neurones de 0/1 ne sont pas détruits.
2. **Allocation progressive** : 121 → 200 neurones — de nouveaux neurones
   s'allouent pour 2-9 **sans ré-entraîner** les anciens.
3. **Les nouveaux chiffres s'apprennent** (acc 2=0.84, 3=0.88). La dégradation
   des derniers (5-9, ~0.28-0.44) vient de la **saturation des 200 neurones**
   et de la fatigue qui redistribue — augmenter la capacité aide.

La **fatigue homéostatique + neurones d'ancrage** permettent un apprentissage
incrémental sans oubli catastrophique — un résultat clé pour le système autonome.

---

# 🌿 Physarum + AnchorNeurons — élagage dynamique sous drift

Le notebook `notebooks/mnist_physarum_pruning.ipynb` teste le **rôle de Physarum**
pour résoudre le goulot d'étranglement (200 neurones saturés → acc 5-9 dégradé).

## Le mécanisme
Physarum excelle dans l'**élagage** (atrophie des tuyaux à faible flux) et la
**réallocation**. On modélise chaque neurone d'ancrage comme un tuyau (flux =
activation cumulée) :
- **Consolider** les autoroutes utiles (fort flux → 0/1, 2, 3)
- **Atrophier** les neurones sous-utilisés → libérer du budget pour 5-9

`AnchorNeurons.physarum_prune()` élague les neurones à plus faible flux.

## Résultats (protocole drift identique, 200 neurones)
| Config | Acc 0/1 fin | **Acc 5-9 moy** | Neurones |
|---|---|---|---|
| **Sans Physarum** | 0.980 | **0.508** | 200 (saturés) |
| prune 0.15 | 1.000 | 0.860 | 170 |
| prune 0.2 | 0.990 | 0.760 | 160 |
| **prune 0.3** | 1.000 | **0.904** | 140 |

**GAIN sur 5-9 : +0.372 (+73%)** — avec MOINS de neurones actifs.

## Conclusion
1. **Physarum RÉSOUT le goulot** : en élaguant les neurones sous-utilisés, il
   libère du budget pour 5-9 **sans saturer** les 200 neurones.
2. **0/1 reste > 0.95 (préservé)** et 5-9 monte nettement (0.51 → 0.90).
3. **Avec moins de neurones** (140 vs 200) : les ressources sont mieux exploitées
   (atrophie des redondants, consolidation des autoroutes utiles).
4. C'est une **démonstration d'adaptation dynamique sous contrainte de
   ressources** — Physarum n'est pas une dégradation ici, c'est le levier qui
   fait passer le système au-dessus des seuils cibles (5-9 > 0.70, 0/1 > 0.95).

## Résultats mesurés
- **Couche lue sur z synaptique** : train 0.453 / test 0.393 (signature 64 dims
  au lieu de 364 flux bruts — très compacte).
- **PC + blob** : détecte la nouveauté (5, 0, 4 créent des tuyaux), consolidation
  durcit (conductance 2.14).
- **Limite** : la classification par tuyaux reste ~0.19 — le seuil de nouveauté
  regroupe plusieurs classes dans un même tuyau.

## Lecture honnête
La connexion **réservoir synaptique → PC** est fonctionnelle et démontre le
mécanisme complet : représentation compacte discriminante (z), prédiction/erreur,
détection de nouveauté, création/consolidation de tuyaux (mémoire). L'intérêt est
le **mécanisme** (apprentissage incrémental sans réentraîner tout), pas la
précision brute.
