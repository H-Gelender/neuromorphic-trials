"""Classification NON supervisée — neurones d'ancrage + WTA dynamique + fatigue.

Retire toute partie supervisée (pas de couche lue / régression). Le système crée
spontanément des clusters (tuyaux) via des règles Hebbiennes.

1. NEURONES D'ANCRAGE (SOM / K-Means Hebbien) :
   Une couche de neurones concourent pour s'activer sur le vecteur creux z.
   Le plus proche (WTA catégoriel) ajuste ses poids vers z (mise à jour Oja).
   → clusters spontanés correspondant aux catégories, SANS étiquettes.
   L'étiquetage n'est qu'une observation a posteriori.

2. WTA DYNAMIQUE (K varie avec la surprise S) :
   - S élevé (scène inconnue) → K augmente (sparsité faible, analyse détaillée)
   - S faible (motif connu) → K diminue (sparsité extrême, compact)

3. FATIGUE SYNAPTIQUE / HOMÉOSTASIE :
   seuil d'activation θ_i(t+1) = θ_i(t) + α·y_i
   empêche qu'un même neurone ne gagne toujours → exploration des primitives.

Inspiration : Kohonen (SOM), Homeostatic plasticity, WTA cortical.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AnchorNeurons", "dynamic_k", "homeostatic_threshold",
           "topdown_feedback"]


def topdown_feedback(z: np.ndarray, prediction: np.ndarray, beta: float = 1.0
                     ) -> np.ndarray:
    """Top-down predictive feedback : inhibition par la prédiction.

    La prédiction du réservoir ẑ (le prototype du neurone gagnant) est renvoyée
    vers l'encodeur comme INHIBITION : les primitives déjà prédites sont éteintes,
    ne laissant passer que l'ERREUR RÉSIDUELLE (le résidu de surprise).

        z_residuel = z - β·ẑ
    """
    z = np.asarray(z, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    pred = pred / (np.linalg.norm(pred) + 1e-8)
    residual = z - beta * pred
    return residual / (np.linalg.norm(residual) + 1e-8)


def dynamic_k(S: float, k_min: int = 1, k_max: int = 5, steepness: float = 2.0,
              midpoint: float = 0.5) -> int:
    """Nombre de gagnants K du WTA, fonction de la surprise S.

    K(S) = k_min + (k_max - k_min) · σ(steepness·(S - midpoint))
    - S faible → K ≈ k_min (sparsité extrême)
    - S élevée → K ≈ k_max (analyse détaillée)
    """
    sig = 1.0 / (1.0 + np.exp(-steepness * (S - midpoint)))
    return int(round(k_min + (k_max - k_min) * sig))


def homeostatic_threshold(theta: np.ndarray, y: np.ndarray, alpha: float = 0.05,
                          decay: float = 0.001) -> np.ndarray:
    """Fatigue synaptique : θ(t+1) = θ(t) + α·y - decay.

    Les neurones très actifs voient leur seuil monter (fatigue) → ils répondent
    moins → les autres peuvent s'activer (exploration).
    """
    return theta + alpha * y - decay


class AnchorNeurons:
    """Neurones d'ancrage — carte auto-organisée non supervisée.

    Chaque neurone est un prototype (poids) qui s'ajuste vers les z qu'il gagne
    (WTA Hebbien/Oja). Sans supervision : les clusters émergent des données.

    L'étiquetage a posteriori : on observe quel neurone répond à quelle classe.
    """

    def __init__(self, d_in: int, n_neurons: int, seed: int = 0, lr: float = 0.1,
                 use_homeostasis: bool = True, homeo_alpha: float = 0.05):
        rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.n_neurons = n_neurons
        self.lr = lr
        self.use_homeostasis = use_homeostasis
        self.homeo_alpha = homeo_alpha
        # prototypes normalisés (unit norme)
        self.W = rng.normal(0, 1 / np.sqrt(d_in), size=(n_neurons, d_in))
        self.W = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-8)
        self.theta = np.zeros(n_neurons)          # seuils de fatigue
        self.activations = np.zeros(n_neurons)    # cumul des activations
        self.labels_seen = {}                      # neurone -> classes observées

    def activate(self, z: np.ndarray) -> np.ndarray:
        """Activations (similarité cosinus, seuillées par fatigue).

        Retourne le vecteur d'activations a (avant WTA), modulé par la fatigue.
        """
        z = np.asarray(z, dtype=float)
        z = z / (np.linalg.norm(z) + 1e-8)
        sim = self.W @ z                      # (n_neurons,) cosinus
        if self.use_homeostasis:
            sim = sim - self.theta            # fatigue : seuil adaptatif
        return sim

    def win(self, z: np.ndarray, k: int = 1) -> np.ndarray:
        """WTA dynamique : les k neurones les plus actifs."""
        a = self.activate(z)
        idx = np.argsort(-a)[:k]
        out = np.zeros(self.n_neurons)
        out[idx] = a[idx]
        return out

    def learn(self, z: np.ndarray, k: int = 1, label: int = None) -> np.ndarray:
        """Présente z, met à jour les k gagnants (Oja/Hebbian), fatigue les gagnants.

        Sans supervision : label optionnel sert seulement à l'observation a
        posteriori (ne pilote pas l'apprentissage).
        """
        z = np.asarray(z, dtype=float)
        zn = z / (np.linalg.norm(z) + 1e-8)
        a = self.activate(z)
        idx = np.argsort(-a)[:k]
        # mise à jour Hebbienne/Oja des gagnants vers z
        for i in idx:
            w = self.W[i]
            w_new = w + self.lr * (zn - w)   # déplacement vers z (Kohonen/Oja)
            self.W[i] = w_new / (np.linalg.norm(w_new) + 1e-8)
            self.activations[i] += 1.0
            # fatigue : le gagnant voit son seuil monter
            if self.use_homeostasis:
                self.theta[i] = self.theta[i] + self.homeo_alpha * 1.0
                self.theta[i] = max(self.theta[i], 0.0)
            # observation a posteriori (étiquetage)
            if label is not None:
                self.labels_seen.setdefault(int(i), {}).setdefault(int(label), 0)
                self.labels_seen[int(i)][int(label)] += 1

        # retour : vecteur d'activation (creux) = signature non supervisée
        out = np.zeros(self.n_neurons)
        out[idx] = a[idx]
        return out

    def predict_label(self, z: np.ndarray) -> tuple[int | None, float]:
        """Classification a posteriori : le neurone gagnant et sa classe dominante.

        Le neurone le plus actif a été observé majoritairement sur une classe.
        """
        a = self.activate(z)
        winner = int(np.argmax(a))
        if winner not in self.labels_seen:
            return None, float(a[winner])
        # classe dominante observée sur ce neurone
        cls = max(self.labels_seen[winner].items(), key=lambda x: x[1])[0]
        return cls, float(a[winner])

    def cluster_labels(self) -> dict:
        """Carte neurone -> classe dominante (a posteriori, observation)."""
        out = {}
        for i, cnt in self.labels_seen.items():
            out[i] = max(cnt.items(), key=lambda x: x[1])[0]
        return out

    # ------------------------------------------------------------------ #
    # ÉLAGAGE TYPE PHYSARUM (réallocation dynamique des ressources)
    # ------------------------------------------------------------------ #
    def physarum_prune(self, prune_frac: float = 0.15, min_keep: int = 5,
                       seed: int = None) -> int:
        """Élague les neurones à faible flux d'utilisation (atrophie Physarum).

        Modélise chaque neurone comme un tuyau : son ACTIVATION cumulée est le
        flux D. Les neurones sous-utilisés ou redondants (faible flux) sont
        ATROPHIÉS (réinitialisés), libérant du budget pour apprendre de
        nouveaux motifs — sans détruire les neurones utiles (fort flux).

        Retourne le nombre de neurones élagués.
        """
        act = self.activations.copy()
        if act.sum() > 0:
            flux = act / act.sum()
        else:
            flux = act
        n_prune = max(0, int(prune_frac * len(act)))
        if len(act) - n_prune < min_keep:
            n_prune = len(act) - min_keep
        idx = np.argsort(flux)[:n_prune]   # les neurones au plus faible flux
        rng = np.random.default_rng(seed)
        for i in idx:
            self.W[i] = rng.normal(0, 1/np.sqrt(self.d_in), size=self.d_in)
            self.W[i] /= np.linalg.norm(self.W[i]) + 1e-8
            self.theta[i] = 0.0
            self.activations[i] = 0.0
            self.labels_seen.pop(i, None)
        return n_prune

    # ------------------------------------------------------------------ #
    # RAPPEL TOP-DOWN GÉNÉRATIF (inverse de la perception)
    # ------------------------------------------------------------------ #
    def reconstruct_latent(self, neuron_idx: int) -> np.ndarray:
        """Régénère le latent ẑ à partir d'un neurone d'ancrage activé.

        Activer le neurone i (one-hot) → projeter par W_anchor :
            ẑ = W_anchor[i]   (le poids du neurone est déjà un prototype latent)
        """
        return self.W[neuron_idx].copy()

    def reconstruct_image(self, neuron_idx: int, enc_W: np.ndarray,
                          patch_size: int = 7, img_shape: tuple = (28, 28)
                          ) -> np.ndarray:
        """Régénère une image depuis un neurone d'ancrage (rétro-projection).

        ẑ = W_anchor[i]  (prototype latent, 512 dims)
        → déconcatène en latents de patches (32 dims chacun)
        → rétro-projection par W_enc^T vers l'espace des patches
        → reconstruit l'image 28x28 par blocs.

        enc_W : matrice de l'encodeur WTA (d_out x d_in) = (32, 49).
        """
        z_hat = self.reconstruct_latent(neuron_idx)
        d_lat = enc_W.shape[0]                 # 32 (latent par patch)
        n_patches = z_hat.shape[0] // d_lat    # 512/32 = 16 patches
        gh = gw = int(np.sqrt(n_patches))      # 4x4 patches

        # rétro-projeter chaque latent de patch vers le patch d'origine
        patch_h = patch_w = patch_size
        img = np.zeros(img_shape)
        for idx in range(n_patches):
            z_p = z_hat[idx*d_lat:(idx+1)*d_lat]
            # pseudo-inverse / transposée : patch ≈ W_enc^T · z_p
            patch = enc_W.T @ z_p
            patch = patch - patch.min()        # normaliser par patch
            pmax = patch.max()
            if pmax > 0:
                patch = patch / pmax
            i, j = idx // gw, idx % gw
            img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patch.reshape(patch_h, patch_w)
        return img

    # -- association croisée visuel -> texte (résonance hebbienne) -- #
    def associate_text(self, neuron_idx: int, text_latent: np.ndarray,
                       alpha: float = 0.1) -> None:
        """Lie un latent textuel au neurone d'ancrage (mémoire associative).

        Phase d'apprentissage : le neurone i est co-activé avec le texte.
        La composante textuelle est mémorisée par moyennage hebbien.
        """
        tl = np.asarray(text_latent, dtype=float)
        tl = tl / (np.linalg.norm(tl) + 1e-8)
        if not hasattr(self, 'text_mem'):
            self.text_mem = {}   # neurone -> (somme, compteur)
        if neuron_idx not in self.text_mem:
            self.text_mem[neuron_idx] = (np.zeros_like(tl), 0)
        s, c = self.text_mem[neuron_idx]
        self.text_mem[neuron_idx] = (s + alpha * tl, c + 1)

    def recall_text(self, neuron_idx: int) -> np.ndarray | None:
        """Rappelle le texte associé au neurone (résonance, lecture W_texte)."""
        if not hasattr(self, 'text_mem') or neuron_idx not in self.text_mem:
            return None
        s, c = self.text_mem[neuron_idx]
        if c == 0:
            return None
        lat = s / c
        return lat / (np.linalg.norm(lat) + 1e-8)

    # ------------------------------------------------------------------ #
    # BOUCLE D'AUTO-ÉVALUATION DE LA SURPRISE (génération -> jugement)
    # ------------------------------------------------------------------ #
    def self_evaluate_loop(self, neuron_idx: int, encode_fn, recon_fn,
                           n_iter: int = 6, lr_affine: float = 0.3,
                           verbose: bool = True) -> dict:
        """Boucle d'auto-évaluation de la surprise S_auto.

        1. GÉNÉRATION : le neurone génère l'image prototype x̂ (rétro-projection).
        2. RÉ-INJECTION : l'image est ré-injectée dans l'encodeur.
        3. COMPARAISON : surprise d'auto-évaluation S_auto = ||z_re - z_cur||.
        4. AFFINEMENT : si S_auto élevée, on déplace z_cur vers le latent re-encodé
           (règle de point fixe / Oja inversée), puis on repart.

        Retourne l'historique des S_auto + le latent affiné final.
        """
        z_cur = self.reconstruct_latent(neuron_idx).copy()
        hist = []
        for _ in range(n_iter):
            img = recon_fn(z_cur)                 # 1. génération
            z_re = encode_fn(img)                 # 2. ré-injection
            S_auto = float(np.linalg.norm(z_re - z_cur))  # 3. comparaison
            hist.append(S_auto)
            if verbose:
                print(f"    it {len(hist)}: S_auto = {S_auto:.4f}")
            # 4. affinement (point fixe) : S_auto élevée → correction
            z_cur = z_cur + lr_affine * (z_re - z_cur)
            z_cur = z_cur / (np.linalg.norm(z_cur) + 1e-8)
        return {'S_auto_history': hist, 'latent_final': z_cur,
                'S_auto_initial': hist[0] if hist else None,
                'S_auto_final': hist[-1] if hist else None}
