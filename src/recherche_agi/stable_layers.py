"""Système de couches stables — gel dynamique + création de couches.

Implémente le cycle respiratoire (compromis Plasticité/Stabilité) :

1. SIGNAL D'ACTIVITÉ STRUCTURELLE S(t) = taux de création - taux d'élagage
   + détection d'oscillation par FFT (pic fréquentiel net = instabilité).
2. TROIS LEVIERS quand une oscillation est détectée :
   - Amortissement (augmenter la viscosité λ du Physarum)
   - Méta-neurones (fusionner les neurones oscillants)
   - Spawning de couche (geler C1 + créer C2 pour absorber la pression)
3. DÉCLENCHEUR DE DÉGEL : pic de surprise incompressible
   S = ||x - W1^T z1||^2 qui reste élevé -> changement d'environnement.
4. PLASTICITÉ SÉLECTIVE : neurogenèse d'extension (W1 gelé, ajout de neurones)
   + métro-plasticité par recuit simulé eta1(t) = eta_base * f(Surprise).
"""
import numpy as np

__all__ = ["StructuralActivitySignal", "RespiratoryController"]


class StructuralActivitySignal:
    """Enregistre la création/élagage au fil du temps et analyse S(t)."""

    def __init__(self, window: int = 64, sample_every: int = 5):
        self.window = window
        self.sample_every = sample_every
        self.samples = []          # valeurs S(t) échantillonnées
        self.creations = 0
        self.prunings = 0
        self._counter = 0
        self._pending_create = 0
        self._pending_prune = 0

    def event_create(self, n: int = 1):
        self._pending_create += n

    def event_prune(self, n: int = 1):
        self._pending_prune += n

    def tick(self):
        """À chaque pas, échantillonne S(t) = création - élagage (lissé)."""
        self._counter += 1
        if self._counter % self.sample_every == 0:
            S = self._pending_create - self._pending_prune
            self.samples.append(float(S))
            # lisser (rémanence) : on garde le cumul mais on amortit
            self._pending_create *= 0.5
            self._pending_prune *= 0.5
            if len(self.samples) > self.window:
                self.samples.pop(0)

    def detect_oscillation(self) -> dict:
        """Analyse fréquentielle de S(t). Retourne si oscillant + la fréquence.

        - Signal plat / bruit blanc -> stable (pas de pic net).
        - Pic fréquentiel net (énergie concentrée sur une bande) -> oscillation
          (cycle limite) -> instabilité structurelle.
        """
        if len(self.samples) < 16:
            return {'oscillating': False, 'dominant_freq': None,
                    'spectral_concentration': 0.0, 'energy': 0.0}
        x = np.array(self.samples)
        x = x - x.mean()
        # énergie (variance) : turbulence de flux
        energy = float(np.mean(x ** 2))
        # FFT
        X = np.abs(np.fft.rfft(x))
        if X.sum() == 0:
            return {'oscillating': False, 'dominant_freq': None,
                    'spectral_concentration': 0.0, 'energy': energy}
        freqs = np.fft.rfftfreq(len(x), d=1.0)
        # ignorer DC (freq 0)
        if len(X) > 1:
            X[0] = 0
            freqs = freqs[1:]
            X = X[1:]
        if len(X) == 0 or X.max() == 0:
            return {'oscillating': False, 'dominant_freq': None,
                    'spectral_concentration': 0.0, 'energy': energy}
        # concentration spectrale : part de l'énergie dans le pic dominant
        peak = X.max()
        concentration = float(peak / X.sum())
        dom_freq = float(freqs[int(np.argmax(X))])
        # oscillation si un pic net (concentration > seuil) ET énergie non nulle
        oscillating = concentration > 0.25 and energy > 0.02
        return {'oscillating': oscillating, 'dominant_freq': dom_freq,
                'spectral_concentration': concentration, 'energy': energy}

    @property
    def last_S(self):
        return self.samples[-1] if self.samples else 0.0


class RespiratoryController:
    """Contrôle le cycle respiratoire : gel/dégel + spawning de couche.

    Pilote la stabilité structurelle :
    - détecte l'oscillation (cycle limite) via le signal S(t)
    - applique les leviers : amortissement λ, méta-neurones, spawning de couche
    - déclenche le dégel par pic de surprise incompressible
    - plasticité sélective (neurogenèse d'extension, η(S) par recuit simulé)
    """

    def __init__(self, surprise_unfreeze: float = 2.0, unfreeze_streak: int = 5,
                 base_lr: float = 0.1, signal_window: int = 64):
        self.signal = StructuralActivitySignal(window=signal_window, sample_every=3)
        self.frozen = False
        self.layers = {}              # {0: {'neurons': n, 'frozen': bool}}
        self.layer_count = 1
        self.lambda_damp = 1.0        # viscosité Physarum (amortissement)
        self.phase = 'exploration'    # exploration | consolidation | perturbation
        self.surprise_unfreeze = surprise_unfreeze
        self.unfreeze_streak = unfreeze_streak
        self.base_lr = base_lr
        self.surprise_history = []
        self._high_surprise_count = 0
        self.log = []                 # historique des événements

    # -- Surprise de reconstruction (déclencheur de dégel) --
    def reconstruction_surprise(self, x: np.ndarray, W1) -> float:
        """S = ||x - W1^T z1||^2 : échec de reconstruction par le dictionnaire gelé."""
        zn = x / (np.linalg.norm(x) + 1e-8)
        sim = W1 @ zn
        z1 = np.zeros_like(sim); z1[int(np.argmax(sim))] = sim.max()
        recon = W1.T @ z1
        return float(np.linalg.norm(zn - recon) ** 2)

    def check_unfreeze(self, surprise: float) -> bool:
        """Pic de surprise incompressible pendant plusieurs images -> dégel."""
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > 10:
            self.surprise_history.pop(0)
        if surprise > self.surprise_unfreeze:
            self._high_surprise_count += 1
        else:
            self._high_surprise_count = 0
        return self._high_surprise_count >= self.unfreeze_streak

    # -- Métro-plasticité (recuit simulé) --
    def effective_lr(self, surprise: float) -> float:
        """eta1(t) = eta_base * f(Surprise) ; retombe à 0 si gelé."""
        if self.frozen:
            return 0.0
        # f(Surprise) : plastique si surprise, amorti sinon
        return self.base_lr * float(np.clip(surprise / self.surprise_unfreeze, 0, 1))

    # -- Leviers d'architecture --
    def apply_damping(self, factor: float = 1.5):
        """Levier 1 : augmenter la viscosité λ (amortir l'oscillation)."""
        self.lambda_damp *= factor

    def record_create(self, n=1): self.signal.event_create(n)
    def record_prune(self, n=1): self.signal.event_prune(n)

    def tick(self):
        """Échantillonne S(t) et détecte l'oscillation -> met à jour la phase."""
        self.signal.tick()
        r = self.signal.detect_oscillation()
        if r['oscillating']:
            if self.phase != 'consolidation':
                self.log.append('oscillation détectée -> consolidation (gel/spawn)')
            self.phase = 'consolidation'
        else:
            # stable : pas d'oscillation
            if self.phase == 'consolidation':
                self.log.append('oscillation amortie -> stabilisation')
            self.phase = 'exploration' if not self.frozen else 'consolidation'
        return r

    def spawn_layer(self):
        """Levier 3 : gel de C1 + déploiement de C2."""
        self.frozen = True
        self.layer_count += 1
        self.layers[self.layer_count-1] = {'frozen': True}
        self.layers[self.layer_count] = {'frozen': False, 'created_at': len(self.log)}
        self.log.append(f'spawning : couche {self.layer_count} créée, C1 gelée')
        return self.layer_count

    def unfreeze_extension(self):
        """Dégel par neurogenèse d'extension : C1 reste gelée mais on autorise
        l'AJOUT de neurones pour absorber la nouveauté."""
        self.frozen = False  # autorise l'extension (ajout de neurones)
        self.log.append('dégel : neurogenèse d\'extension activée')
        return True

    def summary(self):
        return {
            'phase': self.phase,
            'frozen': self.frozen,
            'layers': self.layer_count,
            'lambda_damp': self.lambda_damp,
            'S': self.signal.last_S,
        }
