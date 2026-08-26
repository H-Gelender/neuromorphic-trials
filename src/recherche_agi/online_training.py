"""Entraînement en ligne sur COCO Stuff avec CALLBACK D'ÉQUILIBRE.

Le callback stoppe l'envoi d'images quand un ÉQUILIBRE se crée, via 3 critères
convergents :

A. Taux de variation des poids :  ||W(t) - W(t-1)||_F < eps_W
B. Dissipation de la surprise résiduelle : dS/dt ~ 0 (plateau minimal)
C. Stabilisation du flux Physarum : plus de création/suppression de connexions

+ arrêt automatique au-delà d'une durée max (par défaut 1h).
"""
import time
import numpy as np


class EquilibriumCallback:
    """Détecte l'équilibre d'apprentissage et arrête l'entraînement."""

    def __init__(self, eps_W=1e-4, eps_S=1e-3, stable_window=20,
                 max_duration_s=3600, check_every=10):
        self.eps_W = eps_W
        self.eps_S = eps_S
        self.stable_window = stable_window   # nb d'itérations stables requises
        self.max_duration_s = max_duration_s  # 1h par défaut
        self.check_every = check_every
        self.history = {
            'dW': [],       # variation des poids
            'S': [],        # surprise de reconstruction
            'D': [],        # variation du flux (connexions co_act)
            'time': [],
        }
        self._last_W = None
        self._last_co = None
        self._stable_count = 0
        self._start = time.time()
        self.stopped_by = None

    def on_image(self, W, co_act, surprise):
        """Appelé à chaque image. Retourne True si on doit s'arrêter."""
        # --- critère A : variation des poids ---
        if self._last_W is not None:
            dW = float(np.linalg.norm(W - self._last_W))
        else:
            dW = float('inf')
        self._last_W = W.copy()

        # --- critère C : variation du flux Physarum (connexions) ---
        if self._last_co is not None:
            dD = float(np.linalg.norm(co_act - self._last_co))
        else:
            dD = float('inf')
        self._last_co = co_act.copy()

        # --- critère B : surprise ---
        S = float(surprise)

        self.history['dW'].append(dW)
        self.history['S'].append(S)
        self.history['D'].append(dD)
        self.history['time'].append(time.time() - self._start)

        # ne vérifier l'équilibre qu'à partir d'un certain volume
        n = len(self.history['dW'])
        if n < self.check_every or n % self.check_every != 0:
            return False

        # équilibre si les 3 sont stables sur la fenêtre
        stable = self._is_stable('dW', self.eps_W) and \
                 self._is_stable('D', self.eps_W) and \
                 self._is_stable('S', self.eps_S)
        if stable:
            self._stable_count += 1
        else:
            self._stable_count = 0

        # arrêt par équilibre
        if self._stable_count >= self.stable_window:
            self.stopped_by = 'equilibre'
            return True

        # arrêt par durée max
        if (time.time() - self._start) > self.max_duration_s:
            self.stopped_by = 'duree_max'
            return True

        return False

    def _is_stable(self, key, eps):
        h = self.history[key]
        if len(h) < self.check_every + 1:
            return False
        # stabilité = la DERIVÉE (variation récente) est petite
        recent = np.array(h[-self.check_every:])
        # variation d'une itération à l'autre
        diff = np.abs(np.diff(recent))
        return float(diff.mean()) < eps

    def elapsed(self):
        return time.time() - self._start

    def summary(self):
        return {
            'stopped_by': self.stopped_by,
            'elapsed_s': round(self.elapsed(), 1),
            'n_images': len(self.history['dW']),
            'last_dW': self.history['dW'][-1] if self.history['dW'] else None,
            'last_S': self.history['S'][-1] if self.history['S'] else None,
            'last_D': self.history['D'][-1] if self.history['D'] else None,
        }
