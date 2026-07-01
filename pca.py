import numpy as np
import json
from skops.io import load as skops_load, get_untrusted_types

# ---------------------------------------------------------------------------
# Hardcoded variables
# ---------------------------------------------------------------------------

TRACKER_Z = np.sort(np.array([
    -1.11320,  0.38680,  1.88680,  3.38680,  4.88680,
     6.38680,  7.88680,  9.38680, 10.88680, 12.38680,
]))

DETECTOR_XY_MAX = 22.725  # cm

FEATURE_NAMES = [
    'n_pts_raw', 'n_pts_clean',
    'R_raw',     'R_clean',
    'L1_raw',    'L2_raw',    'L3_raw',
    'L1_clean',  'L2_clean',  'L3_clean',
    'lat_raw',   'lat_clean',
    'theta',     'phi',       'z_max',
    'first_hit_top', 'plane_coverage', 'e_tot',
]

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def _clean_single(coord, threshold=0.99, min_pts=3):
    """Iterative outlier removal.
    Returns (n_pts, R, eig3, principal_axis).
    """
    coord = coord.copy()
    while len(coord) > min_pts:
        R, eig, axis = _pca(coord)
        if R >= threshold:
            return len(coord), R, eig, axis
        Q  = coord - coord.mean(axis=0)
        d2 = (Q * Q).sum(axis=1) - (Q @ axis) ** 2
        coord = np.delete(coord, int(np.argmax(d2)), axis=0)

    # soglia non raggiunta: restituisci quel che resta
    R, eig, axis = _pca(coord)
    return len(coord), R, eig, axis


def getCoord(hits):
    return hits[0, :3, :].numpy().T


def make_pca(coords):
    min_pts = 3
    N = len(coords)
    if N < min_pts:
        return None
    centered = coords - coords.mean(axis=0)

    # SVD — with (N, 3) e full_matrices=False:
    # U → (N, 3),  s → (3,),  Vt → (3, 3)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    # Normalized Eigenvalues
    eig   = (s ** 2) / N                   # shape (3,)
    denom = eig.sum()
    R      = (eig[0] / denom) if denom > 0 else 0.0
    axis   = Vt[0]                          # first pc, shape (3,)
    return R, eig.astype(np.float32), axis.astype(np.float32)


def _pca(coord):
    n        = len(coord)
    c        = coord.mean(axis=0)
    Q        = coord - c
    _, s, Vt = np.linalg.svd(Q, full_matrices=False)
    eig      = np.zeros(3, dtype=np.float64)
    eig[:len(s)] = s ** 2 / n
    denom    = eig.sum()
    R        = float(eig[0] / denom) if denom > 0 else 0.0
    return R, eig, Vt[0]


# ---------------------------------------------------------------------------
# Extra Features
# ---------------------------------------------------------------------------

def _axis_to_angles(axis: np.ndarray):
    """Gets Theta and Phi [deg] from PCA main axis."""
    ax, ay, az = axis
    theta = float(np.degrees(np.arccos(np.clip(abs(az), 0.0, 1.0))))
    phi   = float(np.degrees(np.arctan2(ay, ax)))
    return theta, phi


def _first_hit_top(x_pts, y_pts, z_pts, theta, phi, tol: float = 0.8) -> float:
    if len(z_pts) == 0:
        return 0.0
    dx = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    dy = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    dz = np.cos(np.radians(theta))
    if abs(dz) < 1e-6:
        return 0.0
    cx, cy, cz = float(x_pts.mean()), float(y_pts.mean()), float(z_pts.mean())
    entry_z = np.nan
    for zk in TRACKER_Z[::-1]:
        t = (zk - cz) / dz
        if abs(cx + t * dx) <= DETECTOR_XY_MAX and abs(cy + t * dy) <= DETECTOR_XY_MAX:
            entry_z = zk
            break
    if not np.isfinite(entry_z):
        return 0.0
    return float(np.any(np.abs(z_pts - entry_z) <= tol))


def _plane_coverage(z_pts: np.ndarray, tol: float = 0.8) -> float:
    if len(z_pts) == 0:
        return 0.0
    if len(z_pts) == 1:
        return 1.0
    hit = np.array([np.any(np.abs(z_pts - zk) <= tol) for zk in TRACKER_Z])
    idx = np.where(hit)[0]
    if len(idx) < 2:
        return float(len(idx))
    return float(hit[idx[0]:idx[-1] + 1].sum() / (idx[-1] - idx[0] + 1))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class VegaClassifier:
    """    
    Parameters
    ----------
    model_path : path to vega_model.skops
    """

    def __init__(self, model_path: str):
        # skops stores only a whitelisted set of sklearn/numpy types and
        # cannot execute arbitrary code on load (unlike pickle/joblib).
        untrusted = get_untrusted_types(file=model_path)
        self.clf = skops_load(model_path, trusted=untrusted)

        # The model was trained with n_jobs=-1, but here we score one event
        # at a time. With n_jobs>1, every single predict_proba() call pays
        # the cost of spinning up a joblib thread pool across all cores --
        # overhead that dwarfs the actual single-row prediction work.
        # Forcing n_jobs=1 removes that per-call overhead entirely.
        if hasattr(self.clf, "n_jobs"):
            self.clf.n_jobs = 1

    def score(self, coords: np.ndarray, raw_pca: tuple, cle_pca: tuple, e_tot: float) -> float:
        """
        Parameters
        ----------
        coords  : (N, 3)
        raw_pca : (R, eig, axis)          — output of make_pca
        cle_pca : (n, R, eig, axis)       — output of _clean_single
        """
        R_raw,  eig_raw,   axis_raw = raw_pca
        n_clean, R_clean, eig_clean, _ = cle_pca

        x_pts, y_pts, z_pts = coords[:, 0], coords[:, 1], coords[:, 2]
        theta, phi = _axis_to_angles(axis_raw)

        L1_r, L2_r, L3_r = eig_raw[0],   eig_raw[1],   eig_raw[2]
        L1_c, L2_c, L3_c = eig_clean[0], eig_clean[1], eig_clean[2]

        X = np.array([
            float(len(coords)), float(n_clean),
            R_raw,              R_clean,
            L1_r, L2_r, L3_r,
            L1_c, L2_c, L3_c,
            float(np.sqrt(L1_r**2 + L2_r**2)),
            float(np.sqrt(L1_c**2 + L2_c**2)),
            theta, phi,
            float(z_pts.max()),
            _first_hit_top(x_pts, y_pts, z_pts, theta, phi),
            _plane_coverage(z_pts),
            e_tot,
        ], dtype=np.float32).reshape(1, -1)

        X = np.where(np.isfinite(X), X, 0.0)
        return float(self.clf.predict_proba(X)[0, 1])
    

# ---------------------------------------------------------------------------
# ANALYTICAL
# ---------------------------------------------------------------------------

class SimpleClassifier:
    """
    Parameters
    ----------
    lookup_path : path to lookup_table.json
    """

    def __init__(self, lookup_path: str, thr: float = 0.99):
        self.thr = thr
        self.lookup_file = lookup_path + f'pca_svd_{self.thr}.json'
        self.lookup_table = json.load(open(self.lookup_file))["histograms"]

    def score(self, cle_pca: tuple, e_tot: float) -> float:
        n_hits = cle_pca[0]

        histo = None
        for h in self.lookup_table:
            if h["e_lo"] <= e_tot < h["e_hi"]:
                histo = h
                break
        if histo is None:
            return 0.0

        edges = np.array(histo["hit_bins"])
        sig   = np.array(histo["signal"],     dtype=float)
        bkg   = np.array(histo["background"], dtype=float)

        idx   = np.clip(np.searchsorted(edges, n_hits, side="right") - 1,
                        0, len(sig) - 1)
        total = sig[idx] + bkg[idx]
        return float(sig[idx] / total) if total > 0 else 0.0
    

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def analyze(hits, e_tot, rf, thr: float = 0.99):
    coords  = getCoord(hits)
    raw_pca = make_pca(coords)
    if raw_pca is None:
        return 0.0
    
    cle_pca = _clean_single(coords, thr)

    if isinstance(rf, VegaClassifier):
        return rf.score(coords, raw_pca, cle_pca, e_tot)
    if isinstance(rf, SimpleClassifier):
        return rf.score(cle_pca, e_tot)
    return 0.0