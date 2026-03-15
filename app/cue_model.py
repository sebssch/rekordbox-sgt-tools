"""
PWAV-basiertes Cue-Prediction-Modell (Random Forest Baseline).

Trainiert 4 separate Modelle (Hot A, Hot B, Hot C, Memory) auf den
PWAV-Waveform-Daten von ~5.000 Tracks. Gibt Wahrscheinlichkeits-Maps
zurueck, die vom DJ-Validator in grid-konforme Kandidaten umgewandelt werden.
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from app.waveform import N_BINS, build_feature_vector, read_pwav

log = logging.getLogger("autocue.cue_model")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "cue_model.pkl"
DATA_DIR = Path("data")

# Kanal-Namen
CHANNELS = ["Hot A", "Hot B", "Hot C", "Memory"]

# Schwellenwert fuer Binarisierung der Gauss-Labels → 0/1 Klassen
_LABEL_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(test_split: float = 0.2, n_estimators: int = 200):
    """
    Trainiert 4 Random-Forest-Modelle (je eines pro Cue-Kanal).

    Laedt data/training_X.npy und data/training_Y.npy.
    80/20 Train/Test Split. Speichert models/cue_model.pkl.
    """
    x_path = DATA_DIR / "training_X.npy"
    y_path = DATA_DIR / "training_Y.npy"

    if not x_path.exists() or not y_path.exists():
        print("FEHLER: Trainingsdaten nicht gefunden.")
        print("Zuerst ausfuehren: python -m app.waveform extract")
        return

    X = np.load(x_path)
    Y = np.load(y_path)
    print(f"Trainingsdaten geladen: X={X.shape}, Y={Y.shape}")

    # Y binarisieren: Gauss-Peaks → 0/1 Klassen
    Y_bin = (Y >= _LABEL_THRESHOLD).astype(np.int32)

    # Train/Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_bin, test_size=test_split, random_state=42
    )
    print(f"Split: {X_train.shape[0]} Train, {X_test.shape[0]} Test\n")

    models = {}

    for ch_idx, ch_name in enumerate(CHANNELS):
        print(f"--- {ch_name} (Kanal {ch_idx}) ---")

        # Labels fuer diesen Kanal: (N, N_BINS) → fuer jedes Sample pro Bin 0/1
        y_train_ch = Y_train[:, :, ch_idx]  # (N_train, 100)
        y_test_ch = Y_test[:, :, ch_idx]    # (N_test, 100)

        # Feature-Expansion: Fuer jedes Bin den Bin-Index als Feature hinzufuegen
        # → X_expanded = (N * N_BINS, 402 + 1)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        bin_indices = np.arange(N_BINS, dtype=np.float32) / N_BINS
        X_train_exp = _expand_features(X_train, bin_indices)
        X_test_exp = _expand_features(X_test, bin_indices)

        y_train_flat = y_train_ch.ravel()
        y_test_flat = y_test_ch.ravel()

        # Class Balance Info
        n_pos = y_train_flat.sum()
        n_neg = len(y_train_flat) - n_pos
        print(f"  Positiv: {n_pos} ({100*n_pos/len(y_train_flat):.1f}%)")
        print(f"  Negativ: {n_neg} ({100*n_neg/len(y_train_flat):.1f}%)")

        # Random Forest mit class_weight='balanced' gegen Class-Imbalance
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_exp, y_train_flat)

        # Evaluation
        y_pred = rf.predict(X_test_exp)
        f1 = f1_score(y_test_flat, y_pred, zero_division=0)
        print(f"  F1-Score: {f1:.3f}")

        models[ch_name] = rf
        print()

    # Speichern
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    print(f"Modell gespeichert: {MODEL_PATH}")
    print(f"Dateigroesse: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")


def _expand_features(X: np.ndarray, bin_indices: np.ndarray) -> np.ndarray:
    """
    Expandiert Features: Fuer jedes Sample wird der Bin-Index als
    zusaetzliches Feature hinzugefuegt.

    X:            (N, 402)
    bin_indices:  (N_BINS,)

    Returns:      (N * N_BINS, 403)
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Jedes Sample N_BINS mal wiederholen
    X_rep = np.repeat(X, N_BINS, axis=0)  # (N * N_BINS, 402)

    # Bin-Index-Spalte
    bin_col = np.tile(bin_indices, n_samples).reshape(-1, 1)  # (N * N_BINS, 1)

    return np.hstack([X_rep, bin_col]).astype(np.float32)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """Laedt das trainierte Modell. Wirft FileNotFoundError wenn nicht vorhanden."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Kein trainiertes Modell gefunden: {MODEL_PATH}\n"
            f"Zuerst ausfuehren: python -m app.cue_model train"
        )
    return joblib.load(MODEL_PATH)


def predict(pwav: np.ndarray, bpm: float, genre: str) -> np.ndarray:
    """
    Vorhersage fuer einen einzelnen Track.

    Args:
        pwav:  PWAV-Waveform, shape (400,), float32 [0,1]
        bpm:   Track-BPM
        genre: Genre-String

    Returns:
        prob_map: shape (N_BINS, 4) — Wahrscheinlichkeiten pro Bin und Cue-Typ
                  Kanaele: [Hot A, Hot B, Hot C, Memory]
    """
    models = load_model()

    feature = build_feature_vector(pwav, bpm, genre)  # (402,)
    bin_indices = np.arange(N_BINS, dtype=np.float32) / N_BINS

    # Feature fuer alle Bins expandieren
    X_single = feature.reshape(1, -1)
    X_exp = _expand_features(X_single, bin_indices)  # (N_BINS, 403)

    prob_map = np.zeros((N_BINS, 4), dtype=np.float32)

    for ch_idx, ch_name in enumerate(CHANNELS):
        rf = models[ch_name]
        # predict_proba → Wahrscheinlichkeit fuer Klasse 1 (= Cue vorhanden)
        probs = rf.predict_proba(X_exp)
        # Klasse 1 ist der positive Fall
        if probs.shape[1] == 2:
            prob_map[:, ch_idx] = probs[:, 1]
        else:
            prob_map[:, ch_idx] = probs[:, 0]

    return prob_map


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate():
    """Detaillierte Evaluation mit Tolerance-Accuracy."""
    x_path = DATA_DIR / "training_X.npy"
    y_path = DATA_DIR / "training_Y.npy"

    if not x_path.exists() or not y_path.exists():
        print("FEHLER: Trainingsdaten nicht gefunden.")
        return

    X = np.load(x_path)
    Y = np.load(y_path)
    Y_bin = (Y >= _LABEL_THRESHOLD).astype(np.int32)

    _, X_test, _, Y_test = train_test_split(
        X, Y_bin, test_size=0.2, random_state=42
    )

    models = load_model()
    bin_indices = np.arange(N_BINS, dtype=np.float32) / N_BINS

    print("=== Evaluation (Test-Set) ===\n")

    for ch_idx, ch_name in enumerate(CHANNELS):
        rf = models[ch_name]
        y_true_ch = Y_test[:, :, ch_idx]  # (N_test, 100)

        # Track-fuer-Track Evaluation
        hits_exact = 0
        hits_tol2 = 0    # ±2 Bins = ±2% der Tracklaenge
        hits_tol5 = 0    # ±5 Bins = ±5% der Tracklaenge
        total_cues = 0
        false_pos = 0

        for i in range(X_test.shape[0]):
            # Ground Truth: Bins mit Cue
            gt_bins = set(np.where(y_true_ch[i] == 1)[0])
            if not gt_bins:
                continue
            total_cues += len(gt_bins)

            # Prediction
            X_single = X_test[i].reshape(1, -1)
            X_exp = _expand_features(X_single, bin_indices)
            probs = rf.predict_proba(X_exp)
            pred_probs = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            pred_bins = set(np.where(pred_probs >= 0.5)[0])

            # Exact Match
            hits_exact += len(gt_bins & pred_bins)

            # Tolerance Matches
            for gt_bin in gt_bins:
                for tol, counter_name in [(2, 'hits_tol2'), (5, 'hits_tol5')]:
                    for pred_bin in pred_bins:
                        if abs(gt_bin - pred_bin) <= tol:
                            if counter_name == 'hits_tol2':
                                hits_tol2 += 1
                            else:
                                hits_tol5 += 1
                            break

            # False Positives
            for pred_bin in pred_bins:
                if not any(abs(pred_bin - gt) <= 2 for gt in gt_bins):
                    false_pos += 1

        if total_cues > 0:
            print(f"--- {ch_name} ---")
            print(f"  Cues im Test-Set:     {total_cues}")
            print(f"  Exact Match:          {hits_exact}/{total_cues} ({100*hits_exact/total_cues:.1f}%)")
            print(f"  ±2 Bins (~±8 Beats):  {hits_tol2}/{total_cues} ({100*hits_tol2/total_cues:.1f}%)")
            print(f"  ±5 Bins (~±16 Beats): {hits_tol5}/{total_cues} ({100*hits_tol5/total_cues:.1f}%)")
            print(f"  False Positives:      {false_pos}")
            print()


# ---------------------------------------------------------------------------
# CLI: predict fuer einen einzelnen Track (nach Name)
# ---------------------------------------------------------------------------

def predict_track_by_name(track_name: str):
    """Sucht Track in DB, fuehrt PWAV-Prediction aus, zeigt Ergebnis."""
    import pyrekordbox.config as _cfg
    from app import config as _autocue_cfg
    from app.beatgrid import _MASTER_DB_PATH, _DB_DIR

    _cfg.__config__["rekordbox7"] = {
        "db_path": _MASTER_DB_PATH,
        "db_dir": _DB_DIR,
        "app_dir": os.path.expanduser(
            "~/Library/Application Support/Pioneer/rekordbox6"
        ),
    }
    _cfg.__config__["rekordbox6"] = _cfg.__config__["rekordbox7"]
    _cfg.__config__["pioneer"] = {
        "install_dir": os.path.expanduser("~/Library/Pioneer"),
        "app_dir": "/Applications",
    }

    import logging as _log
    _log.disable(_log.WARNING)
    from pyrekordbox.db6.database import Rekordbox6Database
    db = Rekordbox6Database(path=_MASTER_DB_PATH)
    db.open()
    _log.disable(_log.NOTSET)

    # Track suchen
    content = None
    for c in db.get_content():
        if track_name.lower() in (c.Title or "").lower():
            content = c
            break

    if content is None:
        print(f"Track nicht gefunden: {track_name}")
        db.close()
        return

    print(f"Track: {content.Title}")
    print(f"BPM: {float(content.BPM or 0)/100:.0f}  Genre: {content.GenreName}")

    # PWAV lesen
    anlz_rel = content.AnalysisDataPath
    dat_path = os.path.join(_DB_DIR, "share", anlz_rel.lstrip("/"))
    pwav = read_pwav(dat_path)

    # Prediction
    bpm = float(content.BPM or 0) / 100.0
    genre = str(content.GenreName or "")
    duration_ms = int(content.Length or 0) * 1000  # Length ist in Sekunden → ms
    duration_s = duration_ms / 1000.0

    prob_map = predict(pwav, bpm, genre)

    print(f"\n=== Probability Map (Top Peaks) ===\n")
    for ch_idx, ch_name in enumerate(CHANNELS):
        probs = prob_map[:, ch_idx]
        # Top-3 Peaks
        peaks = []
        for bin_idx in range(N_BINS):
            p = probs[bin_idx]
            if p >= 0.3:
                time_s = bin_idx / N_BINS * duration_s
                time_str = f"{int(time_s//60)}:{time_s%60:05.2f}"
                peaks.append((p, time_str, bin_idx))

        peaks.sort(reverse=True)
        if peaks:
            print(f"  {ch_name}:")
            for p, t, b in peaks[:5]:
                bar = "█" * int(p * 20)
                print(f"    [{t}] P={p:.2f} {bar}")
        else:
            print(f"  {ch_name}: keine Peaks (P < 0.3)")
        print()

    # Vergleich mit manuellen Cues
    all_cues = [cue for cue in db.get_cue() if str(cue.ContentID) == str(content.ID)]
    if all_cues:
        print("=== Manuelle Cues (Ground Truth) ===\n")
        kind_map = {0: "Memory", 1: "Hot A", 2: "Hot B", 3: "Hot C"}
        for cue in sorted(all_cues, key=lambda c: c.InMsec):
            t = cue.InMsec / 1000.0
            kind = kind_map.get(cue.Kind, f"Kind={cue.Kind}")
            time_str = f"{int(t//60)}:{t%60:05.2f}"
            print(f"  [{time_str}] {kind}")

    db.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.cue_model <command>")
        print("  train              — Modell trainieren")
        print("  evaluate           — Modell evaluieren")
        print("  predict <name>     — Prediction fuer Track anzeigen")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "train":
        print("=== PWAV Cue-Model Training ===\n")
        train()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Usage: python -m app.cue_model predict <track_name>")
            sys.exit(1)
        predict_track_by_name(" ".join(sys.argv[2:]))
    else:
        print(f"Unbekannter Befehl: {cmd}")
        sys.exit(1)
