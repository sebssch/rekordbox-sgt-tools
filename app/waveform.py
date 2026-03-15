"""
PWAV-Waveform aus Rekordbox ANLZ .DAT Dateien lesen
und ML-Trainingsdaten fuer das Cue-Prediction-Modell extrahieren.

PWAV-Format: 400 x uint8, untere 5 Bits = Wellenformhoehe (0-31),
obere 3 Bits = Farbinformation (0-7).
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pyrekordbox.anlz import AnlzFile

log = logging.getLogger("autocue.waveform")

# --- Konstanten ---
N_BINS = 100        # Tracklänge wird in 100 Zeitfenster (1 % Aufloesung) aufgeteilt
PWAV_LEN = 400      # Standard PWAV-Laenge (Rekordbox Overview Waveform)
GAUSS_SIGMA = 2.0   # Gauss-Smoothing Sigma fuer Label-Encoding (in Bins)

# Cue-Kind → Kanal-Index in der Label-Map
_KIND_TO_CHANNEL = {
    1: 0,  # Hot A
    2: 1,  # Hot B
    3: 2,  # Hot C
    0: 3,  # Memory
}


# ---------------------------------------------------------------------------
# PWAV lesen
# ---------------------------------------------------------------------------

def read_pwav(anlz_dat_path: str) -> np.ndarray:
    """
    Liest den PWAV-Tag aus einer Rekordbox ANLZ .DAT Datei.

    Returns:
        ndarray shape (400,) float32 normalisiert auf [0, 1].
        Falls PWAV kuerzer oder laenger als 400 → wird auf 400 interpoliert.
    """
    if not os.path.exists(anlz_dat_path):
        raise FileNotFoundError(f"ANLZ-Datei nicht gefunden: {anlz_dat_path}")

    anlz = AnlzFile.parse_file(anlz_dat_path)
    tag = anlz.get_tag("PWAV")
    if tag is None:
        raise ValueError(f"Kein PWAV-Tag in: {anlz_dat_path}")

    wf, _ = tag.get()  # wf = int8 array (0-31)

    # Normalisieren auf [0, 1]
    wf = wf.astype(np.float32) / 31.0

    # Auf einheitliche Laenge bringen (fuer Modell-Input)
    if len(wf) != PWAV_LEN:
        x_old = np.linspace(0, 1, len(wf))
        x_new = np.linspace(0, 1, PWAV_LEN)
        wf = np.interp(x_new, x_old, wf).astype(np.float32)

    return wf


# ---------------------------------------------------------------------------
# Feature-Vektor bauen
# ---------------------------------------------------------------------------

def build_feature_vector(pwav: np.ndarray, bpm: float, genre: str) -> np.ndarray:
    """
    Kombiniert PWAV-Waveform mit Track-Metadaten zu einem Feature-Vektor.

    Returns:
        ndarray shape (402,) float32 — [PWAV(400), BPM_norm, Genre_hash]
    """
    bpm_norm = np.float32(min(bpm, 200.0) / 200.0)
    # Deterministisches Genre-Encoding (0-1 via Hash)
    genre_clean = (genre or "").strip().lower()[:20]
    genre_val = np.float32(abs(hash(genre_clean)) % 10000 / 10000.0)
    return np.concatenate([pwav, [bpm_norm, genre_val]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Label-Map bauen (Gauss-Smoothed Probability Targets)
# ---------------------------------------------------------------------------

def _cue_to_bin(cue_ms: int, duration_ms: int) -> int:
    """Cue-Position in Millisekunden → Bin-Index (0 bis N_BINS-1)."""
    if duration_ms <= 0:
        return 0
    return min(N_BINS - 1, max(0, int(cue_ms / duration_ms * N_BINS)))


def build_label_map(cues: list, duration_ms: int) -> np.ndarray:
    """
    Erzeugt eine Label-Map (N_BINS, 4) aus einer Liste von DjmdCue-Eintraegen.

    Kanaele: [Hot A, Hot B, Hot C, Memory]
    Jeder Cue erzeugt einen Gauss-Peak (sigma=GAUSS_SIGMA) um seine Bin-Position.
    Werte werden auf [0, 1] normalisiert (Peak = 1.0).

    Args:
        cues: Liste von Objekten mit .InMsec (int) und .Kind (int)
        duration_ms: Tracklaenge in Millisekunden
    """
    labels = np.zeros((N_BINS, 4), dtype=np.float32)

    for cue in cues:
        kind = int(cue.Kind)
        channel = _KIND_TO_CHANNEL.get(kind)
        if channel is None:
            continue  # Unbekannter Kind (4-8 etc.) → ignorieren

        bin_idx = _cue_to_bin(int(cue.InMsec), duration_ms)

        # Gauss-Impuls an der Cue-Position
        impulse = np.zeros(N_BINS, dtype=np.float32)
        impulse[bin_idx] = 1.0
        smoothed = gaussian_filter1d(impulse, sigma=GAUSS_SIGMA)

        # Normalisieren (Peak = 1.0) und addieren
        peak = smoothed.max()
        if peak > 0:
            labels[:, channel] += smoothed / peak

    # Clippen auf [0, 1]
    np.clip(labels, 0.0, 1.0, out=labels)
    return labels


# ---------------------------------------------------------------------------
# Trainingsdaten-Extraktion
# ---------------------------------------------------------------------------

def extract_training_data(output_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    """
    Extrahiert X (Features) und Y (Labels) aus allen Tracks mit Cues in der
    Rekordbox-Datenbank.

    Speichert:
        {output_dir}/training_X.npy — Shape (N, 402)
        {output_dir}/training_Y.npy — Shape (N, N_BINS, 4)

    Returns:
        (X, Y) Tuple der numpy-Arrays.
    """
    import pyrekordbox.config as _cfg
    from app import config as _autocue_cfg
    from app.beatgrid import _MASTER_DB_PATH, _DB_DIR

    # pyrekordbox Config
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

    # Alle Cues nach ContentID gruppieren
    print("  Lade Cue-Daten aus Datenbank...")
    all_cues = list(db.get_cue())
    cues_by_content: dict[str, list] = {}
    for cue in all_cues:
        cid = str(cue.ContentID)
        if cid not in cues_by_content:
            cues_by_content[cid] = []
        cues_by_content[cid].append(cue)

    print(f"  {len(cues_by_content)} Tracks mit Cues gefunden")

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    skipped = 0

    contents = list(db.get_content())
    total = len(contents)

    for i, content in enumerate(contents):
        cid = str(content.ID)
        if cid not in cues_by_content:
            continue

        # ANLZ-Pfad
        anlz_rel = content.AnalysisDataPath
        if not anlz_rel:
            skipped += 1
            continue

        dat_path = os.path.join(_DB_DIR, "share", anlz_rel.lstrip("/"))
        if not os.path.exists(dat_path):
            skipped += 1
            continue

        # PWAV lesen
        try:
            pwav = read_pwav(dat_path)
        except (ValueError, FileNotFoundError, Exception):
            skipped += 1
            continue

        # Metadaten
        bpm = float(content.BPM or 0) / 100.0
        genre = str(content.GenreName or "")
        duration_ms = int(content.Length or 0) * 1000  # Length ist in Sekunden → ms

        if bpm <= 0 or duration_ms <= 0:
            skipped += 1
            continue

        # Feature-Vektor
        feature = build_feature_vector(pwav, bpm, genre)

        # Label-Map
        track_cues = cues_by_content[cid]
        label_map = build_label_map(track_cues, duration_ms)

        X_list.append(feature)
        Y_list.append(label_map)

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{total}] {len(X_list)} Tracks extrahiert...")

    db.close()

    if not X_list:
        print("  FEHLER: Keine Trainingsdaten extrahiert!")
        return np.array([]), np.array([])

    X = np.stack(X_list)
    Y = np.stack(Y_list)

    # Speichern
    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, "training_X.npy")
    y_path = os.path.join(output_dir, "training_Y.npy")
    np.save(x_path, X)
    np.save(y_path, Y)

    print(f"\n  Ergebnis:")
    print(f"    X: {X.shape} → {x_path}")
    print(f"    Y: {Y.shape} → {y_path}")
    print(f"    Uebersprungen: {skipped}")

    return X, Y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.waveform <command>")
        print("  extract   — Trainingsdaten aus allen Tracks extrahieren")
        print("  info FILE — PWAV-Info fuer eine ANLZ-Datei anzeigen")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "extract":
        print("=== PWAV Trainingsdaten-Extraktion ===\n")
        X, Y = extract_training_data()
        if X.size > 0:
            print(f"\n  Fertig! {X.shape[0]} Tracks bereit zum Training.")
    elif cmd == "info":
        if len(sys.argv) < 3:
            print("Usage: python -m app.waveform info <ANLZ_DAT_PATH>")
            sys.exit(1)
        path = sys.argv[2]
        wf = read_pwav(path)
        print(f"PWAV Shape: {wf.shape}")
        print(f"Min: {wf.min():.3f}  Max: {wf.max():.3f}  Mean: {wf.mean():.3f}")
        print(f"Erste 20 Werte: {wf[:20]}")
    else:
        print(f"Unbekannter Befehl: {cmd}")
        sys.exit(1)
