"""
v3 Track-Vektorisierung
Berechnet Feature-Vektoren fuer alle Tracks in der Rekordbox-DB.
Zwei Modi: 'db_only' (schnell, nur Metadaten) und 'full' (mit Audio-Energie).

Vektor-Format (22 Dimensionen):
  [0]     BPM (normalisiert)
  [1]     Dauer in Sekunden (normalisiert)
  [2]     Energie-Dichte (mittlere RMS, oder Cue-Dichte als Proxy)
  [3-18]  Cue-Histogram (16 Bins, relative Positionen)
  [19]    Cue-Dichte (Cues pro Minute)
  [20]    Anzahl Hot Cues
  [21]    Memory/Hot Cue Ratio
"""

import argparse
import contextlib
import os
import pickle
import time

import librosa
import numpy as np
from tqdm import tqdm

from app.beatgrid import open_db


@contextlib.contextmanager
def _suppress_c_stderr():
    """Unterdrückt C-Library stderr-Output (z.B. libmpg123 ID3-Warnungen)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)

# --- Pfade ---

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
VECTOR_CACHE = os.path.join(_DATA_DIR, "track_vectors.npz")
META_CACHE = os.path.join(_DATA_DIR, "track_meta.pkl")

VECTOR_DIM = 22
N_HISTOGRAM_BINS = 16


# --- Feature-Extraktion pro Track ---

def _cue_histogram(cues, duration_sec: float) -> np.ndarray:
    """
    Erzeugt ein 16-Bin-Histogram der Cue-Positionen.
    Jeder Bin zaehlt wie viele Cues in diesem Zeitabschnitt liegen.
    Positionen werden relativ zur Track-Dauer normalisiert.
    """
    hist = np.zeros(N_HISTOGRAM_BINS, dtype=np.float64)

    if duration_sec <= 0 or not cues:
        return hist

    for cue in cues:
        t_ms = cue.InMsec
        if t_ms is None or t_ms < 0:
            continue
        rel_pos = (t_ms / 1000.0) / duration_sec
        rel_pos = min(rel_pos, 0.9999)  # Sicherstellen: bleibt in [0, 1)
        bin_idx = int(rel_pos * N_HISTOGRAM_BINS)
        hist[bin_idx] += 1.0

    # Normalisiere: max = 1.0
    mx = hist.max()
    if mx > 0:
        hist /= mx

    return hist


def vectorize_from_db(content) -> np.ndarray:
    """
    Lightweight-Vektor nur aus DB-Daten (kein Audio noetig).
    Energie-Dichte wird durch Cue-Dichte approximiert.

    Args:
        content: DjmdContent-Objekt aus pyrekordbox

    Returns:
        np.ndarray der Laenge VECTOR_DIM (unnormalisiert)
    """
    vec = np.zeros(VECTOR_DIM, dtype=np.float64)

    # [0] BPM (raw, wird spaeter normalisiert)
    bpm = (content.BPM or 0) / 100.0
    vec[0] = bpm

    # [1] Dauer in Sekunden
    duration = content.Length or 0
    vec[1] = float(duration)

    # Cues sammeln
    cues = list(content.Cues) if content.Cues else []
    memory_cues = [c for c in cues if c.Kind == 0]
    hot_cues = [c for c in cues if c.Kind is not None and c.Kind > 0]

    n_total = len(cues)
    n_hot = len(hot_cues)
    n_memory = len(memory_cues)

    # [2] Energie-Proxy: Cue-Dichte (Cues pro Minute)
    if duration > 0:
        vec[2] = n_total / (duration / 60.0)
    else:
        vec[2] = 0.0

    # [3-18] Cue-Histogram (16 Bins)
    hist = _cue_histogram(cues, float(duration))
    vec[3:3 + N_HISTOGRAM_BINS] = hist

    # [19] Cue-Dichte (gleich wie [2], wird spaeter normalisiert)
    vec[19] = vec[2]

    # [20] Anzahl Hot Cues
    vec[20] = float(n_hot)

    # [21] Memory/Hot Ratio
    if n_total > 0:
        vec[21] = n_memory / n_total
    else:
        vec[21] = 0.0

    return vec


def compute_energy_density(audio_path: str) -> float:
    """
    Berechnet mittlere RMS-Energie eines Tracks.
    Schnelle Berechnung: ~2-3 Sekunden pro Track.

    Returns:
        float in [0, 1] (normalisierte RMS-Energie)
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio nicht gefunden: {audio_path}")

    with _suppress_c_stderr():
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=512)[0]

    # Mittlere RMS, normalisiert auf [0, 1]
    mean_rms = float(np.mean(rms))
    # RMS-Werte sind typischerweise 0.0 - 0.3, skaliere hoch
    return min(mean_rms * 3.0, 1.0)


def vectorize_full(content, audio_path: str) -> np.ndarray:
    """
    Vollstaendiger Vektor inkl. Audio-Energie.
    Wie vectorize_from_db(), aber Slot [2] ist echte RMS-Energie.
    """
    vec = vectorize_from_db(content)

    # Slot [2] ueberschreiben mit echter Energie
    try:
        vec[2] = compute_energy_density(audio_path)
    except (FileNotFoundError, Exception):
        pass  # Behalte Cue-Dichte-Proxy

    return vec


# --- Normalisierung ---

def _fit_scaler(vectors: np.ndarray) -> dict:
    """
    Berechnet MinMax-Scaler-Parameter fuer alle Dimensionen.
    Returns: {"min": np.ndarray, "max": np.ndarray}
    """
    v_min = vectors.min(axis=0)
    v_max = vectors.max(axis=0)

    # Vermeide Division durch 0
    diff = v_max - v_min
    diff[diff == 0] = 1.0

    return {"min": v_min, "max": v_max, "diff": diff}


def _apply_scaler(vector: np.ndarray, scaler: dict) -> np.ndarray:
    """Wendet MinMax-Normalisierung an → [0, 1]."""
    return (vector - scaler["min"]) / scaler["diff"]


def _normalize_matrix(vectors: np.ndarray, scaler: dict) -> np.ndarray:
    """Normalisiert gesamte Matrix."""
    return (vectors - scaler["min"]) / scaler["diff"]


# --- Vektor-DB Aufbau ---

def build_vector_db(mode: str = "db_only") -> dict:
    """
    Batch-Vektorisierung aller Tracks in der Rekordbox-DB.

    Args:
        mode: "db_only" (~30s, nur Metadaten) oder "full" (~2-3h, mit Audio)

    Returns:
        dict mit Statistiken: total, vectorized, skipped, errors
    """
    print(f"\n{'='*60}")
    print(f"  v3 Track-Vektorisierung [{mode.upper()}]")
    print(f"{'='*60}")

    start_time = time.time()
    db = open_db()

    # Alle Tracks laden
    all_content = db.get_content().all()
    total = len(all_content)
    print(f"  {total} Tracks in Datenbank")

    # Filter: nur Tracks mit Cues (User hat dort manuelle Arbeit investiert)
    tracks_with_cues = []
    for c in all_content:
        cues = list(c.Cues) if c.Cues else []
        if len(cues) > 0 and c.Length and c.Length > 0:
            tracks_with_cues.append(c)

    n_with_cues = len(tracks_with_cues)
    print(f"  Davon mit Cues: {n_with_cues}")
    print(f"  Ohne Cues (uebersprungen): {total - n_with_cues}")

    # Vektoren berechnen
    vectors = np.zeros((n_with_cues, VECTOR_DIM), dtype=np.float64)
    meta = []
    errors = 0

    print()
    for i, content in enumerate(tqdm(tracks_with_cues,
                                     desc="Vektorisiere", unit="track")):
        try:
            if mode == "full" and content.FolderPath:
                vec = vectorize_full(content, content.FolderPath)
            else:
                vec = vectorize_from_db(content)

            vectors[i] = vec
            meta.append({
                "content_id": content.ID,
                "title": content.Title or "Unbekannt",
                "artist": _get_artist_name(content),
                "bpm": (content.BPM or 0) / 100.0,
                "duration": content.Length or 0,
                "path": content.FolderPath or "",
                "n_cues": len(list(content.Cues)) if content.Cues else 0,
            })

        except Exception as e:
            errors += 1
            tqdm.write(f"  Fehler bei Track {content.ID}: {e}")
            meta.append({
                "content_id": content.ID,
                "title": content.Title or "Unbekannt",
                "artist": "",
                "bpm": 0.0,
                "duration": 0,
                "path": "",
                "n_cues": 0,
            })

    # Normalisierung
    scaler = _fit_scaler(vectors)
    vectors_norm = _normalize_matrix(vectors, scaler)

    # Speichern
    os.makedirs(_DATA_DIR, exist_ok=True)

    np.savez_compressed(
        VECTOR_CACHE,
        vectors_raw=vectors,
        vectors_norm=vectors_norm,
    )

    with open(META_CACHE, "wb") as f:
        pickle.dump({
            "meta": meta,
            "scaler": scaler,
            "mode": mode,
            "n_tracks": n_with_cues,
            "vector_dim": VECTOR_DIM,
        }, f)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Zusammenfassung")
    print(f"{'='*60}")
    print(f"  Gesamt in DB:     {total:5d}")
    print(f"  Vektorisiert:     {n_with_cues:5d}")
    print(f"  Fehler:           {errors:5d}")
    print(f"  Vektor-Dimension: {VECTOR_DIM}")
    print(f"  Dauer:            {elapsed:.1f}s")
    print(f"  Gespeichert:      {VECTOR_CACHE}")
    print()

    return {
        "total": total,
        "vectorized": n_with_cues,
        "skipped": total - n_with_cues,
        "errors": errors,
    }


def load_vector_db() -> tuple[np.ndarray, list[dict], dict]:
    """
    Laedt gespeicherte Vektoren + Metadaten.

    Returns:
        (vectors_norm, meta_list, scaler_dict)
    """
    if not os.path.isfile(VECTOR_CACHE) or not os.path.isfile(META_CACHE):
        raise FileNotFoundError(
            "Vektor-DB nicht gefunden. Zuerst build_vector_db() ausfuehren!"
        )

    data = np.load(VECTOR_CACHE)
    vectors_norm = data["vectors_norm"]

    with open(META_CACHE, "rb") as f:
        meta_data = pickle.load(f)

    return vectors_norm, meta_data["meta"], meta_data["scaler"]


def vectorize_single(content, audio_path: str | None = None,
                     scaler: dict | None = None) -> np.ndarray:
    """
    Vektorisiert einen einzelnen neuen Track.
    Wendet den gespeicherten Scaler an fuer Vergleichbarkeit.

    Args:
        content: DjmdContent-Objekt
        audio_path: Pfad zur Audio-Datei (optional, fuer Energie)
        scaler: Normalisierungs-Scaler (oder auto-load)

    Returns:
        Normalisierter Feature-Vektor (1D numpy array)
    """
    if audio_path and os.path.isfile(audio_path):
        vec = vectorize_full(content, audio_path)
    else:
        vec = vectorize_from_db(content)

    # Scaler laden falls nicht uebergeben
    if scaler is None:
        _, _, scaler = load_vector_db()

    return _apply_scaler(vec, scaler)


# --- Hilfsfunktionen ---

def _get_artist_name(content) -> str:
    """Holt den Artist-Namen aus der Relation (wenn vorhanden)."""
    try:
        if content.Artist:
            return content.Artist.Name or ""
    except Exception:
        pass
    return ""


# --- CLI Entrypoint ---

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.vectorize",
        description=(
            "AutoCue Bibliotheks-Vektorisierung\n"
            "Berechnet Feature-Vektoren fuer alle Tracks in der Rekordbox-DB.\n"
            "Speichert Ergebnis in v3/data/track_vectors.npz + track_meta.pkl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m app.vectorize               # Schnelle Vektorisierung (nur DB-Metadaten, ~30s)
  python -m app.vectorize --mode full   # Vollstaendig mit Audio-Energie (~2-3h)
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["db_only", "full"],
        default="db_only",
        help=(
            "db_only = nur Metadaten aus Rekordbox-DB, sehr schnell (Standard). "
            "full = zusaetzlich echte Audio-Energie per librosa (~2-3h)."
        ),
    )
    return parser


if __name__ == "__main__":
    import logging as _logging
    _logging.getLogger("pyrekordbox").setLevel(_logging.ERROR)

    args = _build_parser().parse_args()
    result = build_vector_db(mode=args.mode)
    print(f"Fertig. {result['vectorized']} Tracks vektorisiert, "
          f"{result['errors']} Fehler.")
