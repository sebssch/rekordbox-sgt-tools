"""
Spektral-Feature-Extraktion fuer ML Cue Prediction

Drei Varianten:
  custom  — Eigene librosa-basierte Features (224 dims)
  openl3  — Vortrainierte torchopenl3 CNN-Embeddings (512 dims)
  auto    — Beide kombiniert (736 dims)

Konfiguration via config.yaml:
  spectral_mode: "custom" | "openl3" | "auto" | "off"
"""
from __future__ import annotations

import hashlib
import logging
import os

import numpy as np

log = logging.getLogger("autocue.spectral")

# --- Dimensionen ---
CUSTOM_DIM = 224      # 7 Features × 32 Segmente
OPENL3_DIM = 512      # torchopenl3 Embedding-Groesse
N_SEGMENTS = 32       # Track wird in 32 gleiche Segmente geteilt
N_CUSTOM_FEATURES = 7 # Features pro Segment

# --- Frequenz-Band-Grenzen (Hz) fuer Mel-Bins bei sr=22050, n_mels=128 ---
# Mel-Bin-Indices (approximiert):
#   Low  (20-300 Hz)   ≈ Bins 0-15
#   Mid  (300-4000 Hz) ≈ Bins 16-80
#   High (4000+ Hz)    ≈ Bins 81-127
_LOW_BINS = slice(0, 16)
_MID_BINS = slice(16, 81)
_HIGH_BINS = slice(81, 128)


def get_spectral_dim(mode: str) -> int:
    """Gibt die Feature-Dimension fuer den jeweiligen Spectral-Mode zurueck."""
    if mode == "custom":
        return CUSTOM_DIM
    elif mode == "openl3":
        return OPENL3_DIM
    elif mode == "auto":
        return CUSTOM_DIM + OPENL3_DIM
    return 0  # "off"


def _cache_path(audio_path: str, mode: str, cache_dir: str) -> str:
    """Generiert den Cache-Dateipfad basierend auf Audio-Pfad und Mode."""
    path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{path_hash}_{mode}.npy")


def _load_from_cache(audio_path: str, mode: str, cache_dir: str) -> np.ndarray | None:
    """Laedt gecachte Spektral-Features falls vorhanden."""
    if not cache_dir:
        return None
    cp = _cache_path(audio_path, mode, cache_dir)
    if os.path.exists(cp):
        try:
            arr = np.load(cp)
            expected = get_spectral_dim(mode)
            if arr.shape == (expected,):
                return arr
        except Exception:
            pass
    return None


def _save_to_cache(features: np.ndarray, audio_path: str, mode: str,
                   cache_dir: str) -> None:
    """Speichert Spektral-Features in Cache."""
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    cp = _cache_path(audio_path, mode, cache_dir)
    try:
        np.save(cp, features)
    except Exception as e:
        log.debug("Cache-Speichern fehlgeschlagen: %s", e)


def extract_custom_spectral(audio_path: str, duration: float) -> np.ndarray:
    """
    Eigene Spektral-Features via librosa.

    Extrahiert 7 Features × 32 Segmente = 224 Dimensionen:
      [0-31]    Low-Band Energie (20-300 Hz) — Bass/Kick
      [32-63]   Mid-Band Energie (300-4000 Hz) — Melodie/Synth
      [64-95]   High-Band Energie (4000+ Hz) — Hi-Hats/Riser
      [96-127]  Spectral Novelty (Frame-Delta) — Uebergaenge
      [128-159] Onset Strength — Rhythmische Dichte
      [160-191] Spectral Contrast — Tonal vs. Perkussiv
      [192-223] Spectral Flatness — Noise vs. Harmonic

    Returns: ndarray shape (224,) float32
    """
    import librosa

    features = np.zeros(CUSTOM_DIM, dtype=np.float32)

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        log.warning("Audio laden fehlgeschlagen: %s — %s", audio_path, e)
        return features

    if len(y) == 0 or duration <= 0:
        return features

    # --- Mel-Spektrogramm ---
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)  # shape: (128, T)
    n_frames = S_db.shape[1]

    if n_frames < N_SEGMENTS:
        return features

    # Frames pro Segment
    seg_size = n_frames // N_SEGMENTS

    # --- Band-Energien (3 × 32) ---
    for seg_idx in range(N_SEGMENTS):
        start = seg_idx * seg_size
        end = start + seg_size
        segment = S_db[:, start:end]

        # Low/Mid/High Band Mittelwerte (normalisiert von dB auf 0-1)
        features[seg_idx] = (np.mean(segment[_LOW_BINS, :]) + 80.0) / 80.0
        features[N_SEGMENTS + seg_idx] = (np.mean(segment[_MID_BINS, :]) + 80.0) / 80.0
        features[2 * N_SEGMENTS + seg_idx] = (np.mean(segment[_HIGH_BINS, :]) + 80.0) / 80.0

    # --- Spectral Novelty (32) ---
    # Differenz zwischen aufeinanderfolgenden Frames, gemittelt pro Segment
    S_diff = np.diff(S_db, axis=1)
    S_diff_energy = np.sqrt(np.mean(S_diff ** 2, axis=0))  # RMS der Aenderung pro Frame
    if len(S_diff_energy) > 0:
        # Normalisieren
        max_diff = np.max(S_diff_energy) if np.max(S_diff_energy) > 0 else 1.0
        S_diff_energy = S_diff_energy / max_diff
        seg_size_diff = len(S_diff_energy) // N_SEGMENTS
        for seg_idx in range(N_SEGMENTS):
            start = seg_idx * seg_size_diff
            end = start + seg_size_diff
            if start < len(S_diff_energy):
                features[3 * N_SEGMENTS + seg_idx] = np.mean(
                    S_diff_energy[start:min(end, len(S_diff_energy))]
                )

    # --- Onset Strength (32) ---
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        if len(onset_env) > 0:
            max_onset = np.max(onset_env) if np.max(onset_env) > 0 else 1.0
            onset_norm = onset_env / max_onset
            seg_size_onset = len(onset_norm) // N_SEGMENTS
            for seg_idx in range(N_SEGMENTS):
                start = seg_idx * seg_size_onset
                end = start + seg_size_onset
                if start < len(onset_norm):
                    features[4 * N_SEGMENTS + seg_idx] = np.mean(
                        onset_norm[start:min(end, len(onset_norm))]
                    )
    except Exception:
        pass

    # --- Spectral Contrast (32) ---
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        # contrast shape: (7, T) — mittlere Band-Contrast-Werte
        contrast_mean = np.mean(contrast, axis=0)  # (T,)
        if len(contrast_mean) > 0:
            # Normalisieren auf [0, 1]
            c_min, c_max = np.min(contrast_mean), np.max(contrast_mean)
            if c_max > c_min:
                contrast_norm = (contrast_mean - c_min) / (c_max - c_min)
            else:
                contrast_norm = np.zeros_like(contrast_mean)
            seg_size_c = len(contrast_norm) // N_SEGMENTS
            for seg_idx in range(N_SEGMENTS):
                start = seg_idx * seg_size_c
                end = start + seg_size_c
                if start < len(contrast_norm):
                    features[5 * N_SEGMENTS + seg_idx] = np.mean(
                        contrast_norm[start:min(end, len(contrast_norm))]
                    )
    except Exception:
        pass

    # --- Spectral Flatness (32) ---
    try:
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)
        flatness = flatness[0]  # shape: (T,)
        if len(flatness) > 0:
            seg_size_f = len(flatness) // N_SEGMENTS
            for seg_idx in range(N_SEGMENTS):
                start = seg_idx * seg_size_f
                end = start + seg_size_f
                if start < len(flatness):
                    features[6 * N_SEGMENTS + seg_idx] = np.mean(
                        flatness[start:min(end, len(flatness))]
                    )
    except Exception:
        pass

    # Clamp auf [0, 1]
    features = np.clip(features, 0.0, 1.0)

    return features


def extract_openl3_embedding(audio_path: str) -> np.ndarray:
    """
    Vortrainierte Audio-Embeddings via torchopenl3 (PyTorch).

    Returns: ndarray shape (512,) float32 — zeitliches Mittel aller Frames.
    """
    features = np.zeros(OPENL3_DIM, dtype=np.float32)

    try:
        import torchopenl3
        import soundfile as sf
    except ImportError:
        log.warning("torchopenl3 nicht installiert. "
                    "Installieren: pip install torchopenl3")
        return features

    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        # Mono konvertieren falls Stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        emb, ts = torchopenl3.get_audio_embedding(
            audio, sr,
            input_repr="mel128",
            content_type="music",
            embedding_size=512,
            hop_size=0.5,       # 0.5s Hop → weniger Frames, schneller
            batch_size=32,
            verbose=False,
        )

        if emb is not None and len(emb) > 0:
            # Zeitliches Mittel ueber alle Frames
            mean_emb = np.mean(emb, axis=0).astype(np.float32)
            # Normalisieren auf [0, 1] via Min-Max
            e_min, e_max = np.min(mean_emb), np.max(mean_emb)
            if e_max > e_min:
                features = (mean_emb - e_min) / (e_max - e_min)
            else:
                features = mean_emb

    except Exception as e:
        log.warning("OpenL3 Embedding fehlgeschlagen: %s — %s", audio_path, e)

    return features


def extract_spectral_features(audio_path: str, duration: float,
                               mode: str = "auto",
                               cache_dir: str = "") -> np.ndarray:
    """
    Dispatcher: Extrahiert Spektral-Features je nach Mode.

    Args:
        audio_path: Pfad zur Audio-Datei
        duration:   Track-Laenge in Sekunden
        mode:       "custom" | "openl3" | "auto" | "off"
        cache_dir:  Verzeichnis fuer Feature-Cache (leer = kein Cache)

    Returns:
        ndarray — Groesse abhaengig vom Mode (224/512/736/0)
    """
    dim = get_spectral_dim(mode)
    if dim == 0 or not audio_path or not os.path.exists(audio_path):
        return np.zeros(dim, dtype=np.float32)

    # Cache pruefen
    cached = _load_from_cache(audio_path, mode, cache_dir)
    if cached is not None:
        return cached

    if mode == "custom":
        features = extract_custom_spectral(audio_path, duration)
    elif mode == "openl3":
        features = extract_openl3_embedding(audio_path)
    elif mode == "auto":
        custom = extract_custom_spectral(audio_path, duration)
        openl3 = extract_openl3_embedding(audio_path)
        features = np.concatenate([custom, openl3])
    else:
        features = np.zeros(dim, dtype=np.float32)

    # Cache speichern
    _save_to_cache(features, audio_path, mode, cache_dir)

    return features
