"""
ML Cue Predictor — LightGBM-basierte Hot Cue Vorhersage

Laedt trainierte Modelle und produziert Kandidaten fuer den Quad-Check.
Ersetzt den PWAV Random Forest (cue_model.py + dj_validator.py).

Kandidaten-Format kompatibel mit pwav_candidates dict:
  {"hot_a": [t1, ...], "hot_c": [t2, ...], "memory": [t3, ...]}
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("autocue.ml_predictor")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODEL_A_PATH = os.path.join(MODEL_DIR, "ml_hot_a.lgb")
_MODEL_C_PATH = os.path.join(MODEL_DIR, "ml_hot_c.lgb")

_model_a = None
_model_c = None
_mem_models: list = []  # Memory Cue Modelle (Slot 2-6)
_mem_models_loaded = False

N_MEM_MODELS = 5  # Slot 2 bis 6


def _load_models():
    """Laedt LightGBM Modelle (lazy, einmal pro Session)."""
    global _model_a, _model_c, _mem_models, _mem_models_loaded

    if _model_a is not None and _model_c is not None:
        return True

    if not os.path.exists(_MODEL_A_PATH) or not os.path.exists(_MODEL_C_PATH):
        log.info("ML-Modelle nicht gefunden: %s", MODEL_DIR)
        return False

    try:
        import lightgbm as lgb
        _model_a = lgb.Booster(model_file=_MODEL_A_PATH)
        _model_c = lgb.Booster(model_file=_MODEL_C_PATH)
        log.info("ML-Modelle geladen: Hot A + Hot C")

        # Memory Cue Modelle laden (optional)
        if not _mem_models_loaded:
            _mem_models = []
            for i in range(N_MEM_MODELS):
                mem_path = os.path.join(MODEL_DIR, f"ml_mem_{i+2}.lgb")
                if os.path.exists(mem_path):
                    _mem_models.append(lgb.Booster(model_file=mem_path))
                else:
                    _mem_models.append(None)
            n_loaded = sum(1 for m in _mem_models if m is not None)
            if n_loaded > 0:
                log.info("ML Memory-Modelle geladen: %d/%d", n_loaded, N_MEM_MODELS)
            _mem_models_loaded = True

        return True
    except Exception as e:
        log.warning("ML-Modelle laden fehlgeschlagen: %s", e)
        _model_a = None
        _model_c = None
        return False


def predict_cue_positions(content, dat_path: str, ext_path: str,
                          grid, db, mik_data=None) -> dict | None:
    """
    Vorhersage der Hot Cue Positionen via LightGBM.

    Args:
        content:  DjmdContent-Objekt
        dat_path: Pfad zur .DAT ANLZ-Datei
        ext_path: Pfad zur .EXT ANLZ-Datei
        grid:     BeatGrid-Objekt
        db:       Rekordbox6Database
        mik_data: Optional vorberechnete MIK-Daten

    Returns:
        Dict {"hot_a": [t1, ...], "hot_c": [t2, ...], "memory": []}
        oder None wenn Modell nicht verfuegbar.
    """
    if not _load_models():
        return None

    duration = content.Length or 0
    if duration <= 0:
        return None

    # Feature-Extraktion (identisch zu export_training_data.py)
    try:
        features = _extract_features(content, dat_path, ext_path, db, mik_data)
    except Exception as e:
        log.warning("Feature-Extraktion fehlgeschlagen: %s", e)
        return None

    # Prediction: relative Positionen (0.0 - 1.0)
    X = features.reshape(1, -1)
    pred_a_rel = float(_model_a.predict(X)[0])
    pred_c_rel = float(_model_c.predict(X)[0])

    # Clamp auf [0.0, 1.0]
    pred_a_rel = max(0.0, min(1.0, pred_a_rel))
    pred_c_rel = max(0.0, min(1.0, pred_c_rel))

    # Relative Position → absolute Zeit
    hot_a_sec = pred_a_rel * duration
    hot_c_sec = pred_c_rel * duration

    # Snap auf naechsten Downbeat
    from app.beatgrid import snap_to_downbeat
    hot_a_sec = snap_to_downbeat(hot_a_sec, grid)
    hot_c_sec = snap_to_downbeat(hot_c_sec, grid)

    # Memory Cue Predictions (Slot 2-6)
    memory_times: list[float] = []
    if _mem_models:
        for model in _mem_models:
            if model is None:
                continue
            pred_rel = float(model.predict(X)[0])
            pred_rel = max(0.0, min(1.0, pred_rel))
            if pred_rel > 0.01:  # Nur gueltige Positionen
                mem_sec = snap_to_downbeat(pred_rel * duration, grid)
                memory_times.append(mem_sec)

    return {
        "hot_a": [hot_a_sec],
        "hot_c": [hot_c_sec],
        "memory": memory_times,
    }


def _extract_features(content, dat_path: str, ext_path: str,
                       db, mik_data=None) -> np.ndarray:
    """Feature-Extraktion identisch zu export_training_data.extract_features_fast()."""
    from app.waveform import read_pwav
    from app.phrase_reader import read_phrases
    from app.beatgrid import read_beatgrid
    from app.vectorize import vectorize_from_db, load_vector_db, _key_to_camelot
    from app.cbr import find_twins, analyze_hot_cue_pattern
    from app.mik_scraper import get_mik_data as _get_mik

    PWAV_DIM = 400
    METADATA_DIM = 4
    MIK_DIM = 9
    PHRASE_DIM = 41
    CBR_DIM = 13
    EXTRA_DIM = 2
    TOTAL = PWAV_DIM + METADATA_DIM + MIK_DIM + PHRASE_DIM + CBR_DIM + EXTRA_DIM

    features = np.zeros(TOTAL, dtype=np.float32)
    offset = 0
    duration = content.Length or 0

    # --- PWAV (400) ---
    try:
        pwav = read_pwav(dat_path)
        if pwav is not None and len(pwav) == PWAV_DIM:
            features[offset:offset + PWAV_DIM] = pwav
    except Exception:
        pass
    offset += PWAV_DIM

    # --- Metadata (4) ---
    bpm = (content.BPM or 0) / 100.0
    features[offset] = bpm / 200.0
    features[offset + 1] = duration / 600.0
    features[offset + 2] = _genre_hash(getattr(content, 'GenreName', '') or '')
    features[offset + 3] = _key_to_camelot(getattr(content, 'KeyName', '') or '')
    offset += METADATA_DIM

    # --- MIK Cues (9) ---
    try:
        if mik_data and mik_data.cue_times and duration > 0:
            for i, t in enumerate(sorted(mik_data.cue_times)[:8]):
                features[offset + i] = t / duration
            features[offset + 8] = len(mik_data.cue_times) / 10.0
        else:
            artist = content.Artist.Name if content.Artist else ""
            title = content.Title or ""
            folder = content.FolderPath or ""
            mik = _get_mik(folder, artist, title)
            if mik and mik.cue_times and duration > 0:
                for i, t in enumerate(sorted(mik.cue_times)[:8]):
                    features[offset + i] = t / duration
                features[offset + 8] = len(mik.cue_times) / 10.0
    except Exception:
        pass
    offset += MIK_DIM

    # --- Phrases (41) ---
    try:
        grid = read_beatgrid(dat_path)
        if os.path.exists(ext_path):
            phrases = read_phrases(ext_path, grid)
            if phrases and duration > 0:
                for i, p in enumerate(phrases[:20]):
                    features[offset + i] = p.kind / 6.0
                    features[offset + 20 + i] = p.time_start_sec / duration
                features[offset + 40] = len(phrases) / 20.0

        features[TOTAL - 2] = grid.count / 2000.0
        features[TOTAL - 1] = (grid.bar_count if hasattr(grid, 'bar_count') else grid.count // 4) / 500.0
    except Exception:
        pass
    offset += PHRASE_DIM

    # --- CBR Twins (13) ---
    try:
        vec = vectorize_from_db(content)
        if vec is not None:
            genre = (getattr(content, 'GenreName', '') or '').lower()
            twins = find_twins(vec, n=5, genre=genre)
            twins = [t for t in twins if str(t.get("content_id", "")) != str(content.ID)][:5]

            if twins:
                twin_ids = [str(t["content_id"]) for t in twins]
                pattern = analyze_hot_cue_pattern(db, twin_ids)

                twin_a_positions = []
                twin_c_positions = []
                for tid in twin_ids:
                    tc = db.get_content(ID=tid)
                    if tc and tc.Cues and tc.Length and tc.Length > 0:
                        for cue in tc.Cues:
                            if cue.Kind == 1 and cue.InMsec:
                                twin_a_positions.append(cue.InMsec / (tc.Length * 1000.0))
                            elif cue.Kind == 3 and cue.InMsec:
                                twin_c_positions.append(cue.InMsec / (tc.Length * 1000.0))

                for i, pos in enumerate(twin_a_positions[:5]):
                    features[offset + i] = pos
                for i, pos in enumerate(twin_c_positions[:5]):
                    features[offset + 5 + i] = pos

                rel_a = pattern.get("hot_a_relative_pos", 0)
                rel_c = pattern.get("hot_c_relative_pos", 0)
                features[offset + 10] = rel_a if rel_a else 0.0
                features[offset + 11] = rel_c if rel_c else 0.0
                features[offset + 12] = pattern.get("cue_spacing", 32) / 64.0
    except Exception:
        pass

    return features


def _genre_hash(genre: str) -> float:
    """Deterministischer Genre-Hash normalisiert auf [0, 1]."""
    if not genre:
        return 0.0
    h = hash(genre.strip().lower()) % 10000
    return h / 10000.0
