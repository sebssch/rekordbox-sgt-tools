"""
ML Training Data Export — Supervised Learning fuer Hot Cue Prediction

Extrahiert Features + Labels aus allen Tracks mit manuellen Hot A + Hot C Cues.
Speichert als NumPy Arrays fuer LightGBM Training.

Nutzung:
    python tools/export_training_data.py [--mode fast|full]

Modi:
    fast  — PWAV + Metadata + MIK + Phrases + CBR Twins (~1 Min)
    full  — Zusaetzlich akustische Features aus librosa (~3h)
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

# Projekt-Root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.beatgrid import open_db, read_beatgrid
from app.waveform import read_pwav
from app.phrase_reader import read_phrases
from app.mik_scraper import get_mik_data
from app.vectorize import vectorize_from_db, load_vector_db, _key_to_camelot
from app.cbr import find_twins, analyze_hot_cue_pattern
from app.spectral import extract_spectral_features, get_spectral_dim
from app import config as _cfg

# --- Feature-Dimensionen ---
PWAV_DIM         = 400
METADATA_DIM     = 4    # BPM, Duration, Genre-Hash, Key
MIK_DIM          = 9    # 8 relative Positionen + Count
PHRASE_DIM       = 41   # 20 Kinds + 20 Rel-Starts + Count
CBR_DIM          = 13   # 5 Twin Hot A + 5 Twin Hot C + Median A + Median C + Spacing
EXTRA_DIM        = 2    # Beat Count, Bar Count

FEATURE_DIM_BASE = PWAV_DIM + METADATA_DIM + MIK_DIM + PHRASE_DIM + CBR_DIM + EXTRA_DIM  # 469
FEATURE_DIM_FAST = FEATURE_DIM_BASE  # Wird dynamisch erweitert via get_feature_dim()


def get_feature_dim(spectral_mode: str = "off") -> int:
    """Gibt die gesamte Feature-Dimension inkl. optionaler Spektral-Features zurueck."""
    return FEATURE_DIM_BASE + get_spectral_dim(spectral_mode)

# Memory Cue Labels: Positionen 2-6 (Index 1-5) — die "inneren" Strukturpositionen
# Pos 1 = First Downbeat (~0%), Pos 7-10 = Outro-Bereich → Regelsystem reicht
N_MEM_LABELS     = 5    # Mem-Slot 2 bis 6


def _genre_hash(genre: str) -> float:
    """Deterministischer Genre-Hash normalisiert auf [0, 1]."""
    if not genre:
        return 0.0
    h = hash(genre.strip().lower()) % 10000
    return h / 10000.0


def _get_anlz_paths(content, db_dir: str) -> tuple[str, str] | None:
    """ANLZ .DAT und .EXT Pfade fuer einen Track."""
    anlz_rel = content.AnalysisDataPath
    if not anlz_rel:
        return None
    dat_path = os.path.join(db_dir, "share", anlz_rel.lstrip("/"))
    ext_path = dat_path.replace(".DAT", ".EXT")
    if not os.path.exists(dat_path):
        return None
    return dat_path, ext_path


def extract_labels(db) -> dict[str, dict]:
    """
    Scannt alle Tracks und extrahiert Hot A + Hot C + Memory Cue Positionen.
    Filtert AutoCue-generierte Cues aus.

    Returns: {content_id: {"hot_a_ms": int, "hot_c_ms": int,
                            "hot_a_rel": float, "hot_c_rel": float,
                            "mem_rel": [float, ...],  # bis zu N_MEM_LABELS
                            "content": obj}}
    """
    labels = {}
    for content in db.get_content():
        if not content.Cues or not content.Length or content.Length <= 0:
            continue

        duration_ms = content.Length * 1000.0
        hot_a_ms = None
        hot_c_ms = None
        mem_positions: list[float] = []

        for cue in sorted(content.Cues, key=lambda c: c.InMsec or 0):
            # AutoCue-generierte Cues ueberspringen
            if _cfg.is_autocue_comment(cue.Comment or ""):
                continue
            if cue.Kind == 1 and cue.InMsec is not None:
                hot_a_ms = int(cue.InMsec)
            elif cue.Kind == 3 and cue.InMsec is not None:
                hot_c_ms = int(cue.InMsec)
            elif cue.Kind == 0 and cue.InMsec is not None:
                mem_positions.append(cue.InMsec / duration_ms)

        if hot_a_ms is not None and hot_c_ms is not None and duration_ms > 0:
            labels[str(content.ID)] = {
                "hot_a_ms": hot_a_ms,
                "hot_c_ms": hot_c_ms,
                "hot_a_rel": hot_a_ms / duration_ms,
                "hot_c_rel": hot_c_ms / duration_ms,
                "mem_rel": mem_positions[:N_MEM_LABELS + 1],  # +1 weil Index 0 (First Downbeat) uebersprungen wird
                "content": content,
            }

    return labels


def extract_features_fast(content, dat_path: str, ext_path: str,
                           db, vectors_norm=None, meta_list=None,
                           spectral_mode: str = "off") -> np.ndarray:
    """
    Feature-Extraktion fuer ML Training.
    Optional mit Spektral-Features (erfordert Audio-Dateien).
    Returns: 1D Array shape (get_feature_dim(spectral_mode),)
    """
    total_dim = get_feature_dim(spectral_mode)
    features = np.zeros(total_dim, dtype=np.float32)
    offset = 0

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
    duration = content.Length or 0
    features[offset]     = bpm / 200.0          # Normalisiert
    features[offset + 1] = duration / 600.0     # Normalisiert
    features[offset + 2] = _genre_hash(getattr(content, 'GenreName', '') or '')
    features[offset + 3] = _key_to_camelot(getattr(content, 'KeyName', '') or '')
    offset += METADATA_DIM

    # --- MIK Cues (9) ---
    try:
        artist = content.Artist.Name if content.Artist else ""
        title = content.Title or ""
        folder = content.FolderPath or ""
        mik = get_mik_data(folder, artist, title)
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

        # --- Extra (2): Beat/Bar Count ---
        features[FEATURE_DIM_FAST - 2] = grid.count / 2000.0
        features[FEATURE_DIM_FAST - 1] = (grid.bar_count if hasattr(grid, 'bar_count') else grid.count // 4) / 500.0
    except Exception:
        pass
    offset += PHRASE_DIM

    # --- CBR Twins (13) ---
    try:
        if vectors_norm is not None and meta_list is not None:
            vec = vectorize_from_db(content)
            if vec is not None:
                genre = (getattr(content, 'GenreName', '') or '').lower()
                twins = find_twins(vec, n=5, genre=genre)
                # Eigenen Track ausschliessen
                twins = [t for t in twins if str(t.get("content_id", "")) != str(content.ID)][:5]

                if twins:
                    twin_ids = [str(t["content_id"]) for t in twins]
                    pattern = analyze_hot_cue_pattern(db, twin_ids)

                    # Twin Hot A Positionen
                    rel_a = pattern.get("hot_a_relative_pos", 0)
                    rel_c = pattern.get("hot_c_relative_pos", 0)

                    # Individuelle Twin-Positionen abrufen
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

                    features[offset + 10] = rel_a if rel_a else 0.0
                    features[offset + 11] = rel_c if rel_c else 0.0
                    features[offset + 12] = pattern.get("cue_spacing", 32) / 64.0
    except Exception:
        pass

    # --- Spektral-Features (optional, am Ende angehaengt) ---
    spectral_dim = get_spectral_dim(spectral_mode)
    if spectral_dim > 0:
        audio_path = content.FolderPath or ""
        duration = content.Length or 0
        cache_dir = str(_cfg.get("spectral_cache_dir", "data/spectral_cache"))
        try:
            spec_feat = extract_spectral_features(
                audio_path, duration, mode=spectral_mode, cache_dir=cache_dir
            )
            if spec_feat is not None and len(spec_feat) == spectral_dim:
                features[FEATURE_DIM_BASE:FEATURE_DIM_BASE + spectral_dim] = spec_feat
        except Exception as e:
            pass  # Spektral-Features auf 0 lassen bei Fehler

    return features


def export_dataset(mode: str = "fast", output_dir: str = "data/ml",
                   spectral_mode: str | None = None) -> dict:
    """
    Hauptfunktion: Extrahiert Features + Labels fuer alle Tracks.

    Args:
        mode:           "fast" oder "full"
        output_dir:     Ausgabeverzeichnis
        spectral_mode:  "custom"|"openl3"|"auto"|"off" (None = aus config.yaml)
    """
    # Spektral-Mode bestimmen
    if spectral_mode is None:
        spectral_mode = str(_cfg.get("spectral_mode", "off"))
    spec_dim = get_spectral_dim(spectral_mode)
    total_dim = get_feature_dim(spectral_mode)

    print("=" * 60)
    print("  ML Training Data Export")
    print(f"  Spektral-Mode: {spectral_mode} (+{spec_dim} dims = {total_dim} total)")
    print("=" * 60)

    t0 = time.time()

    # 1. DB oeffnen
    print("\n1. Rekordbox-DB oeffnen...")
    db = open_db()
    all_content = list(db.get_content())
    print(f"   {len(all_content)} Tracks in Bibliothek")

    # DB-Dir fuer ANLZ-Pfade
    from app.beatgrid import _DB_DIR
    db_dir = _DB_DIR

    # 2. Labels extrahieren
    print("2. Tracks mit manuellen Hot A + Hot C scannen...")
    labels = extract_labels(db)
    print(f"   {len(labels)} Tracks mit manuellen Hot A + Hot C gefunden")

    if not labels:
        print("   KEINE Tracks gefunden. Abbruch.")
        return {"n_tracks": 0}

    # 3. Vektor-DB laden (fuer CBR Twin Features)
    print("3. Vektor-DB laden fuer CBR Twins...")
    vectors_norm, meta_list = None, None
    try:
        vectors_norm, meta_list, _ = load_vector_db()
        print(f"   {len(meta_list)} Vektoren geladen")
    except Exception as e:
        print(f"   Vektor-DB nicht verfuegbar: {e}")
        print("   CBR Twin Features werden auf 0 gesetzt")

    # 4. Features extrahieren
    desc = f"Extrahiere ({spectral_mode})" if spectral_mode != "off" else "Extrahiere"
    print(f"\n4. Feature-Extraktion ({mode} mode, spectral={spectral_mode})...")
    feature_list = []
    label_list = []
    mem_label_list = []
    meta_out = []
    skipped = 0

    for content_id, lbl in tqdm(labels.items(), desc=desc):
        content = lbl["content"]
        paths = _get_anlz_paths(content, db_dir)
        if paths is None:
            skipped += 1
            continue
        dat_path, ext_path = paths

        try:
            feat = extract_features_fast(
                content, dat_path, ext_path, db,
                vectors_norm=vectors_norm, meta_list=meta_list,
                spectral_mode=spectral_mode,
            )
            feature_list.append(feat)
            label_list.append([lbl["hot_a_rel"], lbl["hot_c_rel"]])

            # Memory Cue Labels: Positionen 2-6 (skip 1. = First Downbeat)
            mem_rel = lbl.get("mem_rel", [])
            # Slot 2-6 = Index 1-5 in der Memory-Liste
            mem_labels = [0.0] * N_MEM_LABELS
            for i in range(N_MEM_LABELS):
                idx = i + 1  # Skip index 0 (First Downbeat)
                if idx < len(mem_rel):
                    mem_labels[i] = mem_rel[idx]
            mem_label_list.append(mem_labels)

            meta_out.append({
                "content_id": content_id,
                "title": content.Title or "",
                "artist": content.Artist.Name if content.Artist else "",
                "bpm": (content.BPM or 0) / 100.0,
                "duration": content.Length or 0,
                "genre": getattr(content, 'GenreName', '') or '',
                "key": getattr(content, 'KeyName', '') or '',
                "n_manual_mem": len(lbl.get("mem_rel", [])),
            })
        except Exception as e:
            skipped += 1
            continue

    # 5. Speichern
    X = np.array(feature_list, dtype=np.float32)
    Y = np.array(label_list, dtype=np.float32)
    Y_mem = np.array(mem_label_list, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features_X.npy"), X)
    np.save(os.path.join(output_dir, "labels_Y.npy"), Y)
    np.save(os.path.join(output_dir, "labels_Y_mem.npy"), Y_mem)
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta_out, f)

    elapsed = time.time() - t0

    # 6. Report
    print(f"\n{'=' * 60}")
    print(f"  Export abgeschlossen in {elapsed:.1f}s")
    print(f"  Tracks: {len(X)} exportiert, {skipped} uebersprungen")
    print(f"  Feature-Shape: {X.shape}")
    print(f"  Label-Shape (Hot): {Y.shape}")
    print(f"  Label-Shape (Mem): {Y_mem.shape}")
    print(f"  Hot A rel. Position: Median={np.median(Y[:, 0]):.3f}, "
          f"Mean={np.mean(Y[:, 0]):.3f}, Std={np.std(Y[:, 0]):.3f}")
    print(f"  Hot C rel. Position: Median={np.median(Y[:, 1]):.3f}, "
          f"Mean={np.mean(Y[:, 1]):.3f}, Std={np.std(Y[:, 1]):.3f}")
    # Memory Cue Stats
    n_with_mem = np.sum(Y_mem[:, 0] > 0)
    print(f"  Memory Cues: {n_with_mem}/{len(Y_mem)} Tracks mit Mem-Labels")
    for i in range(N_MEM_LABELS):
        valid = Y_mem[:, i][Y_mem[:, i] > 0]
        if len(valid) > 0:
            print(f"    Slot {i+2}: Median={np.median(valid):.3f}, "
                  f"N={len(valid)}")
    print(f"  Gespeichert in: {output_dir}/")
    print(f"{'=' * 60}")

    return {
        "n_tracks": len(X),
        "n_skipped": skipped,
        "feature_dim": X.shape[1] if len(X) > 0 else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Training Data Export")
    parser.add_argument("--mode", default="fast", choices=["fast", "full"])
    parser.add_argument("--output", default="data/ml")
    parser.add_argument("--spectral", default=None,
                        choices=["custom", "openl3", "auto", "off"],
                        help="Spektral-Feature-Mode (ueberschreibt config.yaml)")
    args = parser.parse_args()

    export_dataset(mode=args.mode, output_dir=args.output,
                   spectral_mode=args.spectral)
