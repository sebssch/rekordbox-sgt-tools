"""
v3 Learning Database — SQLite-basierte Lern-DB
Speichert Fingerprints, Vorhersagen, Korrekturen und gelernte Patterns
fuer die intelligente Inferenz bei neuen Tracks.

Ersetzt die bisherigen predictions.jsonl + learned_params.json mit einer
strukturierten, per-Track-Typ-basierten Lern-Datenbank.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime

import numpy as np


# --- Pfade ---

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_BASE_DIR, "data", "agent_learning.db")
_PREDICTIONS_LOG = os.path.join(_BASE_DIR, "data", "predictions.jsonl")
_LEARNED_PARAMS_PATH = os.path.join(_BASE_DIR, "data", "learned_params.json")


# --- Schema ---

_SCHEMA = """
-- Tracks mit Fingerprint
CREATE TABLE IF NOT EXISTS tracks (
    content_id TEXT PRIMARY KEY,
    title TEXT,
    artist TEXT,
    bpm REAL,
    duration INTEGER,
    genre TEXT,
    key TEXT DEFAULT '',
    energy_vector BLOB,
    fingerprint_hash TEXT,
    cluster_id INTEGER DEFAULT 0,
    updated_at TEXT
);

-- Vorhersagen (ersetzt predictions.jsonl)
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    cue_kind INTEGER NOT NULL,
    predicted_ms INTEGER NOT NULL,
    comment TEXT,
    cbr_spacing INTEGER DEFAULT 32,
    cbr_twin_id TEXT,
    cbr_twin_similarity REAL,
    FOREIGN KEY (content_id) REFERENCES tracks(content_id)
);

-- Korrekturen (Diff zwischen Vorhersage und User-Korrektur)
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id TEXT NOT NULL,
    prediction_id INTEGER,
    cue_kind INTEGER NOT NULL,
    action TEXT NOT NULL,
    predicted_ms INTEGER,
    actual_ms INTEGER,
    delta_ms INTEGER DEFAULT 0,
    delta_beats REAL DEFAULT 0.0,
    learned_at TEXT,
    FOREIGN KEY (content_id) REFERENCES tracks(content_id),
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- Gelernte Patterns (per Fingerprint-Cluster)
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint_hash TEXT NOT NULL,
    parameter TEXT NOT NULL,
    learned_value REAL NOT NULL,
    confidence REAL DEFAULT 0.0,
    n_samples INTEGER DEFAULT 0,
    updated_at TEXT,
    UNIQUE(fingerprint_hash, parameter)
);

-- Meta-Information
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Indices fuer schnelle Abfragen
CREATE INDEX IF NOT EXISTS idx_predictions_content
    ON predictions(content_id);
CREATE INDEX IF NOT EXISTS idx_corrections_content
    ON corrections(content_id);
CREATE INDEX IF NOT EXISTS idx_tracks_fingerprint
    ON tracks(fingerprint_hash);
CREATE INDEX IF NOT EXISTS idx_tracks_bpm
    ON tracks(bpm);
CREATE INDEX IF NOT EXISTS idx_patterns_fingerprint
    ON patterns(fingerprint_hash);
"""


# --- DB-Verbindung ---

def get_db() -> sqlite3.Connection:
    """Oeffnet/erstellt die Learning-DB. Gibt Connection zurueck."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30)  # 30s Timeout bei Lock
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Erstellt alle Tabellen falls nicht vorhanden. Fuehrt Migration durch."""
    conn.executescript(_SCHEMA)
    # Migration: key-Spalte zu bestehenden DBs hinzufuegen
    try:
        conn.execute("ALTER TABLE tracks ADD COLUMN key TEXT DEFAULT ''")
        conn.commit()
    except Exception:
        pass  # Spalte existiert bereits
    conn.commit()


# --- Fingerprint-Berechnung ---

def normalize_genre(raw: str) -> str:
    """
    Normalisiert einen Genre-String (lowercase, strip, max 20 Zeichen).
    Leer zurueckgeben wenn unbekannt — dann greift nur BPM+Duration+Key.
    """
    return (raw or "").strip().lower()[:20]


def _duration_bucket(seconds: float) -> str:
    """
    Gruppiert Tracklaenge in 4 Buckets fuer Fingerprint-Clustering.
    Laengere Tracks haben typischerweise mehr Memory Cues und anderen Aufbau.
    """
    if seconds < 240:
        return "short"     # < 4 min  → Edit / Single
    if seconds < 360:
        return "medium"    # 4–6 min  → Standard
    if seconds < 480:
        return "long"      # 6–8 min  → Extended Play
    return "extended"      # > 8 min  → DJ-Edit


def compute_fingerprint(bpm: float, energy_vector=None,
                         genre: str = "", duration_s: float = 0.0,
                         key: str = "") -> str:
    """
    Berechnet einen Fingerprint-Hash fuer Track-Clustering.
    BPM wird in 5er-Buckets gerundet (z.B. 128 → "125-130").
    Energy-Vektor wird auf 16 Bins downsampled und quantisiert.
    Genre (aus MP3-Tag), Duration-Bucket und Key-Gruppe (major/minor)
    werden zur Differenzierung genutzt.
    Returns: SHA256-Hash-String (erste 16 Zeichen)
    """
    # BPM-Bucket (5er-Schritte)
    bpm_lower = int(bpm // 5) * 5
    bpm_bucket = f"{bpm_lower}-{bpm_lower + 5}"

    # Energy-Vektor quantisieren (16 Bins, 0-9)
    energy_str = ""
    if energy_vector is not None:
        ev = np.asarray(energy_vector, dtype=np.float64)
        if len(ev) > 16:
            # Downsample auf 16 Bins
            indices = np.linspace(0, len(ev) - 1, 16, dtype=int)
            ev = ev[indices]
        elif len(ev) < 16:
            ev = np.pad(ev, (0, 16 - len(ev)))
        # Quantisieren auf 0-9
        ev_norm = np.clip(ev, 0, 1)
        quantized = (ev_norm * 9).astype(int)
        energy_str = "".join(str(q) for q in quantized)

    # Genre normalisieren (aus MP3-Tag, vom User gepflegt)
    genre_norm = normalize_genre(genre)

    # Duration-Bucket (Tracklaenge als Strukturindikator)
    dur_bucket = _duration_bucket(float(duration_s))

    # Key-Gruppe: Camelot-Notation — A = minor, B = major
    # Rekordbox speichert z.B. "11A" (minor) oder "11B" (major)
    key_str = (key or "").strip()
    if key_str.endswith("B"):
        key_group = "major"
    elif key_str.endswith("A"):
        key_group = "minor"
    else:
        key_group = ""

    # Hash berechnen
    fingerprint_input = (
        f"{bpm_bucket}|{energy_str}|{genre_norm}|{dur_bucket}|{key_group}"
    )
    hash_val = hashlib.sha256(fingerprint_input.encode()).hexdigest()
    return hash_val[:16]


# --- Track-Verwaltung ---

def upsert_track(conn: sqlite3.Connection, content_id: str,
                  title: str, artist: str, bpm: float,
                  duration: int, genre: str = "", key: str = "",
                  energy_vector=None) -> None:
    """Track in DB einfuegen oder aktualisieren."""
    # Energy-Vektor als Bytes speichern
    ev_blob = None
    if energy_vector is not None:
        ev = np.asarray(energy_vector, dtype=np.float64)
        ev_blob = ev.tobytes()

    duration_s = float(duration) if duration else 0.0
    fp_hash = compute_fingerprint(bpm, energy_vector, genre,
                                   duration_s=duration_s, key=key)
    now = datetime.now().isoformat(timespec="seconds")

    conn.execute("""
        INSERT INTO tracks (content_id, title, artist, bpm, duration,
                            genre, key, energy_vector, fingerprint_hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_id) DO UPDATE SET
            title=excluded.title, artist=excluded.artist,
            bpm=excluded.bpm, duration=excluded.duration,
            genre=excluded.genre, key=excluded.key,
            energy_vector=excluded.energy_vector,
            fingerprint_hash=excluded.fingerprint_hash,
            updated_at=excluded.updated_at
    """, (content_id, title, artist, bpm, duration,
          genre, key, ev_blob, fp_hash, now))
    conn.commit()


# --- Vorhersagen loggen ---

def log_predictions(conn: sqlite3.Connection, content_id: str,
                     cues: list[dict], cbr_spacing: int = 32,
                     cbr_twin_id: str = None,
                     cbr_twin_similarity: float = 0.0) -> None:
    """
    Vorhersagen loggen (ersetzt _log_prediction in writer.py).
    Loescht vorherige Predictions fuer diesen Track.
    cues: list of dicts with keys: time_ms, kind, comment
    """
    now = datetime.now().isoformat(timespec="seconds")

    # Alte Predictions loeschen (neuester Stand gewinnt)
    # Zuerst: prediction_id in verknuepften Corrections auf NULL setzen
    # (Corrections bleiben erhalten, verlieren nur die Referenz)
    old_ids = conn.execute(
        "SELECT id FROM predictions WHERE content_id = ?", (content_id,)
    ).fetchall()
    if old_ids:
        id_list = [r["id"] for r in old_ids]
        placeholders = ",".join("?" * len(id_list))
        conn.execute(
            f"UPDATE corrections SET prediction_id = NULL "
            f"WHERE prediction_id IN ({placeholders})",
            id_list,
        )
    conn.execute("DELETE FROM predictions WHERE content_id = ?",
                 (content_id,))

    for cue in cues:
        conn.execute("""
            INSERT INTO predictions
                (content_id, timestamp, cue_kind, predicted_ms, comment,
                 cbr_spacing, cbr_twin_id, cbr_twin_similarity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (content_id, now, cue["kind"], cue["time_ms"],
              cue.get("comment", ""), cbr_spacing,
              cbr_twin_id, cbr_twin_similarity))

    conn.commit()


# --- Korrekturen speichern ---

def log_corrections(conn: sqlite3.Connection, content_id: str,
                     corrections: list[dict]) -> None:
    """
    Korrekturen speichern.
    corrections: list of dicts with keys: cue_kind, action,
                 predicted_ms, actual_ms, delta_ms, delta_beats
    """
    now = datetime.now().isoformat(timespec="seconds")

    for corr in corrections:
        # Finde passende Prediction-ID
        pred_id = None
        if corr.get("predicted_ms") is not None:
            row = conn.execute("""
                SELECT id FROM predictions
                WHERE content_id = ? AND cue_kind = ?
                    AND predicted_ms = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (content_id, corr["cue_kind"],
                  corr["predicted_ms"])).fetchone()
            if row:
                pred_id = row["id"]

        conn.execute("""
            INSERT INTO corrections
                (content_id, prediction_id, cue_kind, action,
                 predicted_ms, actual_ms, delta_ms, delta_beats,
                 learned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (content_id, pred_id, corr["cue_kind"], corr["action"],
              corr.get("predicted_ms"), corr.get("actual_ms"),
              corr.get("delta_ms", 0), corr.get("delta_beats", 0.0),
              now))

    conn.commit()


# --- Pattern-Verwaltung ---

def update_pattern(conn: sqlite3.Connection, fingerprint_hash: str,
                    parameter: str, value: float,
                    confidence: float, n_samples: int) -> None:
    """Pattern in DB einfuegen oder aktualisieren (UPSERT)."""
    now = datetime.now().isoformat(timespec="seconds")

    conn.execute("""
        INSERT INTO patterns
            (fingerprint_hash, parameter, learned_value,
             confidence, n_samples, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint_hash, parameter) DO UPDATE SET
            learned_value=excluded.learned_value,
            confidence=excluded.confidence,
            n_samples=excluded.n_samples,
            updated_at=excluded.updated_at
    """, (fingerprint_hash, parameter, value,
          confidence, n_samples, now))
    conn.commit()


def get_patterns_for_fingerprint(conn: sqlite3.Connection,
                                  fingerprint_hash: str) -> dict:
    """
    Alle gelernten Patterns fuer einen Fingerprint laden.
    Returns: {parameter_name: {"value": float, "confidence": float,
                                "n_samples": int}}
    """
    rows = conn.execute("""
        SELECT parameter, learned_value, confidence, n_samples
        FROM patterns WHERE fingerprint_hash = ?
    """, (fingerprint_hash,)).fetchall()

    return {
        row["parameter"]: {
            "value": row["learned_value"],
            "confidence": row["confidence"],
            "n_samples": row["n_samples"],
        }
        for row in rows
    }


# --- Aehnliche Fingerprints finden ---

def find_similar_fingerprints(conn: sqlite3.Connection,
                               fingerprint_hash: str,
                               bpm: float,
                               tolerance_bpm: float = 5.0) -> list[str]:
    """
    Findet aehnliche Fingerprints in der DB.
    Sucht erst exakten Match, dann BPM-Bereich.
    Returns: Liste von fingerprint_hashes (sortiert nach Relevanz).
    """
    result = []

    # 1. Exakter Fingerprint-Match
    rows = conn.execute("""
        SELECT DISTINCT fingerprint_hash FROM tracks
        WHERE fingerprint_hash = ?
    """, (fingerprint_hash,)).fetchall()

    if rows:
        result.append(fingerprint_hash)

    # 2. BPM-Bereich (±tolerance)
    rows = conn.execute("""
        SELECT DISTINCT fingerprint_hash FROM tracks
        WHERE bpm BETWEEN ? AND ?
            AND fingerprint_hash != ?
    """, (bpm - tolerance_bpm, bpm + tolerance_bpm,
          fingerprint_hash)).fetchall()

    for row in rows:
        fp = row["fingerprint_hash"]
        if fp not in result:
            result.append(fp)

    return result


# --- Intelligente Inferenz ---

def get_learned_params_for_track(conn: sqlite3.Connection,
                                  bpm: float, energy_vector=None,
                                  genre: str = "", duration_s: float = 0.0,
                                  key: str = "") -> dict:
    """
    Hauptfunktion fuer Inferenz: Findet passende gelernte Parameter
    fuer einen neuen Track basierend auf seinem Fingerprint.

    1. Berechne Fingerprint
    2. Suche exakten Match in patterns-Tabelle
    3. Falls keiner: Suche aehnliche Fingerprints
    4. Aggregiere Patterns (gewichtet nach Confidence + Samples)

    Returns: Dict mit gelernten Parametern.
             Fehlende Parameter = nicht gelernt → Caller nutzt Defaults.
    """
    result = {}

    fp_hash = compute_fingerprint(bpm, energy_vector, genre,
                                   duration_s=duration_s, key=key)

    # 1. Exakter Match
    patterns = get_patterns_for_fingerprint(conn, fp_hash)
    if patterns:
        for param, data in patterns.items():
            if data["confidence"] > 0.3 and data["n_samples"] >= 2:
                result[param] = data["value"]
        if result:
            return result

    # 2. Aehnliche Fingerprints suchen
    similar = find_similar_fingerprints(conn, fp_hash, bpm)

    if not similar:
        # 3. Globale Patterns (Fingerprint = "_global")
        global_patterns = get_patterns_for_fingerprint(conn, "_global")
        for param, data in global_patterns.items():
            if data["confidence"] > 0.3 and data["n_samples"] >= 3:
                result[param] = data["value"]
        return result

    # Aggregiere Patterns von aehnlichen Fingerprints
    # Gewichtet nach n_samples * confidence
    param_values: dict[str, list[tuple[float, float]]] = {}

    for fp in similar:
        fp_patterns = get_patterns_for_fingerprint(conn, fp)
        for param, data in fp_patterns.items():
            if data["n_samples"] < 2:
                continue
            weight = data["n_samples"] * data["confidence"]
            if param not in param_values:
                param_values[param] = []
            param_values[param].append((data["value"], weight))

    # Gewichteter Durchschnitt
    for param, vw_list in param_values.items():
        total_weight = sum(w for _, w in vw_list)
        if total_weight > 0:
            weighted_val = sum(v * w for v, w in vw_list) / total_weight
            result[param] = weighted_val

    return result


# --- Recursive Intelligence: Auto-Korrektur-Offsets ---

def get_auto_correction_offsets(
    conn: sqlite3.Connection,
    bpm: float,
    genre: str = "",
    duration_s: float = 0.0,
    key: str = "",
    energy_vector=None,
    confidence_threshold: float = 0.80,
) -> dict:
    """
    Liefert gelernte Cue-Offsets wenn Konfidenz > Schwellenwert (Recursive Intelligence).

    Prueft zuerst fingerprint-spezifische Patterns, dann globale.
    Gibt nur Offsets zurueck die mit genuegend Konfidenz gelernt wurden.

    Returns:
        {"hot_a_offset_ms": int, "hot_c_offset_ms": int}
        Wert 0 = kein gelernter Offset (Caller verwendet Default).
    """
    result = {"hot_a_offset_ms": 0, "hot_c_offset_ms": 0}

    fp_hash = compute_fingerprint(bpm, energy_vector, genre,
                                   duration_s=duration_s, key=key)

    # 1. Fingerprint-spezifische Patterns
    patterns = get_patterns_for_fingerprint(conn, fp_hash)
    for param, key in [("hot_a_offset_ms", "hot_a_offset_ms"),
                        ("hot_c_offset_ms", "hot_c_offset_ms")]:
        if param in patterns:
            data = patterns[param]
            if data["confidence"] >= confidence_threshold and data["n_samples"] >= 3:
                result[key] = int(round(data["learned_value"]))

    # 2. Globale Patterns als Fallback (grosszuegigere Mindest-Samples)
    if result["hot_a_offset_ms"] == 0 or result["hot_c_offset_ms"] == 0:
        global_patterns = get_patterns_for_fingerprint(conn, "_global")
        for param, key in [("hot_a_time_offset_ms", "hot_a_offset_ms"),
                            ("hot_c_time_offset_ms", "hot_c_offset_ms")]:
            if result[key] == 0 and param in global_patterns:
                data = global_patterns[param]
                if data["confidence"] >= confidence_threshold and data["n_samples"] >= 5:
                    result[key] = int(round(data["learned_value"]))

    return result


# --- Genauigkeits-Statistiken ---

def get_track_accuracy(conn: sqlite3.Connection,
                        content_id: str = None) -> dict:
    """
    Genauigkeits-Statistiken.
    Ohne content_id: Globale Statistiken.
    Mit content_id: Statistiken fuer einen Track.
    """
    where = ""
    params = ()
    if content_id:
        where = "WHERE content_id = ?"
        params = (content_id,)

    total = conn.execute(
        f"SELECT COUNT(*) FROM corrections {where}", params
    ).fetchone()[0]

    if total == 0:
        return {"total": 0, "kept": 0, "moved": 0, "deleted": 0,
                "accuracy": 0.0}

    kept = conn.execute(
        f"SELECT COUNT(*) FROM corrections {where}"
        f"{' AND' if where else 'WHERE'} action = 'kept'",
        params
    ).fetchone()[0]

    moved = conn.execute(
        f"SELECT COUNT(*) FROM corrections {where}"
        f"{' AND' if where else 'WHERE'} action = 'moved'",
        params
    ).fetchone()[0]

    deleted = conn.execute(
        f"SELECT COUNT(*) FROM corrections {where}"
        f"{' AND' if where else 'WHERE'} action = 'deleted'",
        params
    ).fetchone()[0]

    return {
        "total": total,
        "kept": kept,
        "moved": moved,
        "deleted": deleted,
        "accuracy": kept / total if total > 0 else 0.0,
    }


# --- Review & Learn (Auto-Sync vor Analyse) ---

def review_and_learn(rekordbox_db=None) -> dict:
    """
    Automatischer Review & Learn Prozess:
    1. Scanne alle Tracks mit Predictions
    2. Vergleiche aktuelle Cues mit Vorhersagen
    3. Berechne Fehler und aktualisiere Patterns

    Wird automatisch VOR jeder neuen Analyse aufgerufen.

    Returns: Zusammenfassung {n_tracks, n_corrections, accuracy}
    """
    conn = get_db()

    # Alle Tracks mit Predictions laden
    pred_tracks = conn.execute("""
        SELECT DISTINCT p.content_id, t.bpm, t.fingerprint_hash
        FROM predictions p
        JOIN tracks t ON p.content_id = t.content_id
    """).fetchall()

    if not pred_tracks:
        conn.close()
        return {"n_tracks": 0, "n_corrections": 0, "accuracy": 0.0}

    # Rekordbox-DB oeffnen fuer aktuellen Stand
    if rekordbox_db is None:
        from app.beatgrid import open_db
        rekordbox_db = open_db()

    n_corrections = 0
    all_kept = 0
    all_total = 0

    for row in pred_tracks:
        cid = row["content_id"]
        bpm = row["bpm"] or 0.0
        fp_hash = row["fingerprint_hash"]

        # Aktuelle Cues aus Rekordbox laden
        content = rekordbox_db.get_content(ID=cid)
        if content is None:
            continue

        actual_cues = list(content.Cues) if content.Cues else []

        # Vorhergesagte Cues laden
        preds = conn.execute("""
            SELECT id, cue_kind, predicted_ms, comment
            FROM predictions WHERE content_id = ?
        """, (cid,)).fetchall()

        if not preds:
            continue

        # Korrekturen bereits vorhanden fuer AKTUELLE Predictions?
        # Vergleiche Timestamp: Corrections nach der letzten Prediction → Skip
        latest_pred_time = conn.execute("""
            SELECT MAX(timestamp) FROM predictions WHERE content_id = ?
        """, (cid,)).fetchone()[0]

        latest_corr_time = conn.execute("""
            SELECT MAX(learned_at) FROM corrections WHERE content_id = ?
        """, (cid,)).fetchone()[0]

        if (latest_pred_time is not None
                and latest_corr_time is not None
                and latest_corr_time >= latest_pred_time):
            # Corrections bereits nach letzter Prediction → bereits gelernt
            stats = get_track_accuracy(conn, cid)
            all_kept += stats["kept"]
            all_total += stats["total"]
            continue

        # Matching: Vorhersage vs. aktueller Stand
        corrections = []
        for pred in preds:
            kind = pred["cue_kind"]
            pred_ms = pred["predicted_ms"]

            if kind > 0:
                # Hot Cue: Suche exakten Kind-Match
                match = None
                for cue in actual_cues:
                    if cue.Kind == kind and cue.InMsec is not None:
                        match = cue
                        break

                if match is None:
                    corrections.append({
                        "cue_kind": kind, "action": "deleted",
                        "predicted_ms": pred_ms, "actual_ms": None,
                        "delta_ms": 0, "delta_beats": 0.0,
                    })
                else:
                    delta = match.InMsec - pred_ms
                    action = "kept" if abs(delta) <= 500 else "moved"
                    beats = (abs(delta) / 1000.0) * (bpm / 60.0) if bpm > 0 else 0
                    corrections.append({
                        "cue_kind": kind, "action": action,
                        "predicted_ms": pred_ms,
                        "actual_ms": match.InMsec,
                        "delta_ms": delta,
                        "delta_beats": beats if action == "moved" else 0.0,
                    })
            else:
                # Memory Cue: Naechste Cue gleichen Kinds suchen
                best_dist = float("inf")
                best_cue = None
                for cue in actual_cues:
                    if cue.Kind == 0 and cue.InMsec is not None:
                        dist = abs(cue.InMsec - pred_ms)
                        if dist < best_dist:
                            best_dist = dist
                            best_cue = cue

                if best_cue is None or best_dist > 30000:
                    corrections.append({
                        "cue_kind": 0, "action": "deleted",
                        "predicted_ms": pred_ms, "actual_ms": None,
                        "delta_ms": 0, "delta_beats": 0.0,
                    })
                else:
                    delta = best_cue.InMsec - pred_ms
                    action = "kept" if abs(delta) <= 500 else "moved"
                    beats = (abs(delta) / 1000.0) * (bpm / 60.0) if bpm > 0 else 0
                    corrections.append({
                        "cue_kind": 0, "action": action,
                        "predicted_ms": pred_ms,
                        "actual_ms": best_cue.InMsec,
                        "delta_ms": delta,
                        "delta_beats": beats if action == "moved" else 0.0,
                    })

        if corrections:
            log_corrections(conn, cid, corrections)
            n_corrections += len(corrections)

            # Statistiken zaehlen
            kept = sum(1 for c in corrections if c["action"] == "kept")
            all_kept += kept
            all_total += len(corrections)

            # Patterns aktualisieren fuer diesen Fingerprint
            if fp_hash:
                _update_patterns_from_corrections(
                    conn, fp_hash, corrections, bpm)

    # Globale Patterns aktualisieren
    _update_global_patterns(conn)

    accuracy = all_kept / all_total if all_total > 0 else 0.0

    # Zusammenfassung ausgeben
    n_tracks = len(pred_tracks)
    print(f"\n  Review & Learn: {n_tracks} Tracks, "
          f"{n_corrections} neue Korrekturen, "
          f"Genauigkeit: {accuracy:.1%}")

    conn.close()
    return {
        "n_tracks": n_tracks,
        "n_corrections": n_corrections,
        "accuracy": accuracy,
    }


def _update_patterns_from_corrections(conn: sqlite3.Connection,
                                       fingerprint_hash: str,
                                       corrections: list[dict],
                                       bpm: float) -> None:
    """
    Aktualisiert Patterns basierend auf Korrekturen fuer einen Fingerprint.
    Konservativ: Nur bei >= 3 Korrekturen gleicher Art.
    """
    MIN_CORRECTIONS = 3

    # Hot A Offset
    hot_a = [c for c in corrections
             if c["cue_kind"] == 1 and c["action"] == "moved"]
    if len(hot_a) >= MIN_CORRECTIONS:
        deltas = [c["delta_ms"] for c in hot_a]
        mean_delta = float(np.mean(deltas))
        n_same_dir = max(
            sum(1 for d in deltas if d > 0),
            sum(1 for d in deltas if d < 0)
        )
        consistency = n_same_dir / len(deltas)
        if consistency >= 0.6:
            update_pattern(conn, fingerprint_hash,
                          "hot_a_offset_ms", mean_delta,
                          consistency, len(hot_a))

    # Hot C Offset
    hot_c = [c for c in corrections
             if c["cue_kind"] == 3 and c["action"] == "moved"]
    if len(hot_c) >= MIN_CORRECTIONS:
        deltas = [c["delta_ms"] for c in hot_c]
        mean_delta = float(np.mean(deltas))
        n_same_dir = max(
            sum(1 for d in deltas if d > 0),
            sum(1 for d in deltas if d < 0)
        )
        consistency = n_same_dir / len(deltas)
        if consistency >= 0.6:
            update_pattern(conn, fingerprint_hash,
                          "hot_c_offset_ms", mean_delta,
                          consistency, len(hot_c))

    # Memory Cue Spacing (viele geloescht → 64er bevorzugt)
    mem_total = [c for c in corrections if c["cue_kind"] == 0]
    mem_deleted = [c for c in mem_total if c["action"] == "deleted"]
    if len(mem_total) >= MIN_CORRECTIONS * 2:
        delete_ratio = len(mem_deleted) / len(mem_total)
        if delete_ratio > 0.5:
            update_pattern(conn, fingerprint_hash,
                          "cue_spacing", 64.0,
                          delete_ratio, len(mem_total))
        elif delete_ratio < 0.2:
            update_pattern(conn, fingerprint_hash,
                          "cue_spacing", 32.0,
                          1.0 - delete_ratio, len(mem_total))


def _update_global_patterns(conn: sqlite3.Connection) -> None:
    """Aktualisiert globale Patterns (_global) aus allen Korrekturen."""
    # Hot A global
    rows = conn.execute("""
        SELECT delta_ms FROM corrections
        WHERE cue_kind = 1 AND action = 'moved'
    """).fetchall()

    if len(rows) >= 5:
        deltas = [r["delta_ms"] for r in rows]
        mean_delta = float(np.mean(deltas))
        n_pos = sum(1 for d in deltas if d > 0)
        n_neg = sum(1 for d in deltas if d < 0)
        consistency = max(n_pos, n_neg) / len(deltas)
        if consistency >= 0.6:
            update_pattern(conn, "_global",
                          "hot_a_time_offset_ms", mean_delta,
                          consistency, len(rows))

    # Hot C global
    rows = conn.execute("""
        SELECT delta_ms FROM corrections
        WHERE cue_kind = 3 AND action = 'moved'
    """).fetchall()

    if len(rows) >= 5:
        deltas = [r["delta_ms"] for r in rows]
        mean_delta = float(np.mean(deltas))
        n_pos = sum(1 for d in deltas if d > 0)
        n_neg = sum(1 for d in deltas if d < 0)
        consistency = max(n_pos, n_neg) / len(deltas)
        if consistency >= 0.6:
            update_pattern(conn, "_global",
                          "hot_c_time_offset_ms", mean_delta,
                          consistency, len(rows))

    # Gesamt-Genauigkeit
    stats = get_track_accuracy(conn)
    if stats["total"] > 0:
        update_pattern(conn, "_global",
                      "overall_accuracy", stats["accuracy"],
                      1.0, stats["total"])


# --- Migration ---

def migrate_from_json(predictions_jsonl: str = None,
                       learned_params_json: str = None) -> int:
    """
    Migriert bestehende Daten aus predictions.jsonl und
    learned_params.json in die neue agent_learning.db.
    Returns: Anzahl migrierter Tracks.
    """
    if predictions_jsonl is None:
        predictions_jsonl = _PREDICTIONS_LOG
    if learned_params_json is None:
        learned_params_json = _LEARNED_PARAMS_PATH

    conn = get_db()
    n_migrated = 0

    # 1. predictions.jsonl migrieren
    if os.path.isfile(predictions_jsonl):
        with open(predictions_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cid = entry.get("content_id", "")
                if not cid:
                    continue

                # Track speichern
                upsert_track(
                    conn, cid,
                    title=entry.get("title", ""),
                    artist="",
                    bpm=entry.get("bpm", 0.0),
                    duration=entry.get("duration", 0),
                )

                # Predictions speichern
                cues = entry.get("cues", [])
                cbr = entry.get("cbr", {})
                if cues:
                    log_predictions(
                        conn, cid, cues,
                        cbr_spacing=cbr.get("cue_spacing", 32),
                        cbr_twin_id=cbr.get("best_twin", ""),
                        cbr_twin_similarity=cbr.get(
                            "twin_similarity", 0.0),
                    )
                    n_migrated += 1

    # 2. learned_params.json → globale Patterns
    if os.path.isfile(learned_params_json):
        with open(learned_params_json, "r", encoding="utf-8") as f:
            params = json.load(f)

        n_corr = params.get("n_corrections_used", 0)
        conf = params.get("confidence_scores", {})
        overall_conf = conf.get("overall_accuracy", 0.5)

        param_map = {
            "hot_a_time_offset_ms": params.get("hot_a_time_offset_ms", 0),
            "hot_c_time_offset_ms": params.get("hot_c_time_offset_ms", 0),
            "cue_spacing_threshold": params.get(
                "cue_spacing_threshold", 48),
            "hot_b_offset_beats": params.get("hot_b_offset_beats", 32),
        }

        for param, value in param_map.items():
            if value != 0:
                update_pattern(conn, "_global", param,
                             float(value), overall_conf, n_corr)

    conn.close()
    print(f"  Migration: {n_migrated} Tracks migriert.")
    return n_migrated


# --- Zusammenfassung ---

def print_db_summary(conn: sqlite3.Connection = None) -> None:
    """Gibt eine Zusammenfassung der Learning-DB aus."""
    own_conn = conn is None
    if own_conn:
        conn = get_db()

    n_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    n_preds = conn.execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_corr = conn.execute(
        "SELECT COUNT(*) FROM corrections").fetchone()[0]
    n_patterns = conn.execute(
        "SELECT COUNT(*) FROM patterns").fetchone()[0]

    stats = get_track_accuracy(conn)

    print(f"\n{'='*60}")
    print(f"  agent_learning.db — Zusammenfassung")
    print(f"{'='*60}")
    print(f"  Tracks:       {n_tracks}")
    print(f"  Predictions:  {n_preds}")
    print(f"  Corrections:  {n_corr}")
    print(f"  Patterns:     {n_patterns}")
    print(f"  Genauigkeit:  {stats['accuracy']:.1%}")

    # Globale Patterns anzeigen
    global_patterns = get_patterns_for_fingerprint(conn, "_global")
    if global_patterns:
        print(f"\n  Globale Patterns:")
        for param, data in global_patterns.items():
            print(f"    {param}: {data['value']:.1f} "
                  f"(Conf: {data['confidence']:.2f}, "
                  f"n={data['n_samples']})")

    if own_conn:
        conn.close()
    print()
