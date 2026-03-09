"""
v3 Case-Based Reasoning (CBR) Engine
Findet die aehnlichsten Tracks ('Zwillinge') und leitet daraus
adaptive Cue-Parameter ab:
  - Cue-Spacing (32 oder 64 Beats)
  - Hot-Cue-Platzierungsmuster
  - Energie-Profil-Vergleich (Waveform Match)
"""

from dataclasses import dataclass, field

import numpy as np

from app.beatgrid import open_db, find_content
from app.segments import TrackAnalysis
from app.vectorize import load_vector_db, vectorize_single


# --- Datenmodell ---

@dataclass
class CBRResult:
    """Ergebnis der CBR-Analyse fuer einen Track."""
    twins: list[dict] = field(default_factory=list)
    cue_spacing: int = 32
    hot_cue_pattern: dict = field(default_factory=dict)
    energy_match: dict = field(default_factory=dict)
    structure_label: str = "komplex"
    explanation: str = ""
    # Gelernte Offsets aus Feedback-Loop
    hot_a_offset_ms: int = 0
    hot_c_offset_ms: int = 0


# --- 2a. Similarity-Matching ---

def find_twins(new_vector: np.ndarray, n: int = 5,
               genre: str = "") -> list[dict]:
    """
    Findet die Top-N aehnlichsten Tracks per Cosine-Similarity.
    Brute-Force auf 5146 Vektoren — instant.

    Same-Genre-First: Wenn genre angegeben, wird zuerst innerhalb des
    gleichen Genres gesucht. Werden dort nicht genuegend Zwillinge gefunden,
    wird auf alle Tracks erweitert (Fallback).

    Args:
        new_vector: Normalisierter Feature-Vektor des neuen Tracks
        n: Anzahl Zwillinge
        genre: Genre-String des neuen Tracks (lowercase, aus MP3-Tag)

    Returns:
        Liste von Dicts: [{"index": i, "similarity": 0.95,
                           "content_id": ..., "title": ...}, ...]
    """
    vectors, meta, _ = load_vector_db()

    # Cosine-Similarity: dot(a, b) / (|a| * |b|)
    new_norm = np.linalg.norm(new_vector)
    if new_norm == 0:
        return []

    # Vektor-Normen fuer alle Tracks
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0

    # Dot-Product aller Vektoren mit dem neuen Vektor
    dots = vectors @ new_vector
    similarities = dots / (norms * new_norm)

    def _build_results(indices):
        results = []
        for idx in indices:
            m = meta[idx]
            results.append({
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "content_id": m["content_id"],
                "title": m["title"],
                "artist": m.get("artist", ""),
                "bpm": m["bpm"],
                "duration": m["duration"],
                "genre": m.get("genre", ""),
                "key": m.get("key", ""),
                "path": m["path"],
                "n_cues": m["n_cues"],
            })
        return results

    # Same-Genre-First: nur Tracks im gleichen Genre durchsuchen
    if genre:
        genre_mask = np.array([m.get("genre", "") == genre for m in meta])
        if genre_mask.sum() >= n:
            genre_sims = similarities * genre_mask
            top_genre = np.argsort(genre_sims)[::-1][:n]
            results = _build_results(top_genre)
            if len(results) >= n:
                return results

    # Fallback: alle Tracks (bisheriges Verhalten)
    top_indices = np.argsort(similarities)[::-1][:n]
    return _build_results(top_indices)


# --- 2b. Cue-Spacing-Analyse ---

def analyze_cue_spacing(db, twin_ids: list[str]) -> int:
    """
    Analysiert die Memory-Cue-Abstande der Zwillinge.
    Bestimmt ob der User 32er oder 64er Beat-Raster nutzt.

    Args:
        db: Offene Rekordbox-DB
        twin_ids: Liste von ContentIDs der Zwillinge

    Returns:
        32 oder 64 (empfohlenes Beat-Raster)
    """
    all_spacings_beats = []

    for cid in twin_ids:
        content = db.get_content(ID=cid)
        if content is None:
            continue

        bpm = (content.BPM or 0) / 100.0
        if bpm <= 0:
            continue

        # Memory Cues (Kind=0) sortiert nach Zeit
        mem_cues = sorted(
            [c for c in content.Cues if c.Kind == 0 and c.InMsec is not None],
            key=lambda c: c.InMsec,
        )

        if len(mem_cues) < 2:
            continue

        # Paarweise Abstande in Beats
        for i in range(1, len(mem_cues)):
            dt_sec = (mem_cues[i].InMsec - mem_cues[i - 1].InMsec) / 1000.0
            dt_beats = dt_sec * (bpm / 60.0)

            # Nur realistische Abstande (8-128 Beats)
            if 8 <= dt_beats <= 128:
                all_spacings_beats.append(dt_beats)

    if not all_spacings_beats:
        return 32  # Default

    median_spacing = float(np.median(all_spacings_beats))

    # Entscheidung: gelernter Threshold (default 48)
    from app.learner import load_learned_params
    learned = load_learned_params()
    threshold = learned.get("cue_spacing_threshold", 48)

    if median_spacing >= threshold:
        return 64
    return 32


# --- 2c. Hot-Cue-Muster-Analyse ---

def analyze_hot_cue_pattern(db, twin_ids: list[str]) -> dict:
    """
    Analysiert wie der User Hot Cues bei den Zwillingen platziert hat.

    Returns: {
        "hot_a_relative_pos": float,  # Median relative Position (0-1)
        "hot_b_offset_beats": int,    # Beats vor Hot C
        "hot_c_relative_pos": float,  # Median relative Position
        "n_twins_with_hots": int,     # Wieviele Zwillinge Hot Cues haben
    }
    """
    hot_a_positions = []
    hot_b_offsets = []
    hot_c_positions = []
    n_with_hots = 0

    for cid in twin_ids:
        content = db.get_content(ID=cid)
        if content is None:
            continue

        duration = content.Length or 0
        bpm = (content.BPM or 0) / 100.0
        if duration <= 0 or bpm <= 0:
            continue

        hot_a = None
        hot_b = None
        hot_c = None

        for cue in content.Cues:
            if cue.Kind == 1 and cue.InMsec is not None:
                hot_a = cue
            elif cue.Kind == 2 and cue.InMsec is not None:
                hot_b = cue
            elif cue.Kind == 3 and cue.InMsec is not None:
                hot_c = cue

        if hot_a or hot_b or hot_c:
            n_with_hots += 1

        # Hot A relative Position
        if hot_a:
            rel = (hot_a.InMsec / 1000.0) / duration
            hot_a_positions.append(rel)

        # Hot C relative Position
        if hot_c:
            rel = (hot_c.InMsec / 1000.0) / duration
            hot_c_positions.append(rel)

        # Hot B Offset zu Hot C (in Beats)
        if hot_b and hot_c:
            dt_sec = (hot_c.InMsec - hot_b.InMsec) / 1000.0
            dt_beats = dt_sec * (bpm / 60.0)
            if dt_beats > 0:
                hot_b_offsets.append(dt_beats)

    return {
        "hot_a_relative_pos": float(np.median(hot_a_positions))
            if hot_a_positions else 0.15,
        "hot_b_offset_beats": int(round(np.median(hot_b_offsets)))
            if hot_b_offsets else 32,
        "hot_c_relative_pos": float(np.median(hot_c_positions))
            if hot_c_positions else 0.85,
        "n_twins_with_hots": n_with_hots,
    }


# --- 2d. Waveform-Vergleich ---

def compare_energy_profile(new_analysis: TrackAnalysis,
                           db, twin_ids: list[str]) -> dict:
    """
    Vergleicht Energie-Profil des neuen Tracks mit Zwillingen.
    Findet Energie-Loecher (Breaks) und bestimmt Hot-A-Position.

    Nutzt die vorhandenen Cue-Positionen der Zwillinge als Referenz
    statt voller Audio-Analyse (viel schneller).

    Returns: {
        "first_break_end_sec": float | None,
        "min_break_duration_sec": float,
        "energy_profile_match": str,
    }
    """
    # Analyse des neuen Tracks: Finde Energie-Loecher
    energy = new_analysis.energy
    grid = new_analysis.grid
    segments = new_analysis.segments

    # Erstes Energie-Loch im neuen Track (Break-Segment nach Intro)
    first_break_end = None
    first_break_duration = 0.0

    for i, seg in enumerate(segments):
        if seg.kind == "break" and i > 0:
            first_break_end = seg.end_time
            first_break_duration = seg.end_time - seg.start_time
            break

    # Median Break-Laenge aus Zwillingen (basierend auf Cue-Positionen)
    twin_break_durations = []
    for cid in twin_ids:
        content = db.get_content(ID=cid)
        if content is None:
            continue

        bpm = (content.BPM or 0) / 100.0
        if bpm <= 0:
            continue

        # Hot A = typischerweise am Break
        hot_a = None
        for cue in content.Cues:
            if cue.Kind == 1 and cue.InMsec is not None:
                hot_a = cue
                break

        if hot_a:
            # Suche den Memory Cue direkt VOR Hot A (= Break-Start)
            mem_cues = sorted(
                [c for c in content.Cues
                 if c.Kind == 0 and c.InMsec is not None
                 and c.InMsec < hot_a.InMsec],
                key=lambda c: c.InMsec,
            )
            if mem_cues:
                break_start_ms = mem_cues[-1].InMsec
                break_dur = (hot_a.InMsec - break_start_ms) / 1000.0
                if 2.0 <= break_dur <= 120.0:
                    twin_break_durations.append(break_dur)

    min_break_sec = (float(np.median(twin_break_durations))
                     if twin_break_durations else 8.0)

    # Qualitaets-Einschaetzung
    if first_break_end and first_break_duration >= min_break_sec:
        match_quality = "gut"
    elif first_break_end:
        match_quality = "kurzer_break"
    else:
        match_quality = "kein_break"

    return {
        "first_break_end_sec": first_break_end,
        "first_break_duration_sec": first_break_duration,
        "min_break_duration_sec": min_break_sec,
        "twin_break_count": len(twin_break_durations),
        "energy_profile_match": match_quality,
    }


# --- 2e. Haupt-Orchestrierung ---

def run_cbr(analysis: TrackAnalysis,
            audio_path: str,
            db=None) -> CBRResult:
    """
    Kompletter CBR-Durchlauf fuer einen neuen Track.

    1. Track vektorisieren
    2. Top 5 Zwillinge finden
    3. Cue-Spacing analysieren (32 vs 64)
    4. Hot-Cue-Muster ableiten
    5. Energie-Profil vergleichen
    6. Finalen Check ausgeben

    Args:
        analysis: TrackAnalysis des neuen Tracks
        audio_path: Pfad zur Audio-Datei
        db: Offene Rekordbox-DB (oder None fuer auto-open)

    Returns:
        CBRResult mit allen adaptiven Parametern
    """
    result = CBRResult()

    # --- Vektor-DB pruefen ---
    try:
        vectors, meta, scaler = load_vector_db()
    except FileNotFoundError:
        print("  CBR: Keine Vektor-DB gefunden. Nutze regelbasierte Logik.")
        result.explanation = (
            "CBR: Keine Vektor-DB. Regelbasierte Logik (32-Beat)."
        )
        return result

    # --- DB oeffnen ---
    own_db = False
    if db is None:
        db = open_db()
        own_db = True

    # --- 1. Neuen Track vektorisieren ---
    content = find_content(db, audio_path)
    if content is None:
        result.explanation = "CBR: Track nicht in DB. Regelbasierte Logik."
        return result

    # Genre + Key aus Rekordbox (= aus MP3-Tag) lesen
    from app.vectorize import _get_genre, _get_key_name
    genre = _get_genre(content)
    key   = _get_key_name(content)

    new_vec = vectorize_single(content, audio_path, scaler)

    # --- 2. Zwillinge finden (Same-Genre-First) ---
    twins = find_twins(new_vec, n=5, genre=genre)

    # Eigenen Track aus Ergebnissen entfernen
    twins = [t for t in twins if t["content_id"] != content.ID][:5]
    result.twins = twins

    if not twins:
        result.explanation = "CBR: Keine Zwillinge gefunden. Regelbasierte Logik."
        return result

    twin_ids = [t["content_id"] for t in twins]

    # --- 3. Cue-Spacing ---
    result.cue_spacing = analyze_cue_spacing(db, twin_ids)

    # --- 4. Hot-Cue-Muster ---
    result.hot_cue_pattern = analyze_hot_cue_pattern(db, twin_ids)

    # --- 5. Energie-Profil ---
    result.energy_match = compare_energy_profile(analysis, db, twin_ids)

    # --- 5b. Gelernte Parameter aus Learning-DB (Inferenz) ---
    try:
        from app.learning_db import get_db as get_learning_db, get_learned_params_for_track
        learning_conn = get_learning_db()
        bpm_val  = (content.BPM or 0) / 100.0
        dur_s    = float(content.Length or 0)
        learned = get_learned_params_for_track(
            learning_conn, bpm=bpm_val,
            genre=genre, duration_s=dur_s, key=key,
        )
        learning_conn.close()
    except Exception:
        # Fallback: alte learned_params.json
        from app.learner import load_learned_params
        learned = load_learned_params()

    result.hot_a_offset_ms = int(learned.get("hot_a_time_offset_ms",
                                              learned.get("hot_a_offset_ms", 0)))
    result.hot_c_offset_ms = int(learned.get("hot_c_time_offset_ms",
                                              learned.get("hot_c_offset_ms", 0)))

    # Hot B Offset aus Learner (falls gelernt)
    learned_hot_b = learned.get("hot_b_offset_beats")
    if learned_hot_b and learned_hot_b != 32:
        result.hot_cue_pattern["hot_b_offset_beats"] = int(learned_hot_b)

    # --- 6. Struktur-Label + Erklaerung ---
    if result.cue_spacing == 64:
        result.structure_label = "hypnotisch"
        spacing_reason = "hypnotisch/eintoening"
    else:
        result.structure_label = "komplex"
        spacing_reason = "komplex/dynamisch"

    best_twin = twins[0]
    best_sim = best_twin["similarity"]
    best_name = best_twin["title"]
    best_artist = best_twin.get("artist", "")
    twin_str = f"{best_name}"
    if best_artist:
        twin_str = f"{best_artist} - {best_name}"

    result.explanation = (
        f"Aehnlichster Track gefunden: {twin_str} "
        f"(Similarity: {best_sim:.2f}). "
        f"Wende {result.cue_spacing}-Beat-Logik an, "
        f"da Track-Struktur als {spacing_reason} eingestuft wurde."
    )

    return result


# --- Debug-Ausgabe ---

def print_cbr_result(result: CBRResult) -> None:
    """Gibt CBR-Ergebnis formatiert aus."""
    print(f"\n{'='*60}")
    print(f"  CBR-Analyse")
    print(f"{'='*60}")

    if result.twins:
        print(f"\n  Top 5 Zwillinge:")
        for i, t in enumerate(result.twins, 1):
            artist = t.get("artist", "")
            name = f"{artist} - {t['title']}" if artist else t["title"]
            print(f"    {i}. {name}")
            print(f"       BPM: {t['bpm']:.0f}  "
                  f"Dauer: {t['duration']//60:.0f}:{t['duration']%60:02.0f}  "
                  f"Cues: {t['n_cues']}  "
                  f"Sim: {t['similarity']:.3f}")

    print(f"\n  Cue-Spacing:    {result.cue_spacing} Beats "
          f"({result.structure_label})")

    if result.hot_cue_pattern:
        p = result.hot_cue_pattern
        print(f"  Hot A Position: {p['hot_a_relative_pos']:.1%} "
              f"(Median der Zwillinge)")
        print(f"  Hot B Offset:   {p['hot_b_offset_beats']} Beats vor Hot C")
        print(f"  Hot C Position: {p['hot_c_relative_pos']:.1%}")

    if result.energy_match:
        e = result.energy_match
        if e["first_break_end_sec"]:
            m = int(e["first_break_end_sec"] // 60)
            s = e["first_break_end_sec"] % 60
            print(f"  Erster Break:   endet bei {m}:{s:05.2f} "
                  f"(Dauer: {e['first_break_duration_sec']:.1f}s, "
                  f"Match: {e['energy_profile_match']})")
        else:
            print(f"  Erster Break:   nicht gefunden "
                  f"(Match: {e['energy_profile_match']})")

    print(f"\n  → {result.explanation}")
    print()
