"""
v28.1 Cue-Logik
Leitet Cue-Punkte aus der Track-Analyse ab.
Implementiert die DJ-spezifische Cue-Hierarchie:

  Hot Cue A (1):    "The Break"  — Break Start (Percussion-Abfall nach Intro)
  Hot Cue B (2):    "The Setup"  — Exakt 32 Beats vor Hot Cue C
  Hot Cue C (3):    "The Last Drop" — Letzter Drop via Segment + Visual Edge

  Memory Cues:      Max 10, priorisiert:
                    1. Intro-Struktur (32-Beat-Schritte)
                    2. Anker (First Drop, Second Break)
                    3. Phrasen-Uebergaenge
                    4. Outro-Struktur (KEIN letzter Schlag)

Constraints:
  - Mindestabstand Hot A ↔ Hot B: 128 Beats (32 Bars) — konfigurierbar
  - Bei Unterschreitung: CBR-Fallback (Trainings-Tracks analysieren)
"""

from dataclasses import dataclass

from app.beatgrid import (
    BeatGrid,
    snap_to_downbeat,
    snap_to_phrase_boundary,
    get_time_n_beats_before,
    get_time_n_beats_after,
    get_beat_index_at_time,
    get_time_at_beat,
)
from app.segments import TrackAnalysis, Segment
from app import config as _cfg
from app.validator import validate_hot_a, validate_hot_c, CueDecision
from app.phrase_reader import PhraseSegment, get_break_phrases, get_drop_phrases


# --- Konstanten (aus config.yaml geladen, Defaults als Fallback) ---

_c = _cfg.load_config()
MAX_MEMORY_CUES              = _c.get("max_memory_cues",               10)
MIN_HOT_A_B_DISTANCE_BEATS   = _c.get("min_hot_a_b_distance_beats",    128)
_MEMORY_MIN_HOT_BEATS        = _c.get("memory_min_hot_distance_beats", 32)
_MIK_SNAP_TOLERANCE_BEATS    = _c.get("mik_snap_tolerance_beats",      4)


# --- Datenmodell ---

@dataclass
class CuePoint:
    """Ein Cue-Punkt fuer Rekordbox."""
    time_sec: float
    kind: int          # 0=Memory Cue, 1=Hot A, 2=Hot B, 3=Hot C, ...
    name: str          # z.B. "Memory 1", "Hot A"
    comment: str       # Beschreibung des Cue-Punkts
    color: int = -1    # Farbe (-1 = keine)
    priority: int = 99 # Fuer Memory-Cue-Priorisierung (niedrig = wichtig)

    @property
    def time_ms(self) -> int:
        """Zeit in Millisekunden (fuer DjmdCue.InMsec)."""
        return int(round(self.time_sec * 1000))

    @property
    def time_frame(self) -> int:
        """Zeit in Frames (fuer DjmdCue.InFrame, 150 fps)."""
        return int(round(self.time_sec * 150))

    def __repr__(self) -> str:
        m = int(self.time_sec // 60)
        s = self.time_sec % 60
        kind_str = "Mem" if self.kind == 0 else f"Hot {chr(64 + self.kind)}"
        return f"CuePoint({m}:{s:05.2f} {kind_str} '{self.comment}')"


# --- Segment-Suche ---

def _find_break_start(segments: list[Segment]) -> Segment | None:
    """
    Findet den Break Start: Der Moment, in dem die perkussive Energie
    nach dem Intro/ersten Energie-Plateau abfaellt.
    = Erster Break NACH dem Intro.
    """
    for i, seg in enumerate(segments):
        if seg.kind == "break" and i > 0:
            return seg
    return None


def _find_last_drop_visual(analysis: TrackAnalysis) -> float | None:
    """
    Findet den letzten Drop.
    Primaer: Segment-basierte Erkennung (kombiniert SSM + Novelty + Energy + Percussive Ratio).
    Fallback: Visual Edge Detection (Low/Mid Spektral-Ratio) — nur wenn keine Segmente erkannt.
    """
    # Primaer: Segment-basiert (zuverlaessiger als spektrale Ratio)
    seg_result = _find_last_drop_segment(analysis.segments, analysis.grid)
    if seg_result is not None:
        return seg_result

    # Fallback: Visual Edge Detection (falls gar keine Segmente erkannt wurden)
    if hasattr(analysis, 'visual_edge_last_drop') and analysis.visual_edge_last_drop is not None:
        return analysis.visual_edge_last_drop

    return None


def _find_last_drop_segment(segments: list[Segment],
                             grid: BeatGrid = None) -> float | None:
    """
    Findet den letzten Drop ueber Segment-Klassifizierung.
    Bevorzugt Drops in der zweiten Trackhälfte.
    """
    drops = [s for s in segments if s.kind == "drop"]

    if not drops:
        # Fallback: energiereichstes Nicht-Intro/Outro-Segment in zweiter Haelfte
        if grid is not None:
            mid = float(grid.times[len(grid.times) // 2])
            candidates = [s for s in segments
                          if s.kind not in ("intro", "outro")
                          and s.start_time >= mid
                          and s.energy_mean > 0.5]
            if candidates:
                return max(candidates, key=lambda s: s.energy_mean).start_time
        # Globaler Fallback (kein Grid verfuegbar)
        candidates = [s for s in segments
                      if s.kind not in ("intro", "outro", "break")
                      and s.energy_mean > 0.6]
        if candidates:
            return max(candidates, key=lambda s: s.energy_mean).start_time
        return None

    # Bevorzuge letzten Drop in der zweiten Trackhälfte
    if grid is not None and len(grid.times) > 0:
        duration = float(grid.times[-1])
        mid = float(grid.times[len(grid.times) // 2])
        second_half = [d for d in drops if d.start_time >= mid]
        if second_half:
            return second_half[-1].start_time

    return drops[-1].start_time


def _find_first_drop(segments: list[Segment]) -> Segment | None:
    """Findet den ersten Drop im Track."""
    for seg in segments:
        if seg.kind == "drop":
            return seg
    return None


def _find_second_break(segments: list[Segment]) -> Segment | None:
    """Findet den zweiten Break (nach dem ersten Drop)."""
    found_first_break = False
    for seg in segments:
        if seg.kind == "break":
            if found_first_break:
                return seg
            found_first_break = True
    return None


def _find_first_high_energy(segments: list[Segment],
                             min_energy: float = 0.6) -> Segment | None:
    """Findet das erste Segment mit hoher Energie (kein Intro)."""
    for i, seg in enumerate(segments):
        if i > 0 and seg.energy_mean >= min_energy:
            return seg
    return None


# --- 128-Beat Mindestabstand-Check ---

def _check_min_distance(hot_a_time: float, hot_b_time: float,
                         grid: BeatGrid) -> bool:
    """
    Prueft ob zwischen Hot A und Hot B genug Abstand liegt.
    Dynamisch: max(64, min(128, track_beats // 4)) — lockert bei kurzen/langsamen Tracks.
    Returns: True wenn Abstand OK, False wenn zu nah.
    """
    track_beats = grid.count if hasattr(grid, 'count') else len(grid.times)
    min_beats = max(64, min(MIN_HOT_A_B_DISTANCE_BEATS, track_beats // 4))
    a_beat = get_beat_index_at_time(hot_a_time, grid)
    b_beat = get_beat_index_at_time(hot_b_time, grid)
    distance = abs(b_beat - a_beat)
    return distance >= min_beats


def _resolve_distance_conflict(hot_a_time: float, hot_b_time: float,
                                hot_c_time: float,
                                grid: BeatGrid, segments: list[Segment],
                                cbr=None) -> tuple[float | None, str]:
    """
    Wenn Hot A und Hot B zu nah sind (< 128 Beats):
    Analysiere aehnliche Tracks (CBR) um zu entscheiden ob Hot A
    verschoben oder weggelassen wird.

    Bei kurzen Tracks (Radio-Edits) wird meist auf Hot A verzichtet
    oder ein alternativer Phrasenwechsel gesucht.

    Returns: (neues hot_a_time oder None, Erklaerung)
    """
    # 1. Versuche alternativen Phrasenwechsel fuer Hot A
    #    Suche einen frueheren Break der weit genug entfernt ist
    for seg in segments:
        if seg.kind == "break":
            alt_time = snap_to_downbeat(seg.start_time, grid)
            if _check_min_distance(alt_time, hot_b_time, grid):
                return alt_time, "Alternativer Break (Abstandskorrektur)"

    # 2. CBR-Fallback: Pruefe ob Trainings-Tracks Hot A haben
    #    bei aehnlich kurzen Tracks
    if cbr is not None and hasattr(cbr, 'twins') and cbr.twins:
        # Bei Tracks mit wenig Twins mit Hot Cues: Skip Hot A
        n_with_hots = 0
        if cbr.hot_cue_pattern:
            n_with_hots = cbr.hot_cue_pattern.get("n_twins_with_hots", 0)
        if n_with_hots == 0:
            return None, "Kein Hot A (kurzer Track, keine Referenz)"

    # 3. Letzter Fallback: Hot A weglassen (typisch fuer Radio-Edits)
    total_beats = grid.count
    if total_beats < 400:  # < ~3min bei 128 BPM
        return None, "Kein Hot A (Radio-Edit, Track zu kurz)"

    # 4. Sonst: Hot A trotzdem setzen, aber naeher am Intro
    first_break = _find_break_start(segments)
    if first_break:
        return snap_to_downbeat(first_break.start_time, grid), \
            "Break Start (Abstand knapp)"

    return None, "Kein Hot A (kein geeigneter Ankerpunkt)"


# --- MIK-Hotspot-Selektion ---

def _select_mik_break(
    mik_spots: list[float],
    grid: BeatGrid,
    cbr=None,
    segments: list[Segment] | None = None,
) -> float | None:
    """
    Wählt den besten MIK-Cue als Break-Start-Kandidaten (Hot A).

    Schritt 1: Kandidaten = MIK-Cues in erster Trackhälfte (nach Intro-Guard 5%)
    Schritt 2: Segment-Tie-Breaker — bevorzuge Cues nahe 'break'/'intro' Segmenten
    Schritt 3: CBR-Tie-Breaker — wähle Cue dessen relative Position am nächsten
                an cbr.hot_cue_pattern['hot_a_relative_pos'] liegt
    Fallback:  Letzter Kandidat in erster Hälfte
    """
    if not mik_spots or grid.times is None or len(grid.times) == 0:
        return None

    duration = float(grid.times[-1])
    mid = float(grid.times[len(grid.times) // 2])
    intro_guard = duration * 0.05  # Keine Cues in ersten 5% (Intro)

    candidates = [t for t in sorted(mik_spots) if intro_guard < t < mid]
    if not candidates:
        return None

    # Schritt 2: Segment-Klassifizierung als erster Tie-Breaker
    if segments:
        def _seg_kind_at(t: float) -> str:
            kind = "unknown"
            for seg in segments:
                if seg.start_time <= t:
                    kind = seg.kind
                else:
                    break
            return kind

        break_candidates = [t for t in candidates
                            if _seg_kind_at(t) in ("break", "intro")]
        if break_candidates:
            candidates = break_candidates

    # Schritt 3: CBR hot_a_relative_pos als Tie-Breaker
    if cbr is not None:
        expected_rel = cbr.hot_cue_pattern.get("hot_a_relative_pos")
        if expected_rel and expected_rel > 0 and duration > 0:
            return min(candidates, key=lambda t: abs(t / duration - expected_rel))

    # Fallback: letzter Kandidat in erster Hälfte
    return candidates[-1]


def _select_mik_last_drop(mik_spots: list[float], grid: BeatGrid) -> float | None:
    """Wählt den letzten MIK-Cue in der zweiten Trackhälfte als Last Drop (Hot C)."""
    if not mik_spots or grid.times is None or len(grid.times) == 0:
        return None
    mid = float(grid.times[len(grid.times) // 2])
    second_half = sorted([t for t in mik_spots if t >= mid])
    return second_half[-1] if second_half else None


def _snap_to_mik_cue(
    time_sec: float,
    mik_spots: list[float],
    grid: BeatGrid,
    tolerance_beats: int = 4,
) -> float:
    """
    Snappt time_sec auf den nächsten MIK-Cue, falls innerhalb ±tolerance_beats.
    Gibt time_sec unverändert zurück wenn kein MIK-Cue in Reichweite.
    """
    if not mik_spots:
        return time_sec
    beat_dur = 60.0 / grid.bpm
    tol_sec = tolerance_beats * beat_dur
    closest = min(mik_spots, key=lambda t: abs(t - time_sec))
    if abs(closest - time_sec) <= tol_sec:
        return closest
    return time_sec


def _filter_memory_near_hot(
    memory_cues: list[CuePoint],
    hot_times: list[float],
    grid: BeatGrid,
    min_beats: int = 32,
) -> list[CuePoint]:
    """
    Entfernt Memory Cues die zu nah an einem Hot Cue liegen.
    Mindestabstand: min_beats Beats (BPM-abhaengig).
    """
    if not hot_times:
        return memory_cues
    beat_dur = 60.0 / grid.bpm
    min_sec = min_beats * beat_dur - beat_dur * 0.5  # halber Beat Toleranz
    return [
        mc for mc in memory_cues
        if not any(abs(mc.time_sec - ht) < min_sec for ht in hot_times)
    ]


def _filter_memory_near_end(
    memory_cues: list[CuePoint],
    grid: BeatGrid,
    min_end_beats: int = 32,
) -> list[CuePoint]:
    """
    Entfernt Memory Cues die zu nah am Track-Ende liegen.
    Regel: KEIN Memory Cue in den letzten N Beats des Tracks.
    Betrifft alle Quellen: Outro (Prio 5), MIK-Anker (Prio 3), PWAV-Anker (Prio 4).

    Args:
        min_end_beats: Mindestabstand vom Track-Ende in Beats (Default: 32 = 1 Bar).
    """
    if not memory_cues or grid.count <= 0 or len(grid.times) == 0:
        return memory_cues
    beat_dur = 60.0 / max(grid.bpm, 1.0)
    end_time = float(grid.times[-1])
    min_end_sec = min_end_beats * beat_dur
    return [mc for mc in memory_cues if mc.time_sec <= end_time - min_end_sec]


def _filter_memory_spacing(
    memory_cues: list[CuePoint],
    grid: BeatGrid,
    min_beats: int = 32,
) -> list[CuePoint]:
    """
    Entfernt Memory Cues die zu nah aneinander liegen.
    Mindestabstand: min_beats Beats (BPM-abhaengig).
    Hoechste Prioritaet (kleinste Zahl) hat Vorrang.
    """
    if len(memory_cues) <= 1:
        return memory_cues

    beat_dur = 60.0 / max(grid.bpm, 1.0)
    min_sec = min_beats * beat_dur - beat_dur * 0.5  # halber Beat Toleranz

    # Wichtigste Cues zuerst pruefen (nach Prio, dann Zeit)
    sorted_cues = sorted(memory_cues, key=lambda c: (c.priority, c.time_sec))

    kept: list[CuePoint] = []
    for cue in sorted_cues:
        if not any(abs(cue.time_sec - k.time_sec) < min_sec for k in kept):
            kept.append(cue)

    kept.sort(key=lambda c: c.time_sec)
    return kept


# --- Cue-Generierung ---

def generate_cues(
    analysis: TrackAnalysis,
    cbr=None,
    phrases: list[PhraseSegment] | None = None,
    learned_offsets: dict | None = None,
    pwav_candidates: dict | None = None,
) -> tuple[list[CuePoint], list[CueDecision]]:
    """
    Hauptfunktion: Generiert alle Cue-Punkte aus der Track-Analyse.
    v27: Quad-Check-Validierung (MIK x Phrase x Library x PWAV-ML) fuer Hot A und Hot C.

    Args:
        analysis:        TrackAnalysis mit Segments, Grid, Energy etc.
        cbr:             Optional CBRResult mit adaptiven Parametern.
        phrases:         Optional Liste von PSSI-PhraseSegments (aus phrase_reader).
                         None = akustische Segmente als Fallback-Phrase-Quelle.
        learned_offsets: Gelernte Offsets aus learning_db (Recursive Intelligence).
                         Keys: "hot_a_offset_ms", "hot_c_offset_ms"
        pwav_candidates: Optional PWAV-ML Kandidaten aus dj_validator.extract_candidates().
                         Dict mit Keys "hot_a", "hot_c", "memory" → Listen von Sek.-Zeiten.
                         None = kein PWAV-Modell vorhanden → Fallback auf Triple-Check.

    Returns:
        (cues, decisions): Sortierte CuePoint-Liste + Entscheidungen fuer Status-Report.
    """
    grid = analysis.grid
    segments = analysis.segments
    cues: list[CuePoint] = []

    if not segments:
        return cues, []

    # --- Adaptive Parameter aus CBR oder Defaults ---
    _conf = _cfg.load_config()
    hot_b_offset = _conf.get("hot_b_offset_beats", 32)
    cue_spacing  = 32

    if cbr is not None:
        cue_spacing = cbr.cue_spacing
        learned_b = (cbr.hot_cue_pattern or {}).get("hot_b_offset_beats", hot_b_offset)
        if 16 <= int(learned_b) <= 64:
            hot_b_offset = int(learned_b)

    # --- Basiswerte ---
    mik_spots: list[float] = getattr(analysis, 'mik_hotspots', []) or []
    used_mik_times: set[float] = set()

    duration    = float(grid.times[-1]) if len(grid.times) > 0 else 0.0
    mid_sec     = float(grid.times[len(grid.times) // 2]) if len(grid.times) > 1 else duration / 2.0
    intro_guard = duration * 0.05
    beat_dur    = 60.0 / max(grid.bpm, 1.0)

    # Gelernte Offsets: Recursive Intelligence hat Vorrang vor CBR-Slot
    _lo          = learned_offsets or {}
    learned_a_ms = _lo.get("hot_a_offset_ms") or getattr(cbr, 'hot_a_offset_ms', 0)
    learned_c_ms = _lo.get("hot_c_offset_ms") or getattr(cbr, 'hot_c_offset_ms', 0)

    # --- Kandidaten fuer Triple-Check ---
    mik_cands_a: list[float] = [t for t in sorted(mik_spots)
                                 if intro_guard < t < mid_sec]
    mik_cands_c: list[float] = [t for t in sorted(mik_spots) if t >= mid_sec]

    # MIK × Segment-Priorisierung: break/build-Cues zuerst fuer Hot A,
    # drop-Cues zuerst fuer Hot C. Verhindert falsche Kandidatenwahl bei
    # hoher MIK-Cue-Dichte (8 Cues pro Track, nur 1-2 sind Break/Drop).
    if segments and mik_cands_a:
        _intro_pref_mik = duration * 0.20 if duration > 0 else float('inf')
        def _mik_seg_prio_a(t: float) -> tuple[int, int]:
            pos_tier = 0 if t <= _intro_pref_mik else 1
            for seg in segments:
                if seg.start_time <= t < seg.end_time:
                    if seg.kind in ("break", "build"):
                        return (pos_tier, 0)
                    elif seg.kind in ("verse",):
                        return (pos_tier, 1)
                    elif seg.kind in ("intro",):
                        return (pos_tier, 2)
            return (pos_tier, 3)
        mik_cands_a.sort(key=_mik_seg_prio_a)

    if segments and mik_cands_c:
        def _mik_seg_prio_c(t: float) -> int:
            for seg in segments:
                if seg.start_time <= t < seg.end_time:
                    if seg.kind == "drop":
                        return 0
                    elif seg.kind in ("break", "build"):
                        return 1
            return 2
        mik_cands_c.sort(key=_mik_seg_prio_c)

    if phrases:
        # PSSI-Phrasen (primaere Quelle in v26.1)
        # Hot A: Down, Up (Build) und zweite+ Intro-Phrase ("Intro 2")
        _intro_count = 0
        _valid_intro_times: list[float] = []
        for p in phrases:
            if p.kind_name == "Intro":
                _intro_count += 1
                if _intro_count >= 2:
                    _valid_intro_times.append(p.time_start_sec)

        phrase_cands_a = [
            p.time_start_sec for p in phrases
            if (p.kind_name in ("Down", "Up")
                or p.time_start_sec in _valid_intro_times)
            and intro_guard < p.time_start_sec < mid_sec
        ]
        phrase_cands_c = [
            p.time_start_sec for p in phrases
            if p.kind_name in ("Chorus", "Up") and p.time_start_sec >= mid_sec
        ]
    else:
        # Akustische Segmente als Fallback-Phrase-Kandidaten
        phrase_cands_a = [
            seg.start_time for seg in segments
            if seg.kind in ("break", "build") and intro_guard < seg.start_time < mid_sec
        ]
        phrase_cands_c = [
            seg.start_time for seg in segments
            if seg.kind == "drop" and seg.start_time >= mid_sec
        ]
        # Outro-Drop-Erkennung: Wenn letzter Drop < 30s vor Trackende,
        # und ein vorheriger Drop mindestens gleich viel Energy hat,
        # schiebe den Outro-Drop ans Ende der Liste (= niedrigere Prioritaet).
        if len(phrase_cands_c) >= 2 and duration > 0:
            _last_drop_seg = next(
                (s for s in reversed(segments)
                 if s.kind == "drop" and s.start_time >= mid_sec),
                None,
            )
            if _last_drop_seg and (duration - _last_drop_seg.start_time) < duration * 0.20:
                _prev_drop_seg = next(
                    (s for s in reversed(segments)
                     if s.kind == "drop"
                     and s.start_time >= mid_sec
                     and s.start_time < _last_drop_seg.start_time),
                    None,
                )
                if (_prev_drop_seg
                        and _prev_drop_seg.energy_mean >= _last_drop_seg.energy_mean):
                    phrase_cands_c = [t for t in phrase_cands_c
                                     if t != _last_drop_seg.start_time]
                    phrase_cands_c.append(_last_drop_seg.start_time)
        # Energetische Nicht-Outro-Segmente als Fallback fuer Hot C
        if not phrase_cands_c:
            phrase_cands_c = [
                seg.start_time for seg in segments
                if seg.kind not in ("intro", "outro")
                and seg.start_time >= mid_sec
                and seg.energy_mean > 0.5
            ]
        # Visual Edge als letzter Fallback
        if not phrase_cands_c:
            visual = getattr(analysis, 'visual_edge_last_drop', None)
            if visual is not None:
                phrase_cands_c = [visual]

    # 20%-Intro-Praeferenz: Kandidaten in den ersten 20% des Tracks
    # werden vor spaeteren Kandidaten getestet.
    if phrase_cands_a and duration > 0:
        _intro_pref = duration * 0.20
        phrase_cands_a.sort(key=lambda t: (0 if t <= _intro_pref else 1, t))

    # CBR-Kandidaten (relative Position x Tracklänge)
    cbr_cand_a: float | None = None
    cbr_cand_c: float | None = None
    if cbr is not None and duration > 0:
        patt = cbr.hot_cue_pattern or {}
        rel_a = patt.get("hot_a_relative_pos")
        if rel_a and float(rel_a) > 0:
            cbr_cand_a = float(rel_a) * duration
        rel_c = patt.get("hot_c_relative_pos")
        if rel_c and float(rel_c) > 0:
            cbr_cand_c = float(rel_c) * duration

    # PWAV-ML Kandidaten (4. Quelle im Quad-Check)
    pwav_cands_a: list[float] = []
    pwav_cands_c: list[float] = []
    pwav_memory:  list[float] = []
    if pwav_candidates:
        pwav_cands_a = pwav_candidates.get("hot_a", [])
        pwav_cands_c = pwav_candidates.get("hot_c", [])
        pwav_memory  = pwav_candidates.get("memory", [])

    # ============================================
    # 1. Hot Cue C (3): "The Last Drop" — Quad-Check
    # ============================================
    decision_c = validate_hot_c(
        mik_candidates=mik_cands_c,
        phrase_candidates=phrase_cands_c,
        cbr_candidate=cbr_cand_c,
        grid=grid,
        learned_offset_ms=int(learned_c_ms or 0),
        pwav_candidates=pwav_cands_c,
        duration=duration,
    )

    hot_c_time: float | None = None
    if decision_c.action == "set" and decision_c.time_sec is not None:
        hot_c_time = snap_to_downbeat(decision_c.time_sec, grid)
        for mik_t in mik_spots:
            if abs(mik_t - decision_c.time_sec) <= beat_dur * 4:
                used_mik_times.add(round(mik_t, 2))
                break
        cues.append(CuePoint(
            time_sec=hot_c_time,
            kind=3,
            name="Hot C",
            comment="The Last Drop",
        ))

    # ============================================
    # 2. Hot Cue B (2): "The Setup" — abgeleitet
    # ============================================
    hot_b_time: float | None = None
    if hot_c_time is not None:
        hot_b_time = get_time_n_beats_before(hot_c_time, hot_b_offset, grid)

        if mik_spots:
            snapped = _snap_to_mik_cue(
                hot_b_time, mik_spots, grid,
                tolerance_beats=_MIK_SNAP_TOLERANCE_BEATS,
            )
            hot_b_time = snap_to_downbeat(snapped, grid)  # MIK-Spots sind ggf. off-grid
            for mik_t in mik_spots:
                if abs(mik_t - hot_b_time) < 0.1:
                    used_mik_times.add(round(mik_t, 2))
                    break

        if hot_b_time > grid.times[0]:
            cues.append(CuePoint(
                time_sec=hot_b_time,
                kind=2,
                name="Hot B",
                comment=f"Setup ({hot_b_offset}b vor Drop)",
            ))
        else:
            hot_b_time = None

    decision_b = CueDecision(
        kind=2,
        time_sec=hot_b_time,
        confidence=1.0 if hot_b_time is not None else 0.0,
        sources=["derived"] if hot_b_time is not None else [],
        rule_ok=hot_b_time is not None,
        reason=(f"Abgeleitet: {hot_b_offset} Beats vor Hot C"
                if hot_b_time is not None
                else "Kein Hot C — Hot B entfaellt"),
        action="set" if hot_b_time is not None else "skip",
        comment=f"Setup ({hot_b_offset}b vor Drop)" if hot_b_time is not None else "",
    )

    # ============================================
    # 3. Hot Cue A (1): "The Break" — Quad-Check
    # ============================================
    decision_a = validate_hot_a(
        mik_candidates=mik_cands_a,
        phrase_candidates=phrase_cands_a,
        cbr_candidate=cbr_cand_a,
        grid=grid,
        hot_b_time=hot_b_time,
        hot_c_time=hot_c_time,
        learned_offset_ms=int(learned_a_ms or 0),
        pwav_candidates=pwav_cands_a,
        duration=duration,
    )

    hot_a_time: float | None = None
    if decision_a.action == "set" and decision_a.time_sec is not None:
        snapped_a = snap_to_downbeat(decision_a.time_sec, grid)

        # Constraint-Re-Check nach Snap auf Downbeat
        if hot_b_time is not None and not _check_min_distance(
                snapped_a, hot_b_time, grid):
            decision_a = CueDecision(
                kind=1, time_sec=None, confidence=0.0, sources=[],
                rule_ok=False,
                reason="Constraint verletzt nach Snap (< min Beats zu Hot B)",
                action="skip", comment="",
            )
        elif hot_c_time is not None and snapped_a >= hot_c_time:
            decision_a = CueDecision(
                kind=1, time_sec=None, confidence=0.0, sources=[],
                rule_ok=False,
                reason="Constraint verletzt: Hot A liegt nach Hot C",
                action="skip", comment="",
            )
        else:
            _hot_times = {round(t, 2) for t in [hot_b_time, hot_c_time] if t}
            if round(snapped_a, 2) not in _hot_times:
                hot_a_time = snapped_a
                for mik_t in mik_spots:
                    if abs(mik_t - decision_a.time_sec) <= beat_dur * 4:
                        used_mik_times.add(round(mik_t, 2))
                        break
                cues.append(CuePoint(
                    time_sec=hot_a_time,
                    kind=1,
                    name="Hot A",
                    comment="The Break",
                ))

    # ============================================
    # 4. Memory Cues (max 10, 32-Beat-Raster)
    # ============================================
    memory_cues = _generate_memory_cues(
        analysis, phrases=phrases,
        hot_a_time=hot_a_time, hot_c_time=hot_c_time,
        pwav_memory=pwav_memory, cue_spacing=cue_spacing,
    )

    hot_times = [t for t in [hot_a_time, hot_b_time, hot_c_time] if t is not None]
    memory_cues = _filter_memory_near_hot(
        memory_cues, hot_times, grid, min_beats=_MEMORY_MIN_HOT_BEATS
    )
    memory_cues = _filter_memory_near_end(           # kein letzter Schlag
        memory_cues, grid,
        min_end_beats=_conf.get("memory_min_end_distance_beats", 32),
    )
    memory_cues = _filter_memory_spacing(
        memory_cues, grid, min_beats=_MEMORY_MIN_HOT_BEATS
    )
    memory_cues = _apply_memory_cue_limit(memory_cues)

    cues.extend(memory_cues)
    cues.sort(key=lambda c: (c.time_sec, c.kind))
    cues = _deduplicate_cues(cues)

    return cues, [decision_a, decision_b, decision_c]


# --- Memory Cue Generierung mit Priorisierung ---

def _is_kick_outro(segments: list[Segment]) -> bool:
    """
    Erkennt ob der Track ein Kick-Outro hat (Energie bleibt hoch, endet abrupt)
    oder ein Fade-Outro (Vocals/Melodie, Energie faellt graduell).
    """
    if not segments:
        return False
    outro = segments[-1] if segments[-1].kind == "outro" else None
    if outro is None:
        return False
    # Kick-Outro: Energie im Outro-Segment bleibt ueber 0.4
    return outro.energy_mean > 0.4


def _generate_memory_cues(analysis: TrackAnalysis,
                           phrases: list[PhraseSegment] | None = None,
                           hot_a_time: float | None = None,
                           hot_c_time: float | None = None,
                           pwav_memory: list[float] | None = None,
                           cue_spacing: int = 32) -> list[CuePoint]:
    """
    Generiert Memory Cues im 32-Beat-Raster (v28):

      1. Erster Downbeat
      2. Intro: Rueckwaerts von Hot A, max 3 Cues, 64er-Step bei langem Intro
      3. Outro: Vorwaerts von Hot C, NUR bei Kick-Outro, 32/64-Regel
      4. Struktur-Mitte: PSSI-Phrasen >= 32 Beats auf 32er-Raster
         ML-Cues nur als Validierung (im Comment vermerkt)

    Lieber keinen Memory Cue als einen falschen.
    MIK-Daten werden NICHT verwendet.
    """
    grid = analysis.grid
    segments = analysis.segments
    cues: list[CuePoint] = []

    if not segments or grid.count < 50:
        return cues

    beat_dur = 60.0 / max(grid.bpm, 1.0)
    first_downbeat = _first_downbeat_time(grid)
    first_db_beat = get_beat_index_at_time(first_downbeat, grid)
    last_beat = grid.count - 1

    # ML-Positionen fuer Validierung sammeln (auf 32-Raster gesnappt)
    ml_times: set[float] = set()
    if pwav_memory:
        for pm_t in pwav_memory:
            snapped = snap_to_phrase_boundary(pm_t, grid, n_beats=cue_spacing)
            ml_times.add(round(snapped, 2))

    def _ml_tag(time_sec: float) -> str:
        """Gibt ' (ML)' zurueck wenn eine ML-Position nahe liegt."""
        for ml_t in ml_times:
            if abs(ml_t - time_sec) < beat_dur * 4:
                return " (ML)"
        return ""

    # ---- 1. Erster Downbeat ----
    cues.append(CuePoint(
        time_sec=first_downbeat,
        kind=0,
        name="Memory 1",
        comment="Erster Downbeat",
        priority=1,
    ))

    # ---- 2. Intro: Rueckwaerts von Hot A ----
    if hot_a_time is not None:
        hot_a_beat = get_beat_index_at_time(hot_a_time, grid)
        intro_beats = hot_a_beat - first_db_beat
        n_slots = intro_beats // cue_spacing

        if n_slots >= 6:
            step = 64
        elif n_slots >= 1:
            step = cue_spacing  # 32
        else:
            step = 0

        if step > 0:
            max_intro_cues = 3
            positions: list[int] = []
            # Rueckwaerts von Hot A im Step-Abstand
            beat = hot_a_beat - step
            while beat > first_db_beat and len(positions) < max_intro_cues:
                # Auf Downbeat snappen (naechstes 4er-Vielfaches)
                snapped_beat = round(beat / 4) * 4
                if snapped_beat > first_db_beat and snapped_beat not in positions:
                    positions.append(snapped_beat)
                beat -= step
            positions.reverse()  # chronologisch

            for i, b in enumerate(positions):
                t = get_time_at_beat(b, grid)
                tag = _ml_tag(t)
                cues.append(CuePoint(
                    time_sec=t,
                    kind=0,
                    name=f"Intro {i + 1}",
                    comment=f"Intro Beat {b - first_db_beat}{tag}",
                    priority=2,
                ))

    # ---- 3. Outro: Vorwaerts von Hot C ----
    # Outro-Cues werden IMMER gesetzt — bei Fade-Outro endet die Kette
    # frueher (wo die Energie noch signifikant ist).
    if hot_c_time is not None:
        hot_c_beat = get_beat_index_at_time(hot_c_time, grid)
        kick_outro = _is_kick_outro(segments)

        # safe_end_beat: kein Memory Cue direkt am Track-Ende
        # Bei Fade-Outro zusaetzlich nicht tiefer als 32b ins Fade hinein
        safe_end_beat = last_beat - cue_spacing
        if not kick_outro:
            outro_seg = segments[-1] if segments and segments[-1].kind == "outro" else None
            if outro_seg:
                outro_start_beat = get_beat_index_at_time(outro_seg.start_time, grid)
                # Nicht tiefer als 32b ins Fade-Outro hinein
                fade_limit = outro_start_beat + cue_spacing
                safe_end_beat = min(fade_limit, safe_end_beat)

        outro_beats = safe_end_beat - hot_c_beat
        n_slots = max(0, outro_beats // cue_spacing)

        # Outro-Cues relativ zum Hot C Beat (nicht zum globalen Raster)
        def _outro_positions() -> list[int]:
            """Berechnet Outro-Cue-Positionen relativ zu Hot C."""
            pos: list[int] = []
            if n_slots >= 6:
                # Sehr langes Outro: alle 64 Beats
                b = hot_c_beat + 64
                while b <= safe_end_beat and b < grid.count:
                    pos.append(b)
                    b += 64
            elif n_slots == 5:
                # Sonderregel: Erster mit 64, Rest mit 32
                first = hot_c_beat + 64
                if first <= safe_end_beat:
                    pos.append(first)
                    b = first + cue_spacing
                    while b <= safe_end_beat and b < grid.count:
                        pos.append(b)
                        b += cue_spacing
            elif n_slots >= 1:
                # Normal: alle 32 Beats
                b = hot_c_beat + cue_spacing
                while b <= safe_end_beat and b < grid.count:
                    pos.append(b)
                    b += cue_spacing
            return pos

        outro_label = "Outro" if kick_outro else "Fade"
        for i, b in enumerate(_outro_positions()):
            t = get_time_at_beat(b, grid)
            tag = _ml_tag(t)
            cues.append(CuePoint(
                time_sec=t, kind=0,
                name=f"{outro_label} {i}",
                comment=f"{outro_label} +{b - hot_c_beat}b{tag}",
                priority=5,
            ))

    # ---- 4. Struktur-Mitte: PSSI-Phrasen direkt nutzen ----
    # Phrasen-Positionen werden direkt uebernommen (nicht auf globales 32-Raster
    # gesnappt), da die Track-Struktur intern versetzt sein kann.
    # Mindestlaenge: 16 Beats (halbe Phrase ist noch strukturell relevant).
    mid_start = hot_a_time if hot_a_time is not None else first_downbeat
    mid_end = hot_c_time if hot_c_time is not None else float(grid.times[-1])
    mid_start_beat = get_beat_index_at_time(mid_start, grid)
    mid_end_beat = get_beat_index_at_time(mid_end, grid)
    min_phrase_beats = 16  # Phrasen >= 16 Beats werden beruecksichtigt

    if phrases:
        for i, p in enumerate(phrases):
            if p.kind_name in ("Intro", "Outro"):
                continue

            # Phrase muss im Bereich Hot A..Hot C liegen
            if p.time_start_sec <= mid_start or p.time_start_sec >= mid_end:
                continue

            # Auf naechsten Downbeat snappen (4-Beat-Grenze)
            t = snap_to_downbeat(p.time_start_sec, grid)
            t_beat = get_beat_index_at_time(t, grid)

            # Phrase muss auf einem Downbeat liegen (mod 4 check)
            if t_beat % 4 > 1:
                continue

            # Phrase muss mindestens 16 Beats lang sein
            p_end = p.time_end_sec if hasattr(p, 'time_end_sec') else None
            if p_end is not None:
                p_beats = get_beat_index_at_time(p_end, grid) - t_beat
                if p_beats < min_phrase_beats:
                    continue

            # Duplikat-Check
            already_exists = any(
                abs(c.time_ms - int(round(t * 1000))) <= 500
                for c in cues
            )
            if already_exists:
                continue

            tag = _ml_tag(t)
            cues.append(CuePoint(
                time_sec=t,
                kind=0,
                name=f"Phrase {i + 1}",
                comment=f"{p.kind_name} Start{tag}",
                priority=4,
            ))
    else:
        # Fallback: alle 32 Beats ab Hot A gliedern
        if hot_a_time is not None and hot_c_time is not None:
            beat = mid_start_beat + cue_spacing
            step_idx = 0
            while beat < mid_end_beat:
                t = get_time_at_beat(beat, grid)
                tag = _ml_tag(t)
                already_exists = any(
                    abs(c.time_ms - int(round(t * 1000))) <= 500
                    for c in cues
                )
                if not already_exists:
                    cues.append(CuePoint(
                        time_sec=t, kind=0,
                        name=f"Struktur {step_idx + 1}",
                        comment=f"32b-Raster{tag}",
                        priority=4,
                    ))
                beat += cue_spacing
                step_idx += 1

    return cues


def _apply_memory_cue_limit(cues: list[CuePoint]) -> list[CuePoint]:
    """
    Begrenzt Memory Cues auf MAX_MEMORY_CUES (10).

    Garantierte Slots (werden IMMER behalten):
      Prio 1 = Erster Downbeat
      Prio 2 = Intro-Struktur (max 3)
      Prio 5 = Outro-Struktur (max 3)

    Verbleibende Slots fuellen sich mit:
      Prio 3 = ML-Ankerpunkte
      Prio 4 = Phrasen-Uebergaenge
    """
    if len(cues) <= MAX_MEMORY_CUES:
        return cues

    # Garantierte Cues: Downbeat (1), Intro (2), Outro (5)
    guaranteed = [c for c in cues if c.priority in (1, 2, 5)]
    fillable = [c for c in cues if c.priority not in (1, 2, 5)]

    # Fillable nach Prio sortieren (3 vor 4)
    fillable.sort(key=lambda c: (c.priority, c.time_sec))

    # Auffuellen bis MAX_MEMORY_CUES
    remaining = MAX_MEMORY_CUES - len(guaranteed)
    kept = guaranteed + fillable[:max(0, remaining)]

    # Zurueck in zeitliche Reihenfolge
    kept.sort(key=lambda c: c.time_sec)

    return kept


# --- Hilfsfunktionen ---

def _first_downbeat_time(grid: BeatGrid) -> float:
    """Findet den ersten Downbeat (Beat 1) im Grid."""
    dt = grid.downbeat_times
    if len(dt) > 0:
        return float(dt[0])
    return float(grid.times[0])


def _deduplicate_cues(cues: list[CuePoint],
                       tolerance_ms: int = 200) -> list[CuePoint]:
    """
    Entfernt Duplikate: Cues innerhalb von tolerance_ms werden zusammengefasst.
    Hot Cues haben Vorrang vor Memory Cues.
    """
    if len(cues) <= 1:
        return cues

    cues.sort(key=lambda c: c.time_sec)
    result: list[CuePoint] = []

    for cue in cues:
        merged = False
        for i, existing in enumerate(result):
            if abs(cue.time_ms - existing.time_ms) <= tolerance_ms:
                # Duplikat: Hot Cue > Memory Cue, hoehere Slots > niedrigere
                if cue.kind > existing.kind:
                    result[i] = cue
                merged = True
                break
        if not merged:
            result.append(cue)

    return result


# --- Ausgabe ---

def print_cues(cues: list[CuePoint], grid: BeatGrid) -> None:
    """Gibt die Cue-Liste formatiert aus."""
    print(f"\n{'='*60}")
    print(f"  Generierte Cue-Punkte ({len(cues)} total)")
    print(f"{'='*60}")

    hot_cues = [c for c in cues if c.kind > 0]
    mem_cues = [c for c in cues if c.kind == 0]

    if hot_cues:
        print(f"\n  Hot Cues ({len(hot_cues)}):")
        for c in sorted(hot_cues, key=lambda x: x.kind):
            m = int(c.time_sec // 60)
            s = c.time_sec % 60
            beat = get_beat_index_at_time(c.time_sec, grid)
            bar = beat // 4 + 1
            print(f"    Hot {chr(64 + c.kind):s}  [{m}:{s:05.2f}]  "
                  f"Bar {bar:3d}  — {c.comment}")

    if mem_cues:
        print(f"\n  Memory Cues ({len(mem_cues)}, Max {MAX_MEMORY_CUES}):")
        for c in mem_cues:
            m = int(c.time_sec // 60)
            s = c.time_sec % 60
            beat = get_beat_index_at_time(c.time_sec, grid)
            bar = beat // 4 + 1
            prio = f"P{c.priority}" if c.priority < 99 else ""
            print(f"    Mem    [{m}:{s:05.2f}]  "
                  f"Bar {bar:3d}  {prio:3s} — {c.comment}")

    # Hot A ↔ Hot B Abstandscheck anzeigen
    hot_a = next((c for c in hot_cues if c.kind == 1), None)
    hot_b = next((c for c in hot_cues if c.kind == 2), None)
    if hot_a and hot_b:
        a_beat = get_beat_index_at_time(hot_a.time_sec, grid)
        b_beat = get_beat_index_at_time(hot_b.time_sec, grid)
        dist = abs(b_beat - a_beat)
        status = "✓" if dist >= MIN_HOT_A_B_DISTANCE_BEATS else "✗"
        print(f"\n  Hot A ↔ Hot B: {dist} Beats "
              f"({dist // 4} Bars) {status} "
              f"(Min: {MIN_HOT_A_B_DISTANCE_BEATS})")

    print()
