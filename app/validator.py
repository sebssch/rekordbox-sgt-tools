"""
v26.1 Triple-Check Validator
Validiert jeden Cue unabhaengig gegen drei Quellen:

  Stufe 1: MIK x Phrase (innerhalb +-2 Beats Toleranz)
  Stufe 2: Library (CBR) bestaetigt eine Quelle
  Stufe 3: Kein Konsens → Exit-Strategie (kein Cue setzen)

Granulare Logik:
  - Jeder Hot Cue (A, B, C) wird separat geprueft
  - Teil-Erfolg: Hot A gesetzt auch wenn Hot C kein Konsens
  - Status: set / skip mit Begruendung fuer Terminal-Report

Hard Constraints (aus config.yaml):
  - Hot A: Erster Break/Down-Teil
  - Hot C: Zweiter Drop/Chorus-Teil (2. Haelfte)
  - Hot B: Abgeleitet — exakt N Beats vor Hot C
  - Memory Cues: Max 10, mind. 32 Beats Abstand
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.beatgrid import BeatGrid, get_beat_index_at_time
from app import config as _cfg


# --- Ergebnis-Datenmodell ---

@dataclass
class CueDecision:
    """Ergebnis der Validierung fuer einen einzelnen Cue."""
    kind: int                         # 1=Hot A, 2=Hot B, 3=Hot C, 0=Memory
    time_sec: float | None            # Errechnete Zeit (None = uebersprungen)
    confidence: float                 # 0.0 = kein Konsens, 1.0 = alle drei Quellen
    sources: list[str]                # z.B. ["mik", "phrase", "library"]
    rule_ok: bool                     # Hard Constraints erfuellt?
    reason: str                       # Erklaerung (Terminal + Log)
    action: str                       # "set" | "skip"
    comment: str = ""                 # Cue-Label (z.B. "The Break")

    @property
    def symbol(self) -> str:
        """Visuelles Symbol fuer Terminal-Ausgabe."""
        return "[green]✓[/green]" if self.action == "set" else "[red]✗[/red]"

    @property
    def sources_str(self) -> str:
        """Quellen als kompakter String (z.B. 'MIK+PHRASE')."""
        return "+".join(s.upper() for s in self.sources) if self.sources else "—"

    @property
    def kind_label(self) -> str:
        """Lesbarer Cue-Typ fuer Terminal-Ausgabe."""
        return {1: "Hot A", 2: "Hot B", 3: "Hot C"}.get(self.kind, f"Mem({self.kind})")


# --- Interne Helpers ---

def _beats_apart(t1: float, t2: float, grid: BeatGrid) -> float:
    """Abstand zweier Zeitpunkte in Beats."""
    beat_dur = 60.0 / max(grid.bpm, 1.0)
    return abs(t1 - t2) / beat_dur


def _closest_within(candidate: float, targets: list[float],
                     grid: BeatGrid, tol_beats: float) -> float | None:
    """
    Naechstes Ziel innerhalb der Beat-Toleranz.
    Gibt None zurueck wenn kein Target in Reichweite.
    """
    if not targets:
        return None
    best      = min(targets, key=lambda t: abs(t - candidate))
    best_dist = _beats_apart(candidate, best, grid)
    return best if best_dist <= tol_beats else None


def _compute_confidence(
    chosen: float,
    sources: list[str],
    ml_candidate: float | None,
    cbr_candidate: float | None,
    grid: BeatGrid,
    duration: float,
) -> float:
    """
    Berechnet einen Konfidenz-Score (0.0 - 1.0) fuer eine Cue-Entscheidung.

    Faktoren:
      - Quellen-Anzahl: 2+ Quellen = 0.5 Basis, 1 Quelle = 0.25 Basis
      - ML-Naehe: Wenn ML-Prediction existiert und nah dran → Bonus
      - CBR-Naehe: Wenn CBR existiert und nah dran → Bonus
    """
    beat_dur = 60.0 / max(grid.bpm, 1.0)

    # Basis: Quellen-Anzahl
    n_sources = len(sources)
    if n_sources >= 2:
        conf = 0.5
    elif n_sources == 1:
        conf = 0.25
    else:
        return 0.0

    # Bonus fuer MIK×Phrase (klassischer Konsens, hoechste Zuverlaessigkeit)
    if "mik" in sources and "phrase" in sources:
        conf += 0.15

    # ML-Naehe Bonus (max +0.25)
    if ml_candidate is not None and duration > 0:
        ml_dist_beats = abs(chosen - ml_candidate) / beat_dur
        if ml_dist_beats <= 4:
            conf += 0.25  # ML sehr nah → hohe Konfidenz
        elif ml_dist_beats <= 16:
            conf += 0.15  # ML relativ nah
        elif ml_dist_beats <= 32:
            conf += 0.05  # ML im Bereich

    # CBR-Naehe Bonus (max +0.10)
    if cbr_candidate is not None and duration > 0:
        cbr_dist_beats = abs(chosen - cbr_candidate) / beat_dur
        if cbr_dist_beats <= 8:
            conf += 0.10
        elif cbr_dist_beats <= 16:
            conf += 0.05

    return min(1.0, conf)


def _check_hot_a_constraints(t: float,
                               hot_b_time: float | None,
                               hot_c_time: float | None,
                               grid: BeatGrid) -> bool:
    """
    Prueft Hard Constraints fuer Hot Cue A:
      - mind. MIN_HOT_A_B_DISTANCE_BEATS Abstand zu Hot B
        (dynamisch: bei kurzen/langsamen Tracks auf track_beats // 4 reduziert,
         Minimum 64 Beats)
      - muss vor Hot C liegen
    """
    configured = _cfg.get("min_hot_a_b_distance_beats", 128)
    track_beats = grid.count if hasattr(grid, 'count') else len(grid.times)
    min_beats = max(64, min(configured, track_beats // 4))

    if hot_b_time is not None:
        idx_a = get_beat_index_at_time(t, grid)
        idx_b = get_beat_index_at_time(hot_b_time, grid)
        if abs(idx_b - idx_a) < min_beats:
            return False

    if hot_c_time is not None and t >= hot_c_time:
        return False

    return True


# --- Hot A Validator ---

def validate_hot_a(
    mik_candidates:    list[float],      # MIK-Cue-Zeiten in erster Haelfte (Sek.)
    phrase_candidates: list[float],      # Phrase-Break-Starts (Sek.)
    cbr_candidate:     float | None,     # CBR: hot_a_relative_pos * duration
    grid:              BeatGrid,
    hot_b_time:        float | None = None,
    hot_c_time:        float | None = None,
    learned_offset_ms: int = 0,          # Gelernter Offset (pre-correction)
    pwav_candidates:   list[float] | None = None,  # PWAV-ML Kandidaten (Sek.)
    duration:          float = 0.0,      # Track-Dauer fuer 20%-Intro-Praeferenz
) -> CueDecision:
    """
    Validiert Hot Cue A (The Break — erster Down/Break-Teil).

    Ablauf:
      Stufe 1: Alle MIK×Phrase Matches sammeln, per Score bewerten
               (Segment-Typ + CBR-Naehe + 20%-Intro-Bonus), besten waehlen.
      Stufe 2: CBR-Kandidat bestaetigt MIK oder Phrase (±4*tol Beats).
      Stufe 3: Kein Konsens → skip.

    Gelernter Offset wird vor der Validierung auf alle Kandidaten angewendet.
    """
    tol = float(_cfg.get("phrase_match_tolerance_beats", 2))

    # Pre-correction: gelernten Offset auf Kandidaten anwenden
    # P0: Offset auf ±2000ms begrenzen um negative Zeiten zu vermeiden
    _clamped_offset = max(-2000, min(2000, learned_offset_ms))
    offset_sec  = _clamped_offset / 1000.0
    mik_adj     = [t + offset_sec for t in mik_candidates]
    phrase_adj  = [t + offset_sec for t in phrase_candidates]

    chosen:  float | None = None
    sources: list[str]    = []

    # --- Stufe 1: Kandidaten-Pool aus MIK×Phrase + MIK×ML ---
    # MIK×Phrase Matches (klassischer Konsens)
    _mik_phrase_matches: list[tuple[float, list[str]]] = []
    for mik_t in mik_adj:
        if _closest_within(mik_t, phrase_adj, grid, tol) is not None:
            if _check_hot_a_constraints(mik_t, hot_b_time, hot_c_time, grid):
                _mik_phrase_matches.append((mik_t, ["mik", "phrase"]))

    # MIK×ML Matches (ML-Prediction bestaetigt MIK-Cue)
    _mik_ml_matches: list[tuple[float, list[str]]] = []
    if pwav_candidates:
        _pwav_t = pwav_candidates[0]
        for mik_t in mik_adj:
            # Kein Duplikat wenn bereits in MIK×Phrase
            if any(abs(mik_t - mp[0]) < 0.1 for mp in _mik_phrase_matches):
                continue
            if _closest_within(_pwav_t, [mik_t], grid, tol * 4) is not None:
                if _check_hot_a_constraints(mik_t, hot_b_time, hot_c_time, grid):
                    _mik_ml_matches.append((mik_t, ["mik", "pwav"]))

    # Phrase×ML Matches (funktioniert auch ohne MIK-Daten)
    _phrase_ml_matches: list[tuple[float, list[str]]] = []
    if pwav_candidates:
        _pwav_t = pwav_candidates[0]
        for ph_t in phrase_adj:
            # Kein Duplikat wenn bereits in MIK×Phrase oder MIK×ML
            if any(abs(ph_t - mp[0]) < 0.1 for mp in _mik_phrase_matches):
                continue
            if any(abs(ph_t - mp[0]) < 0.1 for mp in _mik_ml_matches):
                continue
            if _closest_within(_pwav_t, [ph_t], grid, tol * 4) is not None:
                if _check_hot_a_constraints(ph_t, hot_b_time, hot_c_time, grid):
                    _phrase_ml_matches.append((ph_t, ["phrase", "pwav"]))

    # Alle Kandidaten zusammenfuehren
    _all_candidates = _mik_phrase_matches + _mik_ml_matches + _phrase_ml_matches

    if _all_candidates:
        if len(_all_candidates) == 1:
            chosen = _all_candidates[0][0]
            sources = _all_candidates[0][1]
        else:
            # Score: niedrigerer Wert = besser
            cbr_adj_s1 = (cbr_candidate + offset_sec) if cbr_candidate is not None else None
            _intro_zone = duration * 0.20 if duration > 0 else float('inf')
            beat_dur = 60.0 / max(grid.bpm, 1.0)
            _ml_t = pwav_candidates[0] if pwav_candidates else None

            def _score_hot_a(entry: tuple[float, list[str]]) -> float:
                t, srcs = entry
                score = 0.0
                # ML-Naehe: staerkstes Signal (trainiert auf 4430 Tracks)
                if _ml_t is not None and duration > 0:
                    score += abs(t - _ml_t) / duration * 0.6
                # CBR-Naehe
                if cbr_adj_s1 is not None and duration > 0:
                    score += abs(t - cbr_adj_s1) / (duration * 2)
                # 20%-Intro-Bonus
                if t <= _intro_zone:
                    score -= 0.15
                # Mehr Quellen = besser (MIK×Phrase = 2 Quellen-Bonus)
                if "phrase" in srcs and "mik" in srcs:
                    score -= 0.05
                return score

            best = min(_all_candidates, key=_score_hot_a)
            chosen = best[0]
            sources = best[1]

    # --- Stufe 2: CBR bestaetigt eine Quelle ---
    if chosen is None and cbr_candidate is not None:
        cbr_adj = cbr_candidate + offset_sec

        # CBR + MIK
        mik_match = _closest_within(cbr_adj, mik_adj, grid, tol * 4)
        if mik_match is not None and _check_hot_a_constraints(mik_match, hot_b_time, hot_c_time, grid):
            chosen  = mik_match
            sources = ["mik", "library"]

        # CBR + Phrase
        if chosen is None:
            phrase_match = _closest_within(cbr_adj, phrase_adj, grid, tol * 4)
            if phrase_match is not None and _check_hot_a_constraints(phrase_match, hot_b_time, hot_c_time, grid):
                chosen  = phrase_match
                sources = ["phrase", "library"]

    # --- Stufe 2.5: PWAV-ML bestaetigt eine andere Quelle ---
    if chosen is None and pwav_candidates:
        for pwav_t in pwav_candidates:
            # PWAV × MIK
            mik_match = _closest_within(pwav_t, mik_adj, grid, tol * 2)
            if mik_match is not None and _check_hot_a_constraints(mik_match, hot_b_time, hot_c_time, grid):
                chosen  = mik_match
                sources = ["mik", "pwav"]
                break
            # PWAV × Phrase
            phrase_match = _closest_within(pwav_t, phrase_adj, grid, tol * 2)
            if phrase_match is not None and _check_hot_a_constraints(phrase_match, hot_b_time, hot_c_time, grid):
                chosen  = phrase_match
                sources = ["phrase", "pwav"]
                break
            # PWAV × CBR
            if cbr_candidate is not None:
                cbr_adj = cbr_candidate + offset_sec
                if abs(pwav_t - cbr_adj) <= tol * 4 * (60.0 / max(grid.bpm, 1.0)):
                    if _check_hot_a_constraints(pwav_t, hot_b_time, hot_c_time, grid):
                        chosen  = pwav_t
                        sources = ["pwav", "library"]
                        break

        # PWAV allein (niedrigste Konfidenz, aber besser als skip)
        if chosen is None:
            for pwav_t in pwav_candidates:
                if _check_hot_a_constraints(pwav_t, hot_b_time, hot_c_time, grid):
                    chosen  = pwav_t
                    sources = ["pwav"]
                    break

    # --- Stufe 3: MIK-only oder Phrase-only Fallback (besser als Skip) ---
    if chosen is None and mik_adj:
        # MIK allein: fruehesten Kandidaten nehmen der Constraints erfuellt
        for mik_t in mik_adj:
            if _check_hot_a_constraints(mik_t, hot_b_time, hot_c_time, grid):
                chosen  = mik_t
                sources = ["mik"]
                break

    if chosen is None and phrase_adj:
        for ph_t in phrase_adj:
            if _check_hot_a_constraints(ph_t, hot_b_time, hot_c_time, grid):
                chosen  = ph_t
                sources = ["phrase"]
                break

    # --- Stufe 4: Kein Konsens ---
    if chosen is None:
        return CueDecision(
            kind=1, time_sec=None, confidence=0.0, sources=[],
            rule_ok=False,
            reason="No Consensus: MIK, Phrase, Library und PWAV stimmen nicht überein",
            action="skip", comment="",
        )

    # P0: Negative Zeiten abfangen
    if chosen < 0:
        return CueDecision(
            kind=1, time_sec=None, confidence=0.0, sources=[],
            rule_ok=False,
            reason="Ungültige Zeit (negativ) — Offset-Problem",
            action="skip", comment="",
        )

    # Konfidenz berechnen
    _ml_t = pwav_candidates[0] if pwav_candidates else None
    _cbr_t = (cbr_candidate + offset_sec) if cbr_candidate is not None else None
    conf = _compute_confidence(chosen, sources, _ml_t, _cbr_t, grid, duration)

    # Schwellwert-Check: unter min_confidence → lieber nichts setzen
    _min_conf = float(_cfg.get("min_confidence", 0.0))
    if _min_conf > 0 and conf < _min_conf:
        return CueDecision(
            kind=1, time_sec=None, confidence=conf, sources=sources,
            rule_ok=True,
            reason=f"Low Confidence: {conf:.2f} < {_min_conf:.2f} ({'+'.join(s.upper() for s in sources)})",
            action="skip", comment="",
        )

    return CueDecision(
        kind=1, time_sec=chosen, confidence=conf,
        sources=sources, rule_ok=True,
        reason=f"Konsens: {'+'.join(s.upper() for s in sources)} (conf={conf:.2f})",
        action="set", comment="The Break",
    )


# --- Hot C Validator ---

def validate_hot_c(
    mik_candidates:    list[float],      # MIK-Cues in zweiter Haelfte (Sek.)
    phrase_candidates: list[float],      # Phrase-Drop-Starts in 2. Haelfte (Sek.)
    cbr_candidate:     float | None,     # CBR: hot_c_relative_pos * duration
    grid:              BeatGrid,
    learned_offset_ms: int = 0,
    pwav_candidates:   list[float] | None = None,  # PWAV-ML Kandidaten (Sek.)
    duration:          float = 0.0,      # Track-Dauer fuer Outro-Drop-Erkennung
) -> CueDecision:
    """
    Validiert Hot Cue C (The Last Drop — zweiter/letzter Drop/Chorus).

    Weniger strikt als Hot A:
      Stufe 1+2: wie Hot A
      Fallback:  Einzelne beste Quelle wird akzeptiert (Hot C ist kritisch)
    """
    tol = float(_cfg.get("phrase_match_tolerance_beats", 2))

    # P0: Offset auf ±2000ms begrenzen
    _clamped_offset = max(-2000, min(2000, learned_offset_ms))
    offset_sec  = _clamped_offset / 1000.0
    mik_adj     = [t + offset_sec for t in mik_candidates]
    phrase_adj  = [t + offset_sec for t in phrase_candidates]

    chosen:  float | None = None
    sources: list[str]    = []

    # --- Stufe 1: MIK × Phrase ---
    # Alle MIK-Cues die einen Phrase-Match haben sammeln,
    # dann bewerten: Nicht-Outro zuerst, CBR als Tiebreaker (nicht Hauptfaktor).
    _outro_thresh = duration * 0.80 if duration > 0 else float('inf')
    _mik_phrase_matches: list[float] = []
    for mik_t in mik_adj:
        if _closest_within(mik_t, phrase_adj, grid, tol) is not None:
            _mik_phrase_matches.append(mik_t)

    if _mik_phrase_matches:
        # P2: Nicht-Outro Drops zuerst filtern, CBR nur als Tiebreaker
        _non_outro = [t for t in _mik_phrase_matches if t < _outro_thresh]
        _pool = _non_outro if _non_outro else _mik_phrase_matches

        if cbr_candidate is not None and len(_pool) > 1:
            cbr_adj_s1 = cbr_candidate + offset_sec
            chosen = min(_pool, key=lambda t: abs(t - cbr_adj_s1))
        else:
            chosen = _pool[-1]  # Letzten (spaetesten) nicht-Outro Drop nehmen
        sources = ["mik", "phrase"]

    # --- Stufe 1b: Phrase × ML (funktioniert auch ohne MIK) ---
    if chosen is None and pwav_candidates and phrase_adj:
        _phrase_ml_c: list[float] = []
        for pwav_t in pwav_candidates:
            phrase_match = _closest_within(pwav_t, phrase_adj, grid, tol * 4)
            if phrase_match is not None:
                _phrase_ml_c.append(phrase_match)
        if _phrase_ml_c:
            _non_outro_pml = [t for t in _phrase_ml_c if t < _outro_thresh]
            _pool_pml = _non_outro_pml if _non_outro_pml else _phrase_ml_c
            chosen = _pool_pml[-1]
            sources = ["phrase", "pwav"]

    # --- Stufe 1.5: PWAV × MIK oder PWAV × Phrase (vor CBR) ---
    # Nur fuer Hot C: Modell-Accuracy 71% bei ±8 Beats
    # tol * 4 = 8 Beats — entspricht ~1,4 Bins der PWAV-Map (Bin ≈ 2,66 s)
    # Iteration in Modell-Konfidenz-Reihenfolge (hoechste P zuerst, aus dj_validator)
    if chosen is None and pwav_candidates:
        for pwav_t in pwav_candidates:
            # PWAV × MIK
            mik_match = _closest_within(pwav_t, mik_adj, grid, tol * 4)
            if mik_match is not None:
                chosen  = mik_match
                sources = ["mik", "pwav"]
                break
            # PWAV × Phrase
            phrase_match = _closest_within(pwav_t, phrase_adj, grid, tol * 4)
            if phrase_match is not None:
                chosen  = phrase_match
                sources = ["phrase", "pwav"]
                break

    # --- Stufe 2: CBR bestaetigt ---
    if chosen is None and cbr_candidate is not None:
        cbr_adj = cbr_candidate + offset_sec

        mik_match = _closest_within(cbr_adj, mik_adj, grid, tol * 4)
        if mik_match is not None:
            chosen  = mik_match
            sources = ["mik", "library"]

        if chosen is None:
            phrase_match = _closest_within(cbr_adj, phrase_adj, grid, tol * 4)
            if phrase_match is not None:
                chosen  = phrase_match
                sources = ["phrase", "library"]

    # --- Stufe 2.5: PWAV-ML bestaetigt eine andere Quelle ---
    # Iteration in Modell-Konfidenz-Reihenfolge (hoechste P zuerst)
    if chosen is None and pwav_candidates:
        for pwav_t in pwav_candidates:
            # PWAV × MIK
            mik_match = _closest_within(pwav_t, mik_adj, grid, tol * 2)
            if mik_match is not None:
                chosen  = mik_match
                sources = ["mik", "pwav"]
                break
            # PWAV × Phrase
            phrase_match = _closest_within(pwav_t, phrase_adj, grid, tol * 2)
            if phrase_match is not None:
                chosen  = phrase_match
                sources = ["phrase", "pwav"]
                break
            # PWAV × CBR
            if cbr_candidate is not None:
                cbr_adj = cbr_candidate + offset_sec
                if abs(pwav_t - cbr_adj) <= tol * 4 * (60.0 / max(grid.bpm, 1.0)):
                    chosen  = pwav_t
                    sources = ["pwav", "library"]
                    break

    # --- Fallback: Einzelne beste Quelle (Hot C ist zu wichtig um zu ueberspringen) ---
    if chosen is None:
        # PWAV allein hat Vorrang vor reinem MIK/Phrase-Fallback
        if pwav_candidates:
            chosen  = pwav_candidates[0]   # PWAV-Kandidat mit hoechster Modell-Konfidenz
            sources = ["pwav"]
        elif mik_adj:
            # CBR als Anker, sonst Nicht-Outro bevorzugen
            if cbr_candidate is not None and len(mik_adj) > 1:
                cbr_adj_fb = cbr_candidate + offset_sec
                chosen = min(mik_adj, key=lambda t: abs(t - cbr_adj_fb))
            else:
                _non_outro_m = [t for t in mik_adj if t < _outro_thresh]
                chosen = _non_outro_m[-1] if _non_outro_m else mik_adj[-1]
            sources = ["mik"]
        elif phrase_adj:
            # CBR als Anker, sonst Nicht-Outro bevorzugen
            if cbr_candidate is not None and len(phrase_adj) > 1:
                cbr_adj_fb = cbr_candidate + offset_sec
                chosen = min(phrase_adj, key=lambda t: abs(t - cbr_adj_fb))
            else:
                _non_outro_p = [t for t in phrase_adj if t < _outro_thresh]
                chosen = _non_outro_p[-1] if _non_outro_p else phrase_adj[-1]
            sources = ["phrase"]
        elif cbr_candidate is not None:
            chosen  = cbr_candidate + offset_sec
            sources = ["library"]

    if chosen is None:
        return CueDecision(
            kind=3, time_sec=None, confidence=0.0, sources=[],
            rule_ok=False, reason="Keine Quellen fuer Hot C verfuegbar",
            action="skip", comment="",
        )

    # P0: Negative Zeiten abfangen
    if chosen < 0:
        return CueDecision(
            kind=3, time_sec=None, confidence=0.0, sources=[],
            rule_ok=False, reason="Ungültige Zeit (negativ) — Offset-Problem",
            action="skip", comment="",
        )

    # Konfidenz berechnen
    _ml_t = pwav_candidates[0] if pwav_candidates else None
    _cbr_t = (cbr_candidate + offset_sec) if cbr_candidate is not None else None
    conf = _compute_confidence(chosen, sources, _ml_t, _cbr_t, grid, duration)

    # Schwellwert-Check
    _min_conf = float(_cfg.get("min_confidence", 0.0))
    if _min_conf > 0 and conf < _min_conf:
        return CueDecision(
            kind=3, time_sec=None, confidence=conf, sources=sources,
            rule_ok=True,
            reason=f"Low Confidence: {conf:.2f} < {_min_conf:.2f} ({'+'.join(s.upper() for s in sources)})",
            action="skip", comment="",
        )

    return CueDecision(
        kind=3, time_sec=chosen, confidence=conf,
        sources=sources, rule_ok=True,
        reason=f"Konsens: {'+'.join(s.upper() for s in sources)} (conf={conf:.2f})",
        action="set", comment="The Last Drop",
    )


# --- Report Builder ---

def build_status_report(decisions: list[CueDecision], grid: BeatGrid | None = None) -> list[str]:
    """
    Erstellt einen Terminal-Report der Cue-Entscheidungen.

    Returns:
        Liste von Rich-formatierten Zeilen fuer die Ausgabe in batch.py.
    """
    lines = []
    for d in decisions:
        if d.time_sec is not None:
            m = int(d.time_sec // 60)
            s = d.time_sec % 60
            time_str = f"[green]{m}:{s:05.2f}[/green]"
        else:
            time_str = "[dim]  —  [/dim]"

        label   = d.kind_label
        sources = f"[dim][{d.sources_str}][/dim]" if d.sources else "[dim][—][/dim]"

        if d.action == "set":
            line = (
                f"  {d.symbol} {label:<6}  {time_str}  "
                f"{sources:<22}  [white]{d.comment}[/white]"
            )
        else:
            line = (
                f"  {d.symbol} {label:<6}  {time_str}  "
                f"{sources:<22}  [dim]{d.reason[:50]}[/dim]"
            )
        lines.append(line)
    return lines
