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


def _check_hot_a_constraints(t: float,
                               hot_b_time: float | None,
                               hot_c_time: float | None,
                               grid: BeatGrid) -> bool:
    """
    Prueft Hard Constraints fuer Hot Cue A:
      - mind. MIN_HOT_A_B_DISTANCE_BEATS Abstand zu Hot B
      - muss vor Hot C liegen
    """
    min_beats = _cfg.get("min_hot_a_b_distance_beats", 128)

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
) -> CueDecision:
    """
    Validiert Hot Cue A (The Break — erster Down/Break-Teil).

    Ablauf:
      Stufe 1: Jeder MIK-Kandidat wird gegen Phrase-Starts (±tol Beats) geprueft.
               Erstes Match das Hard Constraints besteht → gesetzt.
      Stufe 2: CBR-Kandidat bestaetigt MIK oder Phrase (±4*tol Beats).
      Stufe 3: Kein Konsens → skip.

    Gelernter Offset wird vor der Validierung auf alle Kandidaten angewendet.
    """
    tol = float(_cfg.get("phrase_match_tolerance_beats", 2))

    # Pre-correction: gelernten Offset auf Kandidaten anwenden
    offset_sec  = learned_offset_ms / 1000.0
    mik_adj     = [t + offset_sec for t in mik_candidates]
    phrase_adj  = [t + offset_sec for t in phrase_candidates]

    chosen:  float | None = None
    sources: list[str]    = []

    # --- Stufe 1: MIK × Phrase ---
    for mik_t in mik_adj:
        if _closest_within(mik_t, phrase_adj, grid, tol) is not None:
            if _check_hot_a_constraints(mik_t, hot_b_time, hot_c_time, grid):
                chosen  = mik_t
                sources = ["mik", "phrase"]
                break

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

    # --- Stufe 3: Kein Konsens ---
    if chosen is None:
        return CueDecision(
            kind=1, time_sec=None, confidence=0.0, sources=[],
            rule_ok=False,
            reason="No Consensus: MIK, Phrase und Library stimmen nicht überein",
            action="skip", comment="",
        )

    return CueDecision(
        kind=1, time_sec=chosen, confidence=len(sources) / 3.0,
        sources=sources, rule_ok=True,
        reason=f"Konsens: {'+'.join(s.upper() for s in sources)}",
        action="set", comment="The Break",
    )


# --- Hot C Validator ---

def validate_hot_c(
    mik_candidates:    list[float],      # MIK-Cues in zweiter Haelfte (Sek.)
    phrase_candidates: list[float],      # Phrase-Drop-Starts in 2. Haelfte (Sek.)
    cbr_candidate:     float | None,     # CBR: hot_c_relative_pos * duration
    grid:              BeatGrid,
    learned_offset_ms: int = 0,
) -> CueDecision:
    """
    Validiert Hot Cue C (The Last Drop — zweiter/letzter Drop/Chorus).

    Weniger strikt als Hot A:
      Stufe 1+2: wie Hot A
      Fallback:  Einzelne beste Quelle wird akzeptiert (Hot C ist kritisch)
    """
    tol = float(_cfg.get("phrase_match_tolerance_beats", 2))

    offset_sec  = learned_offset_ms / 1000.0
    mik_adj     = [t + offset_sec for t in mik_candidates]
    phrase_adj  = [t + offset_sec for t in phrase_candidates]

    chosen:  float | None = None
    sources: list[str]    = []

    # --- Stufe 1: MIK × Phrase (von hinten = letzter Match zuerst) ---
    for mik_t in sorted(mik_adj, reverse=True):
        if _closest_within(mik_t, phrase_adj, grid, tol) is not None:
            chosen  = mik_t
            sources = ["mik", "phrase"]
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

    # --- Fallback: Einzelne beste Quelle (Hot C ist zu wichtig um zu ueberspringen) ---
    if chosen is None:
        if mik_adj:
            chosen  = mik_adj[-1]   # Letzter MIK-Cue in 2. Haelfte
            sources = ["mik"]
        elif phrase_adj:
            chosen  = phrase_adj[-1]
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

    return CueDecision(
        kind=3, time_sec=chosen, confidence=len(sources) / 3.0,
        sources=sources, rule_ok=True,
        reason=f"Konsens: {'+'.join(s.upper() for s in sources)}",
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
