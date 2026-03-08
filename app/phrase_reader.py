"""
v26.1 Rekordbox Phrase Reader
Liest PSSI-Phrasen aus ANLZ .EXT-Dateien (Rekordbox Phrase Analysis).
Konvertiert 1-basierte Beat-Nummern in Timestamps via BeatGrid.

Rekordbox PSSI Phrase-Typen (mood=1 Energetic):
  1 = Intro
  2 = Up      (Buildup / aufsteigende Energie)
  3 = Down    (Break / niedrige Energie)
  5 = Chorus  (Drop / Haupt-Energie-Peak)
  6 = Outro

Einsatz: Stufe 1 des Triple-Check-Validators (MIK × Phrase → Konsens).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# --- Rekordbox Phrase-Type-Mapping ---

PSSI_KIND_NAMES: dict[int, str] = {
    1: "Intro",
    2: "Up",
    3: "Down",     # Break / Low-Energy
    4: "Chorus",   # Selten, mood-abhaengige Variante
    5: "Chorus",   # Haupt-Drop / High-Energy
    6: "Outro",
}

# Mapping zu internen Segment-Rollen (kompatibel mit v3/segments.py)
PSSI_TO_ROLE: dict[str, str] = {
    "Intro":  "intro",
    "Up":     "drop",
    "Down":   "break",
    "Chorus": "drop",
    "Outro":  "outro",
}


@dataclass
class PhraseSegment:
    """Ein Rekordbox PSSI-Phrase-Segment."""
    beat_start: int          # 1-basierte Beat-Zahl (PSSI-Format)
    beat_end: int            # Exklusives Ende (= naechster Beat-Start oder end_beat)
    kind: int                # PSSI kind code (1-6)
    kind_name: str           # "Intro", "Up", "Down", "Chorus", "Outro"
    role: str                # Interne Rolle: "intro", "drop", "break", "outro"
    time_start_sec: float    # Zeitlicher Anfang (Sekunden)
    time_end_sec: float      # Zeitliches Ende (Sekunden)

    def __repr__(self) -> str:
        m = int(self.time_start_sec // 60)
        s = self.time_start_sec % 60
        return f"PhraseSegment([{m}:{s:05.2f}] {self.kind_name} beats={self.beat_start}-{self.beat_end})"


# --- Hauptfunktion ---

def read_phrases(anlz_ext_path: str, grid) -> list[PhraseSegment]:
    """
    Liest PSSI-Phrasen aus einer ANLZ .EXT-Datei.

    Args:
        anlz_ext_path: Pfad zur .EXT-Datei (von beatgrid.get_anlz_path())
        grid: BeatGrid-Objekt (fuer Beat→Zeit-Konvertierung)

    Returns:
        Sortierte Liste von PhraseSegment-Objekten.
        Leere Liste wenn keine Phrasen vorhanden oder Fehler aufgetreten.
    """
    if not anlz_ext_path or not os.path.exists(anlz_ext_path):
        return []

    try:
        from pyrekordbox.anlz import AnlzFile
        anlz = AnlzFile.parse_file(anlz_ext_path)
    except Exception:
        return []

    pssi_tag = None
    for tag in anlz.tags:
        if tag.type == "PSSI":
            pssi_tag = tag
            break

    if pssi_tag is None:
        return []

    try:
        content  = pssi_tag.content
        entries  = list(content.entries)
        end_beat = int(content.end_beat)
    except Exception:
        return []

    if not entries:
        return []

    def _beat_to_time(beat_num: int) -> float:
        """Konvertiert 1-basierte Beat-Zahl in Sekunden via BeatGrid."""
        # grid.times ist 0-basiert (grid.times[0] = Beat 1)
        idx = max(0, min(int(beat_num) - 1, len(grid.times) - 1))
        return float(grid.times[idx])

    segments: list[PhraseSegment] = []
    for i, entry in enumerate(entries):
        b_start = int(entry.beat)
        # End-Beat = naechster Eintrag oder Gesamt-End
        b_end = int(entries[i + 1].beat) if (i + 1) < len(entries) else end_beat

        kind      = int(entry.kind)
        kind_name = PSSI_KIND_NAMES.get(kind, f"Kind{kind}")
        role      = PSSI_TO_ROLE.get(kind_name, "drop")

        segments.append(PhraseSegment(
            beat_start     = b_start,
            beat_end       = b_end,
            kind           = kind,
            kind_name      = kind_name,
            role           = role,
            time_start_sec = _beat_to_time(b_start),
            time_end_sec   = _beat_to_time(b_end),
        ))

    return segments


# --- Selektions-Helpers fuer Cue-Logik ---

def get_break_phrases(phrases: list[PhraseSegment]) -> list[PhraseSegment]:
    """Alle 'Down' (Break) Phrasen — Hot A Kandidaten."""
    return [p for p in phrases if p.kind_name == "Down"]


def get_drop_phrases(phrases: list[PhraseSegment]) -> list[PhraseSegment]:
    """Alle 'Chorus'/'Up' (Drop/Energie) Phrasen — Hot C Kandidaten."""
    return [p for p in phrases if p.kind_name in ("Chorus", "Up")]


def first_break_after_intro(phrases: list[PhraseSegment]) -> PhraseSegment | None:
    """
    Erster Break nach dem Intro → Hot A Kandidat.
    Fallback: erster Break ueberhaupt wenn kein Intro vorhanden.
    """
    past_intro = False
    for p in phrases:
        if p.kind_name == "Intro":
            past_intro = True
            continue
        if past_intro and p.kind_name == "Down":
            return p

    # Fallback ohne Intro
    for p in phrases:
        if p.kind_name == "Down":
            return p
    return None


def last_drop_second_half(phrases: list[PhraseSegment], grid) -> PhraseSegment | None:
    """
    Letzter Drop in der zweiten Haelfte → Hot C Kandidat.
    Beruecksichtigt nur 'Chorus' oder 'Up' Phrasen ab Trackmitte.
    """
    if not phrases or len(grid.times) == 0:
        return None

    mid_sec = float(grid.times[len(grid.times) // 2])
    drops   = [
        p for p in phrases
        if p.kind_name in ("Chorus", "Up") and p.time_start_sec >= mid_sec
    ]
    return drops[-1] if drops else None


def phrase_starts_near_time(phrases: list[PhraseSegment],
                             time_sec: float,
                             grid,
                             tolerance_beats: float = 2.0) -> PhraseSegment | None:
    """
    Sucht den naechsten Phrase-Start innerhalb der Toleranz.
    Gibt None zurueck wenn kein Match.
    """
    if not phrases or grid.bpm <= 0:
        return None

    beat_dur = 60.0 / grid.bpm
    tol_sec  = tolerance_beats * beat_dur

    best = None
    best_dist = float("inf")
    for p in phrases:
        dist = abs(p.time_start_sec - time_sec)
        if dist < best_dist:
            best_dist = dist
            best = p

    if best is not None and best_dist <= tol_sec:
        return best
    return None
