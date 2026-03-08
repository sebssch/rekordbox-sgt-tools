"""
v26 Feedback-Loop — Korrektur-Erkennung
Vergleicht AutoCue-Vorhersagen (predictions.jsonl) mit dem aktuellen
Stand in der Rekordbox-DB. Erkennt was der User geaendert hat:
verschoben, geloescht, behalten, manuell hinzugefuegt.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

from app.beatgrid import open_db
from app import config as _cfg


# --- Pfade ---

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTIONS_LOG = os.path.join(_BASE_DIR, "data", "predictions.jsonl")


# --- Datenmodelle ---

@dataclass
class CueCorrection:
    """Eine einzelne Cue-Korrektur."""
    content_id: str
    cue_kind: int               # 0=Memory, 1=Hot A, 2=Hot B, 3=Hot C
    cue_comment: str            # Original v3-Comment
    action: str                 # "kept", "moved", "deleted"
    predicted_ms: int           # Was v3 vorhergesagt hat
    actual_ms: int | None       # Wo der User die Cue hat (None=deleted)
    delta_ms: int               # Verschiebung in ms (0=kept)
    delta_beats: float          # Verschiebung in Beats


@dataclass
class TrackFeedback:
    """Feedback fuer einen einzelnen Track."""
    content_id: str
    title: str
    bpm: float
    timestamp: str              # Wann v3 den Track verarbeitet hat
    cbr_spacing: int            # Was CBR vorgeschlagen hat
    corrections: list[CueCorrection] = field(default_factory=list)
    n_kept: int = 0
    n_moved: int = 0
    n_deleted: int = 0
    n_user_added: int = 0       # Nicht-v3-Cues die der User hinzugefuegt hat
    accuracy: float = 0.0       # n_kept / total_predicted


# --- Prediction Log laden ---

def _load_predictions() -> dict[str, dict]:
    """
    Laedt predictions.jsonl und gibt ein Dict zurueck,
    gruppiert nach content_id (neuester Eintrag gewinnt).
    """
    predictions = {}

    if not os.path.isfile(_PREDICTIONS_LOG):
        return predictions

    with open(_PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cid = entry.get("content_id", "")
                if cid:
                    # Neuerer Eintrag ueberschreibt aelteren
                    predictions[cid] = entry
            except json.JSONDecodeError:
                continue

    return predictions


# --- Cue-Matching ---

def _match_hot_cue(predicted_cue: dict, actual_cues: list,
                   bpm: float) -> CueCorrection:
    """
    Matcht eine vorhergesagte Hot Cue (Kind 1-3) mit dem DB-Stand.
    Es gibt nur 1 Cue pro Kind-Slot → direkter Vergleich.
    """
    kind = predicted_cue["kind"]
    pred_ms = predicted_cue["time_ms"]
    comment = predicted_cue.get("comment", "")
    content_id = predicted_cue.get("_content_id", "")

    # Suche Cue gleichen Kinds in DB
    match = None
    for cue in actual_cues:
        if cue.Kind == kind and cue.InMsec is not None:
            match = cue
            break

    if match is None:
        # Hot Cue geloescht oder Kind geaendert
        return CueCorrection(
            content_id=content_id,
            cue_kind=kind,
            cue_comment=comment,
            action="deleted",
            predicted_ms=pred_ms,
            actual_ms=None,
            delta_ms=0,
            delta_beats=0.0,
        )

    actual_ms = match.InMsec
    delta_ms = actual_ms - pred_ms
    delta_beats = _ms_to_beats(abs(delta_ms), bpm)

    # Tolerance: ±500ms = "kept"
    if abs(delta_ms) <= 500:
        action = "kept"
    else:
        action = "moved"

    return CueCorrection(
        content_id=content_id,
        cue_kind=kind,
        cue_comment=comment,
        action=action,
        predicted_ms=pred_ms,
        actual_ms=actual_ms,
        delta_ms=delta_ms,
        delta_beats=delta_beats if action == "moved" else 0.0,
    )


def _match_memory_cues(predicted_mems: list[dict], actual_mems: list,
                       bpm: float, content_id: str) -> list[CueCorrection]:
    """
    Matcht vorhergesagte Memory Cues mit dem DB-Stand.
    Greedy-Matching nach geringster Zeitdistanz.
    """
    corrections = []
    used_actual = set()

    # Sortiere beide nach Zeit
    pred_sorted = sorted(predicted_mems, key=lambda c: c["time_ms"])
    actual_sorted = sorted(
        [c for c in actual_mems if c.InMsec is not None],
        key=lambda c: c.InMsec,
    )

    for pred in pred_sorted:
        pred_ms = pred["time_ms"]
        comment = pred.get("comment", "")

        # Finde naechste unbenutzte Actual-Cue
        best_idx = None
        best_dist = float("inf")

        for j, act in enumerate(actual_sorted):
            if j in used_actual:
                continue
            dist = abs(act.InMsec - pred_ms)
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx is not None and best_dist <= 30000:
            # Match gefunden (max 30s Abstand)
            used_actual.add(best_idx)
            actual_ms = actual_sorted[best_idx].InMsec
            delta_ms = actual_ms - pred_ms

            if abs(delta_ms) <= 500:
                action = "kept"
                delta_beats = 0.0
            else:
                action = "moved"
                delta_beats = _ms_to_beats(abs(delta_ms), bpm)

            corrections.append(CueCorrection(
                content_id=content_id,
                cue_kind=0,
                cue_comment=comment,
                action=action,
                predicted_ms=pred_ms,
                actual_ms=actual_ms,
                delta_ms=delta_ms,
                delta_beats=delta_beats,
            ))
        else:
            # Keine passende Cue in DB → geloescht
            corrections.append(CueCorrection(
                content_id=content_id,
                cue_kind=0,
                cue_comment=comment,
                action="deleted",
                predicted_ms=pred_ms,
                actual_ms=None,
                delta_ms=0,
                delta_beats=0.0,
            ))

    return corrections


def _ms_to_beats(ms: int, bpm: float) -> float:
    """Konvertiert Millisekunden in Beats."""
    if bpm <= 0:
        return 0.0
    return (ms / 1000.0) * (bpm / 60.0)


# --- Feedback sammeln ---

def collect_feedback(db=None) -> list[TrackFeedback]:
    """
    Scannt alle v3-verarbeiteten Tracks und vergleicht
    Vorhersagen mit aktuellem DB-Stand.

    Returns:
        Liste von TrackFeedback-Objekten
    """
    predictions = _load_predictions()

    if not predictions:
        print("  Keine Vorhersagen in predictions.jsonl gefunden.")
        print("  Verarbeite zuerst Tracks mit dry_run=False.")
        return []

    if db is None:
        db = open_db()

    results = []

    for cid, pred in predictions.items():
        content = db.get_content(ID=cid)
        if content is None:
            continue

        bpm = pred.get("bpm", 0.0)
        title = pred.get("title", "")
        timestamp = pred.get("timestamp", "")
        cbr_data = pred.get("cbr", {})
        cbr_spacing = cbr_data.get("cue_spacing", 32)
        predicted_cues = pred.get("cues", [])

        if not predicted_cues:
            continue

        # Aktuelle Cues aus DB
        actual_cues = list(content.Cues) if content.Cues else []

        # Aufteilen: Hot Cues vs. Memory Cues
        pred_hots = [c for c in predicted_cues if c["kind"] > 0]
        pred_mems = [c for c in predicted_cues if c["kind"] == 0]
        actual_mems = [c for c in actual_cues if c.Kind == 0]

        corrections = []

        # Hot Cues matchen (1 pro Slot)
        for ph in pred_hots:
            ph["_content_id"] = cid
            corr = _match_hot_cue(ph, actual_cues, bpm)
            corrections.append(corr)

        # Memory Cues matchen (greedy)
        mem_corrections = _match_memory_cues(
            pred_mems, actual_mems, bpm, cid)
        corrections.extend(mem_corrections)

        # Statistiken
        n_kept = sum(1 for c in corrections if c.action == "kept")
        n_moved = sum(1 for c in corrections if c.action == "moved")
        n_deleted = sum(1 for c in corrections if c.action == "deleted")

        # User-added: Cues die nicht von AutoCue stammen und nicht in Vorhersagen sind
        n_user_added = sum(
            1 for c in actual_cues
            if not _cfg.is_autocue_comment(c.Comment or "")
            and c.Kind == 0
        )

        total_predicted = len(corrections)
        accuracy = n_kept / total_predicted if total_predicted > 0 else 0.0

        results.append(TrackFeedback(
            content_id=cid,
            title=title,
            bpm=bpm,
            timestamp=timestamp,
            cbr_spacing=cbr_spacing,
            corrections=corrections,
            n_kept=n_kept,
            n_moved=n_moved,
            n_deleted=n_deleted,
            n_user_added=n_user_added,
            accuracy=accuracy,
        ))

    return results


# --- Ausgabe ---

def print_feedback_summary(feedback: list[TrackFeedback]) -> None:
    """Gibt eine Zusammenfassung des Feedbacks aus."""
    if not feedback:
        print("\n  Kein Feedback verfuegbar.")
        return

    print(f"\n{'='*60}")
    print(f"  v3 Feedback-Analyse")
    print(f"{'='*60}")

    total_kept = sum(f.n_kept for f in feedback)
    total_moved = sum(f.n_moved for f in feedback)
    total_deleted = sum(f.n_deleted for f in feedback)
    total_user_added = sum(f.n_user_added for f in feedback)
    total_all = total_kept + total_moved + total_deleted

    avg_accuracy = np.mean([f.accuracy for f in feedback]) if feedback else 0

    print(f"\n  Tracks analysiert:   {len(feedback)}")
    print(f"  Cues insgesamt:      {total_all}")
    print(f"    Behalten:          {total_kept:4d} "
          f"({total_kept/total_all*100:.0f}%)" if total_all else "")
    print(f"    Verschoben:        {total_moved:4d} "
          f"({total_moved/total_all*100:.0f}%)" if total_all else "")
    print(f"    Geloescht:         {total_deleted:4d} "
          f"({total_deleted/total_all*100:.0f}%)" if total_all else "")
    print(f"    User hinzugefuegt: {total_user_added:4d}")
    print(f"  Genauigkeit:         {avg_accuracy:.1%}")

    # --- Details pro Cue-Typ ---
    for kind, label in [(1, "Hot A"), (2, "Hot B"), (3, "Hot C"),
                        (0, "Memory")]:
        corrs = [c for f in feedback for c in f.corrections
                 if c.cue_kind == kind]
        if not corrs:
            continue

        n = len(corrs)
        n_k = sum(1 for c in corrs if c.action == "kept")
        n_m = sum(1 for c in corrs if c.action == "moved")
        n_d = sum(1 for c in corrs if c.action == "deleted")

        print(f"\n  {label} ({n} Cues):")
        print(f"    Behalten: {n_k}  Verschoben: {n_m}  Geloescht: {n_d}")

        # Verschiebungs-Details
        moved = [c for c in corrs if c.action == "moved"]
        if moved:
            deltas = [c.delta_ms for c in moved]
            avg_delta = np.mean(deltas)
            direction = "spaeter" if avg_delta > 0 else "frueher"
            print(f"    Mittlere Verschiebung: "
                  f"{abs(avg_delta)/1000:.1f}s {direction} "
                  f"(~{np.mean([c.delta_beats for c in moved]):.1f} Beats)")

    # --- Pro Track ---
    print(f"\n  Details pro Track:")
    for f in feedback:
        status = "✓" if f.accuracy >= 0.8 else "△" if f.accuracy >= 0.5 else "✗"
        print(f"    {status} {f.title}: "
              f"{f.accuracy:.0%} Genauigkeit "
              f"({f.n_kept}↔ {f.n_moved}→ {f.n_deleted}✗)")

    print()
