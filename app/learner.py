"""
v26 Parameter-Learner
Extrahiert Lernmuster aus User-Korrekturen und passt
die Parameter schrittweise an.

Konservative Strategie: Aenderungen nur bei genuegend
Datenpunkten (min. MIN_CORRECTIONS Korrekturen gleicher Art).
"""

import json
import os
from datetime import datetime

import numpy as np

from app.feedback import TrackFeedback, CueCorrection
from app import config as _cfg


# --- Pfade + Konstanten ---

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LEARNED_PARAMS_PATH = os.path.join(_BASE_DIR, "data", "learned_params.json")

# Mindestanzahl Korrekturen gleicher Art (aus config.yaml)
MIN_CORRECTIONS = _cfg.get("min_corrections", 3)

# Defaults (identisch mit den hardcoded Werten in cbr.py / cue_logic.py)
_DEFAULTS = {
    "version": 1,
    "updated_at": None,
    "n_corrections_used": 0,

    # CBR-Parameter
    "cue_spacing_threshold": 48,      # Median >= X → 64-Beat-Spacing
    "hot_b_offset_beats": 32,         # Beats vor Hot C

    # Systematische Offsets (in ms)
    "hot_a_time_offset_ms": 0,
    "hot_c_time_offset_ms": 0,

    # Segment-Klassifikation
    "break_energy_threshold": 0.40,
    "drop_energy_delta": 0.25,
    "drop_energy_min": 0.60,

    # Genauigkeits-Tracking
    "confidence_scores": {
        "hot_a_accuracy": None,
        "hot_b_accuracy": None,
        "hot_c_accuracy": None,
        "memory_accuracy": None,
        "overall_accuracy": None,
    },
}


# --- Parameter laden / speichern ---

def load_learned_params() -> dict:
    """
    Laedt gelernte Parameter aus learned_params.json.
    Gibt Defaults zurueck wenn Datei nicht existiert.
    """
    if not os.path.isfile(_LEARNED_PARAMS_PATH):
        return _DEFAULTS.copy()

    with open(_LEARNED_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Defaults fuer fehlende Keys ergaenzen
    for key, default in _DEFAULTS.items():
        if key not in params:
            params[key] = default

    return params


def save_learned_params(params: dict) -> None:
    """Speichert Parameter in learned_params.json."""
    params["updated_at"] = datetime.now().isoformat(timespec="seconds")

    os.makedirs(os.path.dirname(_LEARNED_PARAMS_PATH), exist_ok=True)

    with open(_LEARNED_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


# --- Lern-Logik ---

def learn_from_feedback(feedback: list[TrackFeedback]) -> dict:
    """
    Analysiert Feedback und passt Parameter an.
    Konservativ: Aenderungen nur bei >= MIN_CORRECTIONS Datenpunkten.

    Args:
        feedback: Liste von TrackFeedback-Objekten

    Returns:
        Aktualisierte Parameter (auch gespeichert)
    """
    if not feedback:
        print("  Kein Feedback zum Lernen.")
        return load_learned_params()

    params = load_learned_params()
    changes = []

    # Alle Korrekturen sammeln
    all_corrections = [c for f in feedback for c in f.corrections]

    if not all_corrections:
        print("  Keine Korrekturen gefunden.")
        return params

    # ====================================================
    # 1. Hot A: Systematische Verschiebung erkennen
    # ====================================================
    hot_a_moved = [c for c in all_corrections
                   if c.cue_kind == 1 and c.action == "moved"]

    if len(hot_a_moved) >= MIN_CORRECTIONS:
        deltas = [c.delta_ms for c in hot_a_moved]
        mean_delta = int(np.mean(deltas))

        # Nur anpassen wenn Richtung konsistent (>60% gleiche Richtung)
        n_positive = sum(1 for d in deltas if d > 0)
        n_negative = sum(1 for d in deltas if d < 0)
        consistency = max(n_positive, n_negative) / len(deltas)

        if consistency >= 0.6 and abs(mean_delta) > 200:
            old = params["hot_a_time_offset_ms"]
            # Exponential Moving Average: 70% alter Wert, 30% neuer
            params["hot_a_time_offset_ms"] = int(old * 0.7 + mean_delta * 0.3)
            direction = "spaeter" if mean_delta > 0 else "frueher"
            changes.append(
                f"Hot A Offset: {abs(mean_delta)}ms {direction} "
                f"(aus {len(hot_a_moved)} Korrekturen, "
                f"Konsistenz: {consistency:.0%})")

    # ====================================================
    # 2. Hot C: Systematische Verschiebung erkennen
    # ====================================================
    hot_c_moved = [c for c in all_corrections
                   if c.cue_kind == 3 and c.action == "moved"]

    if len(hot_c_moved) >= MIN_CORRECTIONS:
        deltas = [c.delta_ms for c in hot_c_moved]
        mean_delta = int(np.mean(deltas))
        n_positive = sum(1 for d in deltas if d > 0)
        n_negative = sum(1 for d in deltas if d < 0)
        consistency = max(n_positive, n_negative) / len(deltas)

        if consistency >= 0.6 and abs(mean_delta) > 200:
            old = params["hot_c_time_offset_ms"]
            params["hot_c_time_offset_ms"] = int(old * 0.7 + mean_delta * 0.3)
            direction = "spaeter" if mean_delta > 0 else "frueher"
            changes.append(
                f"Hot C Offset: {abs(mean_delta)}ms {direction} "
                f"(aus {len(hot_c_moved)} Korrekturen)")

    # ====================================================
    # 3. Hot B Offset: Beats-Abstand zu Hot C lernen
    # ====================================================
    hot_b_moved = [c for c in all_corrections
                   if c.cue_kind == 2 and c.action == "moved"]

    if len(hot_b_moved) >= MIN_CORRECTIONS:
        # Verschiebung in Beats interpretieren
        beat_deltas = [c.delta_beats for c in hot_b_moved]
        mean_beats = np.mean(beat_deltas)

        # Wenn Hot B systematisch verschoben: Offset anpassen
        old_offset = params["hot_b_offset_beats"]
        # Negative delta_beats = naeher an Hot C → weniger Offset
        new_offset = int(round(old_offset - mean_beats))
        new_offset = max(16, min(64, new_offset))  # Clamp 16-64

        if new_offset != old_offset:
            params["hot_b_offset_beats"] = new_offset
            changes.append(
                f"Hot B Offset: {old_offset} → {new_offset} Beats "
                f"(aus {len(hot_b_moved)} Korrekturen)")

    # ====================================================
    # 4. Memory Cues: Spacing-Praeferenz erkennen
    # ====================================================
    mem_deleted = [c for c in all_corrections
                   if c.cue_kind == 0 and c.action == "deleted"]
    mem_total = [c for c in all_corrections if c.cue_kind == 0]

    if len(mem_total) >= MIN_CORRECTIONS * 2:
        delete_ratio = len(mem_deleted) / len(mem_total)

        # Wenn >50% Memory Cues geloescht → User will weniger Cues
        if delete_ratio > 0.5:
            old_threshold = params["cue_spacing_threshold"]
            # Threshold senken → mehr Tracks bekommen 64er Spacing
            params["cue_spacing_threshold"] = max(32, old_threshold - 4)
            changes.append(
                f"Cue-Spacing-Threshold: {old_threshold} → "
                f"{params['cue_spacing_threshold']} "
                f"({delete_ratio:.0%} Memory Cues geloescht)")

        # Wenn <20% geloescht → aktuelle Dichte passt gut
        elif delete_ratio < 0.2:
            old_threshold = params["cue_spacing_threshold"]
            if old_threshold < 48:
                params["cue_spacing_threshold"] = min(56, old_threshold + 2)
                changes.append(
                    f"Cue-Spacing-Threshold: {old_threshold} → "
                    f"{params['cue_spacing_threshold']} "
                    f"(nur {delete_ratio:.0%} geloescht, Dichte passt)")

    # ====================================================
    # 5. Genauigkeits-Tracking
    # ====================================================
    for kind, key in [(1, "hot_a"), (2, "hot_b"), (3, "hot_c"),
                      (0, "memory")]:
        kind_corrs = [c for c in all_corrections if c.cue_kind == kind]
        if kind_corrs:
            n_kept = sum(1 for c in kind_corrs if c.action == "kept")
            params["confidence_scores"][f"{key}_accuracy"] = round(
                n_kept / len(kind_corrs), 3)

    # Overall
    if all_corrections:
        n_kept = sum(1 for c in all_corrections if c.action == "kept")
        params["confidence_scores"]["overall_accuracy"] = round(
            n_kept / len(all_corrections), 3)

    # ====================================================
    # Speichern + Zusammenfassung
    # ====================================================
    params["n_corrections_used"] = len(all_corrections)
    save_learned_params(params)

    print(f"\n{'='*60}")
    print(f"  v3 Learner — Parameter-Update")
    print(f"{'='*60}")
    print(f"  Korrekturen analysiert: {len(all_corrections)}")
    print(f"  Tracks mit Feedback:    {len(feedback)}")

    if changes:
        print(f"\n  Anpassungen:")
        for ch in changes:
            print(f"    → {ch}")
    else:
        print(f"\n  Keine Anpassungen noetig "
              f"(zu wenig Daten oder Parameter bereits optimal)")

    conf = params["confidence_scores"]
    print(f"\n  Genauigkeit:")
    for key, label in [("hot_a_accuracy", "Hot A"),
                       ("hot_b_accuracy", "Hot B"),
                       ("hot_c_accuracy", "Hot C"),
                       ("memory_accuracy", "Memory"),
                       ("overall_accuracy", "Gesamt")]:
        val = conf.get(key)
        if val is not None:
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {label:8s} {bar} {val:.0%}")

    print(f"\n  Gespeichert: {_LEARNED_PARAMS_PATH}")
    print()

    return params
