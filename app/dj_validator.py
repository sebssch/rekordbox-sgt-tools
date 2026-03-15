"""
DJ-Validator: Wandelt ML-Probability-Maps in grid-konforme Kandidaten-Listen um.

Dieses Modul ist die Bruecke zwischen dem ML-Modell (cue_model.py) und dem
bestehenden Quad-Check-Validator (validator.py). Es extrahiert Peaks aus der
Probability-Map und stellt sicher, dass jeder Kandidat:

1. Auf einem Downbeat liegt (snap_to_downbeat)
2. Die Mindest-Wahrscheinlichkeit ueberschreitet

Die finale Entscheidung (Konsens, Struktur-Checks) trifft weiterhin
validate_hot_a() / validate_hot_c() im bestehenden Regelwerk.
"""

import logging

import numpy as np

from app.beatgrid import BeatGrid, snap_to_downbeat, get_beat_index_at_time

log = logging.getLogger("autocue.dj_validator")

# Kanal-Indizes (muessen mit cue_model.py uebereinstimmen)
_CH_HOT_A = 0
_CH_HOT_B = 1
_CH_HOT_C = 2
_CH_MEMORY = 3


def extract_candidates(
    prob_map: np.ndarray,
    duration_s: float,
    grid: BeatGrid,
    hot_threshold: float = 0.4,
    memory_threshold: float = 0.3,
) -> dict:
    """
    Extrahiert Peak-Positionen aus der Probability-Map und gibt
    grid-konforme Kandidaten-Listen zurueck.

    Args:
        prob_map:         (N_BINS, 4) Wahrscheinlichkeiten pro Bin/Kanal
        duration_s:       Tracklaenge in Sekunden
        grid:             BeatGrid fuer Snap-to-Downbeat
        hot_threshold:    Min-P fuer Hot-Cue-Kandidaten (Standard: 0.4)
        memory_threshold: Min-P fuer Memory-Cue-Kandidaten (Standard: 0.3)

    Returns:
        {
            "hot_a":  [float, ...],   # Zeitpunkte in Sekunden, sortiert
            "hot_c":  [float, ...],
            "memory": [float, ...],
        }
    """
    n_bins = prob_map.shape[0]

    hot_a_cands = _extract_peaks(
        prob_map[:, _CH_HOT_A], duration_s, grid, n_bins, hot_threshold
    )
    hot_c_cands = _extract_peaks(
        prob_map[:, _CH_HOT_C], duration_s, grid, n_bins, hot_threshold
    )
    memory_cands = _extract_peaks(
        prob_map[:, _CH_MEMORY], duration_s, grid, n_bins, memory_threshold
    )

    return {
        "hot_a": hot_a_cands,
        "hot_c": hot_c_cands,
        "memory": memory_cands,
    }


def _extract_peaks(
    probs: np.ndarray,
    duration_s: float,
    grid: BeatGrid,
    n_bins: int,
    threshold: float,
) -> list[float]:
    """
    Extrahiert lokale Maxima aus einer 1D-Wahrscheinlichkeitskurve.

    1. Lokale Maxima mit P > threshold finden
    2. Bin-Index → Zeitpunkt in Sekunden
    3. Snap auf naechsten Downbeat
    4. Deduplizierung: keine zwei Kandidaten naeher als 8 Beats

    Returns:
        Sortierte Liste von Zeitpunkten in Sekunden (absteigend nach P).
    """
    if duration_s <= 0 or len(probs) == 0:
        return []

    beat_dur = 60.0 / max(grid.bpm, 1.0)

    # 1. Lokale Maxima finden (einfacher Peak-Detektor)
    peaks: list[tuple[float, int]] = []  # (probability, bin_index)
    for i in range(n_bins):
        if probs[i] < threshold:
            continue
        # Lokales Maximum: groesser als Nachbarn
        left = probs[i - 1] if i > 0 else 0.0
        right = probs[i + 1] if i < n_bins - 1 else 0.0
        if probs[i] >= left and probs[i] >= right:
            peaks.append((float(probs[i]), i))

    if not peaks:
        return []

    # Nach Wahrscheinlichkeit sortieren (hoechste zuerst)
    peaks.sort(reverse=True)

    # 2 + 3. Bin → Zeit → Snap to Downbeat
    candidates: list[tuple[float, float]] = []  # (prob, snapped_time)
    for prob, bin_idx in peaks:
        time_sec = bin_idx / n_bins * duration_s
        snapped = snap_to_downbeat(time_sec, grid)

        candidates.append((prob, snapped))

    # 5. Deduplizierung: Keine zwei Kandidaten naeher als 8 Beats
    min_dist = 8 * beat_dur
    filtered: list[float] = []
    for prob, t in candidates:
        if not any(abs(t - existing) < min_dist for existing in filtered):
            filtered.append(t)

    return filtered


def _is_phrase_aligned(
    time_sec: float,
    grid: BeatGrid,
    tolerance_beats: int = 4,
) -> bool:
    """
    Prueft ob ein Zeitpunkt auf einer 16- oder 32-Beat-Phrase-Grenze liegt.

    Der Zeitpunkt muss innerhalb von ±tolerance_beats des naechsten
    16- oder 32-Beat-Vielfachen (relativ zum ersten Downbeat) liegen.
    """
    if grid.count < 2:
        return True

    beat_dur = 60.0 / max(grid.bpm, 1.0)
    first_downbeat = float(grid.times[0])

    # Abstand vom ersten Downbeat in Beats
    beats_from_start = (time_sec - first_downbeat) / beat_dur

    # Naechstes 16-Beat-Vielfaches
    nearest_16 = round(beats_from_start / 16.0) * 16.0
    dist_16 = abs(beats_from_start - nearest_16)

    # Naechstes 32-Beat-Vielfaches
    nearest_32 = round(beats_from_start / 32.0) * 32.0
    dist_32 = abs(beats_from_start - nearest_32)

    return min(dist_16, dist_32) <= tolerance_beats
