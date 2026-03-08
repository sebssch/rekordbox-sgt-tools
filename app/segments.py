"""
v3 Acoustic Segmentation Engine
Erkennt strukturelle Grenzen durch Self-Similarity Matrix, Novelty Curves
und Spectral-Flux-Analyse. Keine Rekordbox-Phrasen — rein akustisch.
"""

from dataclasses import dataclass, field
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from app.beatgrid import BeatGrid, generate_phrase_candidates, _nearest_index, get_beat_index_at_time


@dataclass
class Segment:
    """Ein erkanntes Segment im Track."""
    start_time: float
    end_time: float
    kind: str              # "intro", "build", "drop", "break", "verse", "outro", "transition"
    confidence: float      # 0.0–1.0
    energy_mean: float     # Durchschnitts-RMS im Segment
    energy_delta: float    # Aenderung zur vorherigen Sektion
    start_beat_idx: int = 0


@dataclass
class TrackAnalysis:
    """Gesamtanalyse eines Tracks — alle akustischen Features beat-synchron."""
    grid: BeatGrid
    energy: np.ndarray         # RMS pro Beat
    spectral_flux: np.ndarray  # Spectral Flux pro Beat
    percussive_ratio: np.ndarray  # Percussion-Anteil pro Beat
    novelty: np.ndarray        # Novelty Curve pro Beat
    ssm_boundaries: np.ndarray # Erkannte Grenzen (Beat-Indices)
    complexity: float          # 0.0 (flach) bis 1.0 (komplex)
    segments: list[Segment] = field(default_factory=list)
    visual_edge_last_drop: float | None = None  # Zeit des letzten Drops via Visual Edge Detection
    mik_hotspots: list[float] = field(default_factory=list)  # MIK-Cue-Positionen (in Sekunden)


# --- Beat-synchrone Feature-Extraktion ---

def _beat_sync_feature(feature_frames: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    """Aggregiert Frame-basierte Features auf Beat-Ebene (Median pro Beat)."""
    synced = librosa.util.sync(feature_frames, beat_frames, aggregate=np.median)
    return synced.flatten()


def compute_energy_profile(y: np.ndarray, sr: int,
                           grid: BeatGrid) -> np.ndarray:
    """RMS-Energie pro Beat, normalisiert auf [0, 1]."""
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    beat_frames = librosa.time_to_frames(grid.times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, len(rms) - 1)

    synced = _beat_sync_feature(rms.reshape(1, -1), beat_frames)
    # Normalisieren
    max_val = synced.max()
    if max_val > 0:
        synced = synced / max_val
    return synced


def compute_spectral_flux(y: np.ndarray, sr: int,
                          grid: BeatGrid) -> np.ndarray:
    """
    Spectral Flux pro Beat — misst wie stark sich das Spektrum aendert.
    Hohe Werte = Uebergang/Transition.
    """
    S = np.abs(librosa.stft(y, hop_length=512))
    # Flux = Summe der positiven Differenzen zwischen aufeinanderfolgenden Frames
    flux = np.zeros(S.shape[1])
    flux[1:] = np.sum(np.maximum(0, np.diff(S, axis=1)), axis=0)

    beat_frames = librosa.time_to_frames(grid.times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, len(flux) - 1)

    synced = _beat_sync_feature(flux.reshape(1, -1), beat_frames)
    max_val = synced.max()
    if max_val > 0:
        synced = synced / max_val
    return synced


def compute_percussive_ratio(y: np.ndarray, sr: int,
                             grid: BeatGrid) -> np.ndarray:
    """
    Percussion-Anteil pro Beat via HPSS.
    Hohe Werte = Drums dominant, niedrige Werte = Break/melodisch.
    """
    D = librosa.stft(y, hop_length=512)
    H, P = librosa.decompose.hpss(D)

    h_energy = np.sum(np.abs(H) ** 2, axis=0)
    p_energy = np.sum(np.abs(P) ** 2, axis=0)
    total = h_energy + p_energy
    ratio = np.where(total > 0, p_energy / total, 0.0)

    beat_frames = librosa.time_to_frames(grid.times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, len(ratio) - 1)

    synced = _beat_sync_feature(ratio.reshape(1, -1), beat_frames)
    return synced


def compute_novelty_curve(y: np.ndarray, sr: int,
                          grid: BeatGrid) -> np.ndarray:
    """
    Novelty Curve basierend auf Mel-Spektrogramm-Aenderungen.
    Peaks = strukturelle Grenzen.
    """
    # Mel-Spektrogramm
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Kosinus-Distanz zwischen aufeinanderfolgenden Frames
    S_norm = S_db / (np.linalg.norm(S_db, axis=0, keepdims=True) + 1e-10)
    novelty = np.zeros(S_norm.shape[1])
    novelty[1:] = 1.0 - np.sum(S_norm[:, :-1] * S_norm[:, 1:], axis=0)
    novelty = np.maximum(0, novelty)

    beat_frames = librosa.time_to_frames(grid.times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, len(novelty) - 1)

    synced = _beat_sync_feature(novelty.reshape(1, -1), beat_frames)
    max_val = synced.max()
    if max_val > 0:
        synced = synced / max_val
    return synced


# --- Self-Similarity Matrix & Boundary Detection ---

def detect_ssm_boundaries(y: np.ndarray, sr: int,
                          grid: BeatGrid,
                          k: int = 5) -> np.ndarray:
    """
    Erkennt strukturelle Grenzen via Self-Similarity Matrix (Mel-basiert).
    Gibt Beat-Indices der erkannten Grenzen zurueck.

    k: Kernel-Groesse fuer den Novelty-Kernel (in Beats).
    """
    # Beat-synchrones Mel-Spektrogramm
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    beat_frames = librosa.time_to_frames(grid.times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, S_db.shape[1] - 1)
    S_beat = librosa.util.sync(S_db, beat_frames, aggregate=np.median)

    # Recurrence Matrix (Self-Similarity)
    R = librosa.segment.recurrence_matrix(
        S_beat, width=4, mode="affinity", sym=True
    )

    # Novelty aus der SSM ableiten (Checkerboard Kernel)
    n = R.shape[0]
    novelty = np.zeros(n)
    for i in range(k, n - k):
        # 4 Quadranten um Punkt (i, i) vergleichen
        tl = R[i - k:i, i - k:i].mean()   # top-left (same segment)
        br = R[i:i + k, i:i + k].mean()   # bottom-right (same segment)
        tr = R[i - k:i, i:i + k].mean()   # top-right (cross-segment)
        bl = R[i:i + k, i - k:i].mean()   # bottom-left (cross-segment)
        novelty[i] = (tl + br) / 2 - (tr + bl) / 2

    novelty = np.maximum(0, novelty)
    # Glaetten
    novelty = median_filter(novelty, size=3)

    # Peaks finden
    if novelty.max() > 0:
        novelty_norm = novelty / novelty.max()
    else:
        novelty_norm = novelty

    threshold = 0.35
    min_distance = 24  # Mindestens 24 Beats zwischen Grenzen (~6 Bars)
    peaks, properties = find_peaks(
        novelty_norm, height=threshold, distance=min_distance
    )

    return peaks


# --- Komplexitaetsberechnung ---

def compute_track_complexity(energy: np.ndarray, spectral_flux: np.ndarray,
                             percussive_ratio: np.ndarray) -> float:
    """
    Berechnet wie 'komplex' ein Track ist (0.0 = flach/repetitiv, 1.0 = komplex).
    Basiert auf: Energie-Varianz, Spectral-Flux-Varianz, Percussive-Varianz.
    """
    e_var = np.std(energy)
    f_var = np.std(spectral_flux)
    p_var = np.std(percussive_ratio)

    # Gewichtete Kombination, normalisiert auf ~0-1
    raw = 0.3 * e_var + 0.4 * f_var + 0.3 * p_var
    # Sigmoid-aehnliche Normalisierung
    complexity = float(np.tanh(raw * 5))
    return np.clip(complexity, 0.0, 1.0)


# --- Logical-Snap: Akustik vs. Mathematik ---

def logical_snap(acoustic_boundaries: np.ndarray,
                 grid: BeatGrid,
                 complexity: float,
                 phrase_intervals: list[int] = None,
                 max_drift_beats: int = 4) -> np.ndarray:
    """
    Waehlt fuer jede akustische Grenze den besten Kandidaten-Punkt:
    - Bei niedriger Komplexitaet: bevorzugt mathematische 32-Beat-Grenzen
    - Bei hoher Komplexitaet: bevorzugt die akustische Position

    acoustic_boundaries: Beat-Indices der erkannten akustischen Grenzen
    complexity: 0.0 (flach) bis 1.0 (komplex)
    max_drift_beats: Maximale Abweichung von der akustischen Grenze

    Returns: Beat-Indices der finalen Grenzen (auf Grid gesnappt)
    """
    if phrase_intervals is None:
        phrase_intervals = [8, 16, 32]

    if len(acoustic_boundaries) == 0:
        return np.array([], dtype=int)

    # Alle mathematischen Kandidaten (Beat-Indices)
    math_candidates = set()
    for n in phrase_intervals:
        for idx in range(0, grid.count, n):
            math_candidates.add(idx)
    math_candidates = np.array(sorted(math_candidates))

    # Gewichtung: complexity steuert ob Akustik oder Mathe gewinnt
    # w_acoustic = complexity, w_math = 1 - complexity
    w_acoustic = np.clip(complexity, 0.2, 0.8)
    w_math = 1.0 - w_acoustic

    final_boundaries = []

    for ab in acoustic_boundaries:
        # Finde mathematische Kandidaten in der Naehe
        nearby_mask = np.abs(math_candidates - ab) <= max_drift_beats
        nearby_math = math_candidates[nearby_mask]

        if len(nearby_math) == 0:
            # Kein mathematischer Kandidat in Reichweite
            if complexity > 0.5:
                # Komplexer Track: akustische Position beibehalten
                final_boundaries.append(int(ab))
            else:
                # Flacher Track: naechsten mathematischen Punkt nehmen
                closest_idx = _nearest_index(
                    math_candidates.astype(float), float(ab)
                )
                final_boundaries.append(int(math_candidates[closest_idx]))
        else:
            # Score fuer jeden Kandidaten berechnen
            best_score = -1
            best_candidate = int(ab)

            for mc in nearby_math:
                distance = abs(mc - ab)
                # Akustik-Score: wie nah am akustischen Punkt (0 = perfekt)
                acoustic_score = 1.0 - (distance / max(max_drift_beats, 1))
                # Mathe-Score: bevorzugt groessere Intervalle (32 > 16 > 8)
                math_score = 0.0
                for n in sorted(phrase_intervals, reverse=True):
                    if mc % n == 0:
                        math_score = 1.0
                        break
                    elif mc % (n // 2) == 0:
                        math_score = 0.5

                total = w_acoustic * acoustic_score + w_math * math_score
                if total > best_score:
                    best_score = total
                    best_candidate = int(mc)

            final_boundaries.append(best_candidate)

    # Deduplizieren und sortieren
    return np.unique(np.array(final_boundaries))


# --- Segment-Klassifizierung ---

def classify_segments(boundaries_beat_idx: np.ndarray,
                      grid: BeatGrid,
                      energy: np.ndarray,
                      percussive_ratio: np.ndarray,
                      spectral_flux: np.ndarray) -> list[Segment]:
    """
    Klassifiziert Segmente zwischen den erkannten Grenzen.
    Nutzt Energie, Percussive Ratio und Spectral Flux.
    """
    if len(boundaries_beat_idx) == 0:
        return []

    # Grenzen mit Track-Start und -Ende ergaenzen
    all_bounds = np.concatenate([[0], boundaries_beat_idx, [grid.count - 1]])
    all_bounds = np.unique(all_bounds)

    segments = []
    prev_energy = 0.0

    for i in range(len(all_bounds) - 1):
        start_idx = int(all_bounds[i])
        end_idx = int(all_bounds[i + 1])

        if start_idx >= end_idx:
            continue

        seg_energy = energy[start_idx:end_idx]
        seg_perc = percussive_ratio[start_idx:end_idx]
        seg_flux = spectral_flux[start_idx:end_idx]

        e_mean = float(np.mean(seg_energy))
        e_delta = e_mean - prev_energy
        p_mean = float(np.mean(seg_perc))
        f_mean = float(np.mean(seg_flux))

        # Position im Track (0.0 = Anfang, 1.0 = Ende)
        position = start_idx / max(grid.count - 1, 1)

        # Klassifizierung basierend auf Features
        is_first = (i == 0)
        is_last = (i == len(all_bounds) - 2)
        kind = _classify_segment_kind(
            e_mean, e_delta, p_mean, f_mean, position,
            start_idx, end_idx, grid.count,
            is_first=is_first, is_last=is_last
        )

        # Konfidenz basierend auf Feature-Klarheit
        confidence = _compute_segment_confidence(
            e_mean, e_delta, p_mean, kind
        )

        segments.append(Segment(
            start_time=float(grid.times[start_idx]),
            end_time=float(grid.times[min(end_idx, grid.count - 1)]),
            kind=kind,
            confidence=confidence,
            energy_mean=e_mean,
            energy_delta=e_delta,
            start_beat_idx=start_idx,
        ))

        prev_energy = e_mean

    return segments


def _classify_segment_kind(e_mean: float, e_delta: float,
                           p_mean: float, f_mean: float,
                           position: float,
                           start_idx: int, end_idx: int,
                           total_beats: int,
                           is_first: bool = False,
                           is_last: bool = False) -> str:
    """Regelbasierte Segment-Klassifizierung."""
    # Erstes und letztes Segment sind immer Intro/Outro
    if is_first:
        return "intro"
    if is_last:
        return "outro"

    # Break: niedrige Energie
    if e_mean < 0.4:
        return "break"

    # Drop: hohe Energie mit starkem Anstieg oder generell hoch + perkussiv
    if e_delta > 0.25 and e_mean > 0.6:
        return "drop"
    if e_mean > 0.7 and p_mean > 0.4:
        return "drop"

    # Build: ansteigende Energie, noch unter Drop-Niveau
    if e_delta > 0.1 and e_mean < 0.7:
        return "build"

    # Break: starker Energieabfall UND niedrige Energie
    if e_delta < -0.2 and e_mean < 0.5:
        return "break"

    return "verse"


def _compute_segment_confidence(e_mean: float, e_delta: float,
                                p_mean: float, kind: str) -> float:
    """Berechnet Konfidenz fuer die Segment-Klassifizierung."""
    if kind == "drop":
        return min(1.0, 0.5 + abs(e_delta) + p_mean)
    if kind == "break":
        return min(1.0, 0.5 + abs(e_delta) + (1.0 - p_mean))
    if kind in ("intro", "outro"):
        return 0.8
    if kind == "build":
        return min(1.0, 0.4 + abs(e_delta))
    return 0.5


# --- Energy-Transition-Erkennung ---

def detect_energy_transitions(energy: np.ndarray,
                              min_delta: float = 0.25,
                              window: int = 16,
                              min_distance: int = 32) -> np.ndarray:
    """
    Erkennt Beat-Positionen wo die Energie signifikant springt.
    Faengt Break→Drop und Drop→Break Uebergaenge, die SSM/Novelty verpassen.
    """
    n = len(energy)
    transitions = []

    for i in range(window, n - window, 4):
        e_before = np.mean(energy[max(0, i - window):i])
        e_after = np.mean(energy[i:min(n, i + window)])
        delta = abs(e_after - e_before)

        if delta >= min_delta:
            transitions.append((i, delta))

    if not transitions:
        return np.array([], dtype=int)

    # Nur die staerksten Transitions behalten (Non-Maximum-Suppression)
    transitions.sort(key=lambda x: -x[1])
    selected = []
    for beat_idx, delta in transitions:
        too_close = False
        for existing in selected:
            if abs(beat_idx - existing) < min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(beat_idx)

    return np.array(sorted(selected), dtype=int)


# --- Visual Edge Detection ---

def detect_visual_edge_last_drop(y: np.ndarray, sr: int,
                                  grid: BeatGrid,
                                  analysis: 'TrackAnalysis' = None) -> float | None:
    """
    Visual Edge Detection: Findet den Beginn des letzten energetischen Blocks
    (Last Drop) durch Erkennung des Farbwechsels Gruen → Rot in der
    Rekordbox-RGB-Wellenform.

    Methode:
    1. Berechne beat-synchrones Spektrum in 3 Baendern:
       - Low (20-250 Hz) → Rot in Rekordbox
       - Mid (250-4000 Hz) → Gruen in Rekordbox
       - High (4000-16000 Hz) → Blau in Rekordbox
    2. Berechne low/mid Ratio pro Beat
    3. Finde den LETZTEN signifikanten Anstieg des low/mid Ratio
       (= Gruen wird Rot = Bass/Kick setzt ein)
    4. Suche nur in der zweiten Haelfte des Tracks

    Args:
        y: Audio-Signal (mono)
        sr: Sample-Rate
        grid: BeatGrid des Tracks
        analysis: Optional TrackAnalysis (fuer Segment-Hints)

    Returns:
        Zeit in Sekunden des letzten Drop-Beginns, oder None wenn nicht gefunden.
    """
    # 1. Mel-Spektrogramm mit hoher Frequenzaufloesung
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512,
                                       n_mels=128, fmax=16000)

    # 2. Frequenzbaender definieren (Mel-Bin-Bereiche)
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmax=16000)
    low_mask = mel_freqs < 250    # Bass/Kick → Rot
    mid_mask = (mel_freqs >= 250) & (mel_freqs < 4000)  # Melodic → Gruen

    # 3. Energie pro Band und Beat synchronisieren
    low_energy = np.sum(S[low_mask, :], axis=0)
    mid_energy = np.sum(S[mid_mask, :], axis=0)

    beat_frames = librosa.time_to_frames(grid.times, sr=sr, hop_length=512)
    beat_frames = np.clip(beat_frames, 0, S.shape[1] - 1)

    low_beat = _beat_sync_feature(low_energy.reshape(1, -1), beat_frames).flatten()
    mid_beat = _beat_sync_feature(mid_energy.reshape(1, -1), beat_frames).flatten()

    # 4. Low/Mid Ratio (smoothed)
    ratio = low_beat / (mid_beat + 1e-10)
    ratio_smooth = median_filter(ratio, size=8)

    # 5. Nur zweite Haelfte des Tracks betrachten
    half_point = len(ratio_smooth) // 2

    # 6. Suchfenster bestimmen: NICHT in den Outro hineinsuchen
    diff = np.diff(ratio_smooth)

    search_start = half_point

    # search_end: Outro-Start bevorzugen, sonst max 80% des Tracks
    search_end = int(len(diff) * 0.80)  # Fallback: 80% des Tracks

    if analysis and analysis.segments:
        # Suche Outro-Segment → Suchgrenze = Outro-Start
        for seg in analysis.segments:
            if seg.kind == "outro":
                outro_beat = get_beat_index_at_time(seg.start_time, grid)
                # Etwas Puffer: 8 Beats vor dem Outro
                outro_beat = max(0, outro_beat - 8)
                if outro_beat > search_start:
                    search_end = min(search_end, outro_beat)
                break

    if search_start >= search_end:
        return None  # Track zu kurz

    search_diff = diff[search_start:search_end]

    if len(search_diff) == 0 or search_diff.max() <= 0:
        return None

    # Finde Peaks im Anstieg
    positive_vals = search_diff[search_diff > 0]
    if len(positive_vals) == 0:
        return None

    peaks, props = find_peaks(search_diff,
                               height=np.percentile(positive_vals, 75),
                               distance=16)

    if len(peaks) == 0:
        # Fallback: groesster Anstieg in der zweiten Haelfte (vor Outro)
        peak_idx = np.argmax(search_diff)
    else:
        # Letzter signifikanter Peak vor dem Outro = letzter Drop
        peak_idx = peaks[-1]

    # Beat-Index im gesamten Track
    drop_beat_idx = search_start + peak_idx

    # Optional: Verfeinere mit Segment-Info (Drop-Segment in der Naehe)
    if analysis and analysis.segments:
        for seg in reversed(analysis.segments):
            if seg.kind == "drop":
                seg_beat = get_beat_index_at_time(seg.start_time, grid)
                if abs(seg_beat - drop_beat_idx) <= 16:
                    drop_beat_idx = seg_beat
                    break

    # Zeit zurueckgeben
    if drop_beat_idx < len(grid.times):
        return float(grid.times[drop_beat_idx])
    return None


# --- Post-Processing ---

def _energy_gated_boundaries(boundaries: np.ndarray, energy: np.ndarray,
                              min_delta: float = 0.15,
                              window: int = 8) -> np.ndarray:
    """
    Behaelt nur Boundaries wo sich die Energie signifikant aendert.
    Vergleicht Durchschnittsenergie vor/nach der Boundary.
    """
    if len(boundaries) == 0:
        return boundaries

    gated = []
    for b in boundaries:
        before_start = max(0, b - window)
        after_end = min(len(energy), b + window)
        if before_start >= b or b >= after_end:
            continue
        e_before = np.mean(energy[before_start:b])
        e_after = np.mean(energy[b:after_end])
        if abs(e_after - e_before) >= min_delta:
            gated.append(b)

    return np.array(gated, dtype=int) if gated else np.array([], dtype=int)


def _filter_redundant_boundaries(new_bounds: np.ndarray,
                                  existing_bounds: np.ndarray,
                                  min_distance: int = 16) -> np.ndarray:
    """Entfernt Boundaries die zu nah an bereits vorhandenen liegen."""
    if len(new_bounds) == 0 or len(existing_bounds) == 0:
        return new_bounds

    filtered = []
    for b in new_bounds:
        distances = np.abs(existing_bounds - b)
        if distances.min() >= min_distance:
            filtered.append(b)
    return np.array(filtered, dtype=int)


def _merge_short_segments(segments: list[Segment], grid: BeatGrid,
                          min_beats: int = 32) -> list[Segment]:
    """
    Merged Segmente die kuerzer als min_beats sind mit dem benachbarten
    Segment, das aehnlichere Energie hat.
    """
    if len(segments) <= 2:
        return segments

    merged = list(segments)
    changed = True

    while changed:
        changed = False
        new_merged = []
        skip_next = False

        for i in range(len(merged)):
            if skip_next:
                skip_next = False
                continue

            seg = merged[i]
            beat_idx_start = seg.start_beat_idx
            if i + 1 < len(merged):
                beat_idx_end = merged[i + 1].start_beat_idx
            else:
                beat_idx_end = grid.count - 1

            seg_length = beat_idx_end - beat_idx_start

            if seg_length < min_beats and len(new_merged) > 0:
                # Merge mit vorherigem Segment (erweitere dessen end_time)
                prev = new_merged[-1]
                new_merged[-1] = Segment(
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    kind=prev.kind,
                    confidence=max(prev.confidence, seg.confidence),
                    energy_mean=(prev.energy_mean + seg.energy_mean) / 2,
                    energy_delta=prev.energy_delta,
                    start_beat_idx=prev.start_beat_idx,
                )
                changed = True
            elif seg_length < min_beats and i + 1 < len(merged):
                # Merge mit naechstem Segment
                nxt = merged[i + 1]
                new_merged.append(Segment(
                    start_time=seg.start_time,
                    end_time=nxt.end_time,
                    kind=nxt.kind,
                    confidence=max(seg.confidence, nxt.confidence),
                    energy_mean=(seg.energy_mean + nxt.energy_mean) / 2,
                    energy_delta=seg.energy_delta,
                    start_beat_idx=seg.start_beat_idx,
                ))
                skip_next = True
                changed = True
            else:
                new_merged.append(seg)

        merged = new_merged

    return merged


def _merge_same_kind(segments: list[Segment]) -> list[Segment]:
    """Merged aufeinanderfolgende Segmente gleichen Typs."""
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg.kind == prev.kind:
            # Zusammenfuegen
            merged[-1] = Segment(
                start_time=prev.start_time,
                end_time=seg.end_time,
                kind=prev.kind,
                confidence=(prev.confidence + seg.confidence) / 2,
                energy_mean=(prev.energy_mean + seg.energy_mean) / 2,
                energy_delta=prev.energy_delta,
                start_beat_idx=prev.start_beat_idx,
            )
        else:
            merged.append(seg)

    return merged


# --- Hauptfunktion ---

def analyze_track(y: np.ndarray, sr: int,
                  grid: BeatGrid) -> TrackAnalysis:
    """
    Vollstaendige akustische Analyse eines Tracks.
    Gibt alle Features und erkannte Segmente zurueck.
    """
    # Features berechnen
    energy = compute_energy_profile(y, sr, grid)
    spectral_flux = compute_spectral_flux(y, sr, grid)
    percussive_ratio = compute_percussive_ratio(y, sr, grid)
    novelty = compute_novelty_curve(y, sr, grid)

    # Komplexitaet
    complexity = compute_track_complexity(energy, spectral_flux, percussive_ratio)

    # SSM Boundaries
    ssm_bounds = detect_ssm_boundaries(y, sr, grid)

    # Novelty Peaks — nur starke Peaks mit grossem Abstand
    if novelty.max() > 0:
        novelty_norm = novelty / novelty.max()
    else:
        novelty_norm = novelty
    novelty_peaks, _ = find_peaks(novelty_norm, height=0.6, distance=64)

    # Nur Novelty-Peaks behalten wo auch ein signifikanter Energie-Wechsel ist
    # (Combined Evidence: Novelty allein reicht nicht)
    novelty_peaks = _energy_gated_boundaries(novelty_peaks, energy, min_delta=0.15)

    # Nur Novelty-Peaks behalten, die weit genug von SSM-Boundaries sind
    # (SSM ist primaer, Novelty ergaenzt nur wo SSM nichts erkannt hat)
    filtered_novelty = _filter_redundant_boundaries(
        novelty_peaks, ssm_bounds, min_distance=24
    )

    # Energy-Transitions als dritte Quelle
    energy_bounds = detect_energy_transitions(energy)
    # Nur Energy-Boundaries die weit genug von SSM + Novelty sind
    existing_bounds = np.concatenate([ssm_bounds, filtered_novelty])
    if len(existing_bounds) > 0 and len(energy_bounds) > 0:
        filtered_energy = _filter_redundant_boundaries(
            energy_bounds, existing_bounds, min_distance=16
        )
    else:
        filtered_energy = energy_bounds

    # Kombiniere alle drei Quellen
    all_sources = [ssm_bounds, filtered_novelty]
    if len(filtered_energy) > 0:
        all_sources.append(filtered_energy)
    all_acoustic = np.unique(np.concatenate(all_sources))

    # Logical Snap
    final_bounds = logical_snap(all_acoustic, grid, complexity)

    # Segmente klassifizieren
    segments = classify_segments(
        final_bounds, grid, energy, percussive_ratio, spectral_flux
    )

    # Post-Processing: kleine Segmente mergen, gleiche Typen zusammenfassen
    segments = _merge_short_segments(segments, grid, min_beats=32)
    segments = _merge_same_kind(segments)

    # Intro/Outro nach Merging sicherstellen
    if segments:
        segments[0] = Segment(
            start_time=segments[0].start_time,
            end_time=segments[0].end_time,
            kind="intro",
            confidence=segments[0].confidence,
            energy_mean=segments[0].energy_mean,
            energy_delta=segments[0].energy_delta,
            start_beat_idx=segments[0].start_beat_idx,
        )
        segments[-1] = Segment(
            start_time=segments[-1].start_time,
            end_time=segments[-1].end_time,
            kind="outro",
            confidence=segments[-1].confidence,
            energy_mean=segments[-1].energy_mean,
            energy_delta=segments[-1].energy_delta,
            start_beat_idx=segments[-1].start_beat_idx,
        )

    # TrackAnalysis erstellen (noch ohne visual_edge)
    track_analysis = TrackAnalysis(
        grid=grid,
        energy=energy,
        spectral_flux=spectral_flux,
        percussive_ratio=percussive_ratio,
        novelty=novelty,
        ssm_boundaries=ssm_bounds,
        complexity=complexity,
        segments=segments,
    )

    # Visual Edge Detection: letzten Drop via RGB-Farbwechsel finden
    track_analysis.visual_edge_last_drop = detect_visual_edge_last_drop(
        y, sr, grid, analysis=track_analysis
    )

    return track_analysis
