"""
v26 Beatgrid-Synchronisation
Liest das Rekordbox-Beatgrid aus ANLZ-Dateien und bietet exaktes Snap-to-Grid.
"""

from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path

import pyrekordbox.config as _cfg
from app import config as _autocue_cfg

# Pfad aus config.yaml oder macOS-Standard
_MASTER_DB_PATH = _autocue_cfg.get_rekordbox_db_path()
_DB_DIR = os.path.dirname(_MASTER_DB_PATH)

# pyrekordbox Config-Workaround (muss vor DB-Import passieren)
_cfg.__config__["rekordbox7"] = {
    "db_path": _MASTER_DB_PATH,
    "db_dir": _DB_DIR,
    "app_dir": os.path.expanduser(
        "~/Library/Application Support/Pioneer/rekordbox6"
    ),
}
_cfg.__config__["rekordbox6"] = _cfg.__config__["rekordbox7"]
_cfg.__config__["pioneer"] = {
    "install_dir": os.path.expanduser("~/Library/Pioneer"),
    "app_dir": "/Applications",
}

from pyrekordbox.db6.database import Rekordbox6Database
from pyrekordbox.anlz import AnlzFile


@dataclass
class BeatGrid:
    """Repraesentiert das Beatgrid eines Tracks."""
    times: np.ndarray        # Sekunden-Timestamp pro Beat
    beat_numbers: np.ndarray  # Beat im Takt (1,2,3,4,1,2,3,4,...)
    bpms: np.ndarray         # BPM pro Beat
    bpm: float               # Durchschnitts-BPM
    count: int               # Gesamtzahl Beats

    @property
    def downbeat_mask(self) -> np.ndarray:
        """Boolean-Maske: True fuer jeden Downbeat (Beat 1)."""
        return self.beat_numbers == 1

    @property
    def downbeat_times(self) -> np.ndarray:
        """Zeitstempel aller Downbeats (Beat 1)."""
        return self.times[self.downbeat_mask]

    @property
    def bar_count(self) -> int:
        """Anzahl vollstaendiger Takte."""
        return int(self.downbeat_mask.sum())

    def phrase_boundary_times(self, n_beats: int = 32) -> np.ndarray:
        """Zeitstempel aller n-Beat-Grenzen (z.B. alle 32 Beats)."""
        indices = np.arange(0, self.count, n_beats)
        return self.times[indices]


# --- Datenbank ---

def open_db(db_path: str = None) -> Rekordbox6Database:
    """Oeffnet die Rekordbox master.db."""
    if db_path is None:
        db_path = _MASTER_DB_PATH
    import logging
    logging.disable(logging.WARNING)
    db = Rekordbox6Database(path=db_path)
    db.open()
    logging.disable(logging.NOTSET)
    return db


def get_anlz_path(db: Rekordbox6Database, audio_path: str) -> tuple[str, str]:
    """
    Findet die ANLZ-Dateipfade (.DAT und .EXT) fuer einen Track.
    Returns: (dat_path, ext_path)
    """
    resolved = str(Path(audio_path).resolve())
    for content in db.get_content():
        if content.FolderPath == resolved:
            anlz_rel = content.AnalysisDataPath
            if not anlz_rel:
                raise FileNotFoundError(
                    f"Keine ANLZ-Daten fuer: {audio_path}"
                )
            dat_path = os.path.join(_DB_DIR, "share", anlz_rel.lstrip("/"))
            ext_path = dat_path.replace(".DAT", ".EXT")
            return dat_path, ext_path

    raise FileNotFoundError(f"Track nicht in DB: {audio_path}")


def find_content(db: Rekordbox6Database, audio_path: str):
    """Findet das Content-Objekt fuer einen Track."""
    resolved = str(Path(audio_path).resolve())
    for content in db.get_content():
        if content.FolderPath == resolved:
            return content
    return None


# --- Beatgrid lesen ---

def read_beatgrid(anlz_dat_path: str) -> BeatGrid:
    """
    Liest das Beatgrid aus einer ANLZ .DAT Datei (PQTZ Tag).
    """
    if not os.path.exists(anlz_dat_path):
        raise FileNotFoundError(f"ANLZ-Datei nicht gefunden: {anlz_dat_path}")

    anlz = AnlzFile.parse_file(anlz_dat_path)

    for tag in anlz.tags:
        if tag.type == "PQTZ":
            times = np.array(tag.times, dtype=np.float64)
            beats = np.array(tag.beats, dtype=np.int32)
            bpms = np.array(tag.bpms, dtype=np.float64)

            return BeatGrid(
                times=times,
                beat_numbers=beats,
                bpms=bpms,
                bpm=float(np.median(bpms)),
                count=len(times),
            )

    raise ValueError(f"Kein PQTZ-Tag in: {anlz_dat_path}")


def read_beatgrid_for_track(db: Rekordbox6Database, audio_path: str) -> BeatGrid:
    """Convenience: DB + Pfad → BeatGrid."""
    dat_path, _ = get_anlz_path(db, audio_path)
    return read_beatgrid(dat_path)


# --- Snap-to-Grid Funktionen ---

def _nearest_index(times: np.ndarray, target: float) -> int:
    """Findet den Index des naechsten Zeitpunkts."""
    idx = np.searchsorted(times, target)
    if idx == 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    # Vergleiche links und rechts
    if abs(times[idx] - target) < abs(times[idx - 1] - target):
        return idx
    return idx - 1


def snap_to_beat(time_sec: float, grid: BeatGrid) -> float:
    """Snappt einen Zeitpunkt auf den naechsten Beat im Grid."""
    idx = _nearest_index(grid.times, time_sec)
    return float(grid.times[idx])


def snap_to_downbeat(time_sec: float, grid: BeatGrid) -> float:
    """Snappt auf den naechsten Downbeat (Beat 1)."""
    dt = grid.downbeat_times
    idx = _nearest_index(dt, time_sec)
    return float(dt[idx])


def snap_to_phrase_boundary(time_sec: float, grid: BeatGrid,
                            n_beats: int = 32) -> float:
    """Snappt auf die naechste n-Beat-Grenze."""
    boundaries = grid.phrase_boundary_times(n_beats)
    idx = _nearest_index(boundaries, time_sec)
    return float(boundaries[idx])


def get_time_at_beat(beat_index: int, grid: BeatGrid) -> float:
    """Gibt die Zeit fuer einen Beat-Index (0-basiert) zurueck."""
    beat_index = max(0, min(beat_index, grid.count - 1))
    return float(grid.times[beat_index])


def get_beat_index_at_time(time_sec: float, grid: BeatGrid) -> int:
    """Gibt den Beat-Index (0-basiert) fuer einen Zeitpunkt zurueck."""
    return _nearest_index(grid.times, time_sec)


def get_time_n_beats_before(time_sec: float, n_beats: int,
                            grid: BeatGrid) -> float:
    """Berechnet die Zeit n Beats VOR einem Zeitpunkt."""
    idx = get_beat_index_at_time(time_sec, grid)
    target_idx = max(0, idx - n_beats)
    return float(grid.times[target_idx])


def get_time_n_beats_after(time_sec: float, n_beats: int,
                           grid: BeatGrid) -> float:
    """Berechnet die Zeit n Beats NACH einem Zeitpunkt."""
    idx = get_beat_index_at_time(time_sec, grid)
    target_idx = min(grid.count - 1, idx + n_beats)
    return float(grid.times[target_idx])


def generate_phrase_candidates(grid: BeatGrid,
                               intervals: list[int] = None) -> np.ndarray:
    """
    Generiert alle Kandidaten-Zeitpunkte an 8/16/32-Beat-Grenzen.
    Gibt ein sortiertes, dedupliziertes Array zurueck.
    """
    if intervals is None:
        intervals = [8, 16, 32]

    all_times = []
    for n in intervals:
        all_times.append(grid.phrase_boundary_times(n))

    combined = np.unique(np.concatenate(all_times))
    return combined
