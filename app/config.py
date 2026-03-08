"""
v26 Konfigurations-Loader
Laedt config.yaml aus dem Projekt-Root und stellt alle Parameter bereit.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from functools import lru_cache
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH  = _PROJECT_ROOT / "config.yaml"


# --- Defaults (identisch mit config.yaml) ---

def _defaults() -> dict:
    return {
        "version":                       "26",
        "cue_prefix":                    "A:",
        "min_hot_a_b_distance_beats":    128,
        "hot_b_offset_beats":            32,
        "max_memory_cues":               10,
        "memory_min_hot_distance_beats": 16,
        "mik_snap_tolerance_beats":      4,
        "colors": {
            "hot_a":  -1,
            "hot_b":  -1,
            "hot_c":  -1,
            "memory": -1,
        },
        "analyse_playlist": "--analyse-tracks",
        "rekordbox_db_path": "",
        "mik_db_path":       "",
        "min_corrections":   3,
        "log_file":          "app.log",
        "log_level":         "INFO",
    }


# --- Loader ---

@lru_cache(maxsize=1)
def load_config() -> dict:
    """
    Laedt config.yaml. Gibt Defaults zurueck wenn Datei fehlt.
    Ergebnis wird gecacht (einmaliges Lesen pro Prozess).
    """
    if not _CONFIG_PATH.exists():
        return _defaults()

    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Fehlende Keys mit Defaults auffuellen
    defaults = _defaults()
    for key, val in defaults.items():
        if key not in cfg:
            cfg[key] = val
        elif isinstance(val, dict):
            for k, v in val.items():
                if k not in cfg[key]:
                    cfg[key][k] = v

    return cfg


def get(key: str, default=None):
    """Convenience: einzelnen Config-Wert abfragen."""
    return load_config().get(key, default)


# --- Pfad-Helpers ---

def get_rekordbox_db_path() -> str:
    """Gibt den Rekordbox master.db-Pfad zurueck (config oder macOS-Standard)."""
    configured = get("rekordbox_db_path") or ""
    if configured.strip():
        return os.path.expanduser(configured)
    return os.path.expanduser("~/Library/Pioneer/rekordbox/master.db")


def get_mik_db_path() -> str:
    """Gibt den MIK Collection11.mikdb-Pfad zurueck (config oder macOS-Standard)."""
    configured = get("mik_db_path") or ""
    if configured.strip():
        return os.path.expanduser(configured)
    return os.path.expanduser(
        "~/Library/Application Support/MixedInKey/Collection11.mikdb"
    )


# --- Logging Setup ---

def setup_logging() -> logging.Logger:
    """
    Richtet das zentrale Logging ein:
      - File Handler: app.log (konfigurierbar)
      - Stream Handler: stderr (nur WARNING+)

    Gibt den Root-Logger zurueck.
    """
    cfg = load_config()
    level_name = cfg.get("log_level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = cfg.get("log_file", "app.log")

    logger = logging.getLogger("autocue")
    if logger.handlers:
        return logger  # Bereits konfiguriert

    logger.setLevel(logging.DEBUG)  # Root auf DEBUG, Handler steuern Level

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler (rotierend, max 5 MB × 3 Dateien)
    if log_file:
        log_path = _PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Console Handler (nur WARNING+ damit rich-Output nicht gestört wird)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# --- Cue-Prefix Helpers ---

def get_cue_prefix() -> str:
    """Gibt den konfigurierten Cue-Kommentar-Prefix zurueck (z.B. 'A:')."""
    return get("cue_prefix", "A:")


def is_autocue_comment(comment: str) -> bool:
    """
    Prueft ob ein Kommentar von AutoCue stammt.
    Erkennt sowohl neuen Prefix (z.B. 'A:') als auch Legacy 'v3:' / 'AutoCue'.
    """
    if not comment:
        return False
    prefix = get_cue_prefix()
    return (
        comment.startswith(prefix)
        or comment.startswith("v3:")
        or comment.startswith("v26:")
        or comment.startswith("AutoCue")
    )
