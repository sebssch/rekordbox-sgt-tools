# v26/mik_scraper.py
# Hybrid-Datenbeschaffung: MIK (Mixed In Key) Cue-Positionen
# Variante A: ID3-TXXX-Frames (mutagen), Variante B: MIK-SQLite-Datenbank
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app import config as _cfg

# MIK-Datenbank-Pfad (aus config.yaml oder macOS-Standard)
MIK_DB_PATH = Path(_cfg.get_mik_db_path())


@dataclass
class MikData:
    """MIK-Daten fuer einen Track."""
    cue_times: list[float] = field(default_factory=list)  # Cue-Positionen in Sekunden
    energy: float | None = None                            # MIK Energy Level (1-10)
    source: str = "unknown"                                # "id3" oder "db"


def _scrape_id3_tags(audio_path: str) -> MikData | None:
    """
    Variante A: ID3-TXXX-Frames nach MIK-spezifischen Eintraegen durchsuchen.
    Benoetigt: pip install mutagen
    """
    try:
        from mutagen.id3 import ID3
    except ImportError:
        return None

    try:
        tags = ID3(audio_path)
    except Exception:
        return None

    cue_times: list[float] = []
    energy = None

    for tag in tags.getall("TXXX"):
        desc = tag.desc.upper() if tag.desc else ""
        val_str = tag.text[0] if tag.text else ""

        if "MIK_ENERGY" in desc:
            try:
                energy = float(val_str)
            except ValueError:
                pass

        elif "CUE" in desc or "HOTCUE" in desc or "HOT_CUE" in desc:
            try:
                val = float(val_str)
                # Sekunden (< 7200 = 2h) oder Millisekunden (groesser)
                cue_times.append(val if val < 7200 else val / 1000.0)
            except ValueError:
                pass

    # Private Frames (PRIV) — MIK-spezifische Binaerdaten
    for tag in tags.getall("PRIV"):
        owner = str(getattr(tag, 'owner', '')).upper()
        if "MIK" in owner or "MIXEDINKEY" in owner:
            # Binaere Cue-Daten: Parsing je nach Format (noch nicht implementiert)
            pass

    if cue_times or energy is not None:
        return MikData(cue_times=sorted(cue_times), energy=energy, source="id3")

    return None


def _scrape_mik_db(artist: str, title: str) -> MikData | None:
    """
    Variante B: MIK-SQLite-Datenbank via Artist+Title-Matching abfragen.
    Datenbank: ~/Library/Application Support/MixedInKey/Collection11.mikdb
    Kein FilePath in DB — Matching via ZARTIST + ZNAME.
    """
    if not MIK_DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(str(MIK_DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        row = None

        # 1. Exakter Match (Artist + Titel)
        if artist and title:
            row = conn.execute(
                "SELECT Z_PK, ZENERGY FROM ZSONG "
                "WHERE LOWER(TRIM(ZNAME))=LOWER(TRIM(?)) "
                "  AND LOWER(TRIM(ZARTIST))=LOWER(TRIM(?)) LIMIT 1",
                (title, artist),
            ).fetchone()

        # 2. Fuzzy: nur Titel
        if row is None and title:
            row = conn.execute(
                "SELECT Z_PK, ZENERGY FROM ZSONG "
                "WHERE UPPER(ZNAME) LIKE UPPER(?) LIMIT 1",
                (f"%{title.strip()}%",),
            ).fetchone()

        if row is None:
            conn.close()
            return None

        song_pk = row["Z_PK"]
        energy = float(row["ZENERGY"]) if row["ZENERGY"] is not None else None

        # Cue-Positionen laden (ZTIME in Sekunden)
        cues = conn.execute(
            "SELECT ZTIME FROM ZCUEPOINT WHERE ZSONG=? ORDER BY ZTIME",
            (song_pk,),
        ).fetchall()
        conn.close()

        cue_times = [float(c["ZTIME"]) for c in cues if c["ZTIME"] is not None]

        if cue_times or energy is not None:
            return MikData(cue_times=cue_times, energy=energy, source="db")

        return None

    except Exception:
        return None


def get_mik_data(audio_path: str, artist: str = "", title: str = "") -> MikData | None:
    """
    Hybrid-Lookup: Erst ID3-Tags (Variante A), dann MIK-Datenbank (Variante B).

    Args:
        audio_path: Absoluter Pfad zur Audio-Datei
        artist:     Kuenstlername (aus Rekordbox DjmdContent.ArtistName)
        title:      Titelname (aus Rekordbox DjmdContent.Title)

    Returns:
        MikData mit Cue-Positionen und Energy oder None wenn kein Treffer.
    """
    # Variante A: ID3-Tags (schnell, keine DB-Abfrage)
    result = _scrape_id3_tags(audio_path)
    if result and result.cue_times:
        return result

    # Variante B: MIK-SQLite-Datenbank
    if artist or title:
        db_result = _scrape_mik_db(artist, title)
        if db_result:
            return db_result

    return None
