#!/usr/bin/env python3
"""
Reset-Tool: Loescht Cue- und Memory-Punkte fuer alle Tracks einer Playlist.

Sicherheit:
  - Doppelte Bestaetigung (ja/nein + Eingabe 'LOESCHEN')
  - Niemals automatisch — immer interaktive Terminal-Eingabe

Beispiele:
  python tools/reset_playlist_cues.py --playlist "--analyse-tracks"
  python tools/reset_playlist_cues.py --playlist "Techno" --only-memory
  python tools/reset_playlist_cues.py --playlist "House" --only-hot
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pyrekordbox as prb
from app.batch import find_playlist, get_playlist_tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cue_kind_label(kind: int) -> str:
    if kind == 0:
        return "Mem"
    return f"Hot {chr(64 + kind)}"


def _collect_cues(content, only_memory: bool, only_hot: bool) -> list:
    """Gibt die zu loeschenden Cues eines Tracks zurueck."""
    cues = []
    for cue in content.Cues:
        if only_memory and cue.Kind != 0:
            continue
        if only_hot and cue.Kind == 0:
            continue
        cues.append(cue)
    return cues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Loescht Cue-Punkte fuer alle Tracks einer Playlist."
    )
    parser.add_argument("--playlist", required=True,
                        help="Name der Rekordbox-Playlist")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only-memory", action="store_true",
                       help="Nur Memory Cues loeschen")
    group.add_argument("--only-hot", action="store_true",
                       help="Nur Hot Cues loeschen")
    args = parser.parse_args()

    # --- DB oeffnen ---
    db = prb.Rekordbox6Database()

    # --- Playlist finden ---
    playlist = find_playlist(db, args.playlist)
    if playlist is None:
        print(f"\n  FEHLER: Playlist '{args.playlist}' nicht gefunden.")
        sys.exit(1)

    tracks = get_playlist_tracks(db, playlist)
    if not tracks:
        print(f"\n  Playlist '{args.playlist}' ist leer.")
        sys.exit(0)

    # --- Typ-Label ---
    if args.only_memory:
        typ = "Memory Cues"
    elif args.only_hot:
        typ = "Hot Cues"
    else:
        typ = "alle Cues (Hot + Memory)"

    # --- Uebersicht aufbauen ---
    print(f"\n{'=' * 60}")
    print(f"  PLAYLIST CUE-RESET")
    print(f"{'=' * 60}")
    print(f"  Playlist:  {args.playlist}")
    print(f"  Tracks:    {len(tracks)}")
    print(f"  Modus:     {typ}")
    print(f"{'=' * 60}\n")

    total_cues = 0
    track_details = []

    for content in tracks:
        artist = content.Artist.Name if content.Artist else ""
        title = content.Title or "?"
        label = f"{artist} - {title}" if artist else title

        cues = _collect_cues(content, args.only_memory, args.only_hot)
        n = len(cues)
        total_cues += n

        if n > 0:
            hot = sum(1 for c in cues if c.Kind > 0)
            mem = sum(1 for c in cues if c.Kind == 0)
            detail = []
            if hot:
                detail.append(f"{hot} Hot")
            if mem:
                detail.append(f"{mem} Mem")
            print(f"  {label[:60]:<60s}  {' + '.join(detail)}")
            track_details.append((content, cues))

    if total_cues == 0:
        print("  Keine Cues zum Loeschen gefunden.")
        sys.exit(0)

    affected = len(track_details)
    print(f"\n  Gesamt: {total_cues} Cues in {affected} Tracks\n")

    # --- Bestaetigung 1 ---
    print(f"{'=' * 60}")
    answer = input(f"  {total_cues} Cues in {affected} Tracks loeschen? (ja/nein): ").strip().lower()
    if answer != "ja":
        print("\n  Abgebrochen.")
        sys.exit(0)

    # --- Bestaetigung 2 ---
    print()
    print("  ACHTUNG: Diese Aktion kann NICHT rueckgaengig gemacht werden!")
    print("  Erstelle vorher ein Backup der Rekordbox-Datenbank.\n")
    confirm = input("  Zum Bestaetigen tippe 'LOESCHEN' ein: ").strip()
    if confirm != "LOESCHEN":
        print("\n  Abgebrochen — Eingabe war nicht 'LOESCHEN'.")
        sys.exit(0)

    # --- Loeschen ---
    print()
    deleted = 0
    for content, cues in track_details:
        artist = content.Artist.Name if content.Artist else ""
        title = content.Title or "?"
        label = f"{artist} - {title}" if artist else title

        for cue in cues:
            kind_str = _cue_kind_label(cue.Kind)
            ms = cue.InMsec or 0
            print(f"    Loesche {kind_str:5s} @ {ms / 1000:7.2f}s  — {label[:50]}")
            db.session.delete(cue)
            deleted += 1

    db.commit()

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"  ERGEBNIS: {deleted} Cues in {affected} Tracks geloescht.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
