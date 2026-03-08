"""
AutoCue — Rekordbox Playlist Generator (macOS)

Generiert intelligente Monats-Playlisten für Rekordbox.
Zwei Modi wählbar:
  1. Direkt in master.db schreiben (Attribute=4, SmartList-XML)
  2. XML-Export ins ./export/-Verzeichnis (importierbar über Rekordbox Bridge)

Struktur in Rekordbox:
  Year/
    2026/
      01-2026  (Smart-Playlist: stockDate >= 2026-01-01 AND <= 2026-01-31)
      02-2026
      …
      12-2026

Nutzung:
    python -m app.playlists
"""

import calendar
import logging
import os
import random
import sys
import uuid as _uuid_mod
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

log = logging.getLogger("autocue.playlists")

# ─── macOS-Konstanten ─────────────────────────────────────────────────────────

_DB_PATH = os.path.expanduser(
    "~/Library/Pioneer/rekordbox/master.db"
)
_EXPORT_DIR = Path(__file__).parent.parent / "export"

# Bekannte IDs aus bestehender DB-Struktur
_ROOT_PARENT_ID = "root"
_YEAR_FOLDER_ID = "1478233765"   # "Year"-Ordner (Attribute=1)
_YEAR_FOLDER_NAME = "Year"


# ═══════════════════════════════════════════════════════════════
#  Prozess-Schutz (macOS)
# ═══════════════════════════════════════════════════════════════

def is_rekordbox_running() -> bool:
    """
    Prüft ob Rekordbox auf macOS aktuell läuft (Prozessname: 'rekordbox').
    Nutzt psutil für plattformübergreifenden Prozess-Scan.
    """
    if not _HAVE_PSUTIL:
        log.warning("psutil nicht gefunden — Prozess-Check übersprungen.")
        return False
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info["name"] or "").lower() == "rekordbox":
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


# ═══════════════════════════════════════════════════════════════
#  DB-Modus: Direkt in master.db schreiben
# ═══════════════════════════════════════════════════════════════

def _get_engine():
    """
    Gibt den SQLAlchemy-Engine der verschlüsselten Rekordbox-DB zurück.
    Nutzt pyrekordbox für die SQLCipher-Entschlüsselung.
    """
    from app.beatgrid import open_db
    db = open_db()
    return db.engine


def _rand_id() -> str:
    """Generiert eine zufällige Playlist-ID (unsigned 32-bit int als String)."""
    return str(random.randint(100_000_000, 2**32 - 1))


def _rand_node_id() -> int:
    """Generiert eine zufällige SmartList-Node-ID (signed 32-bit int)."""
    return random.randint(-(2**31), 2**31 - 1)


def _new_uuid() -> str:
    """Generiert eine neue UUID für das UUID-Feld."""
    return str(_uuid_mod.uuid4())


def _now_str() -> str:
    """Aktuellen Zeitstempel im SQLite-Format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _next_seq(conn, parent_id: str) -> int:
    """Gibt den nächsten Seq-Wert innerhalb eines Ordners zurück."""
    from sqlalchemy import text
    row = conn.execute(
        text("SELECT MAX(Seq) FROM djmdPlaylist WHERE ParentID = :pid"),
        {"pid": parent_id}
    ).fetchone()
    return (row[0] or 0) + 1


def _next_usn(conn) -> int:
    """Gibt den nächsten rb_local_usn zurück."""
    from sqlalchemy import text
    row = conn.execute(
        text("SELECT MAX(rb_local_usn) FROM djmdPlaylist")
    ).fetchone()
    return (row[0] or 0) + 1


def find_folder_db(conn, name: str, parent_id: str) -> str | None:
    """
    Sucht einen Ordner (Attribute=1) anhand von Name und ParentID.
    Gibt die ID zurück oder None wenn nicht gefunden.
    """
    from sqlalchemy import text
    row = conn.execute(
        text("SELECT ID FROM djmdPlaylist "
             "WHERE Name=:name AND Attribute=1 AND ParentID=:pid"),
        {"name": name, "pid": parent_id}
    ).fetchone()
    return str(row[0]) if row else None


def find_smart_playlist_db(conn, name: str, parent_id: str) -> dict | None:
    """
    Sucht eine Smart-Playlist (Attribute=4) anhand von Name und ParentID.
    Gibt {'id': str, 'name': str} zurück oder None.
    """
    from sqlalchemy import text
    row = conn.execute(
        text("SELECT ID, Name FROM djmdPlaylist "
             "WHERE Name=:name AND Attribute=4 AND ParentID=:pid"),
        {"name": name, "pid": parent_id}
    ).fetchone()
    return {"id": str(row[0]), "name": row[1]} if row else None


def create_folder_db(conn, name: str, parent_id: str) -> str:
    """
    Legt einen neuen Ordner (Attribute=1) an.
    Gibt die neue ID zurück.
    """
    from sqlalchemy import text
    new_id  = _rand_id()
    seq     = _next_seq(conn, parent_id)
    usn     = _next_usn(conn)
    now     = _now_str()

    conn.execute(text("""
        INSERT INTO djmdPlaylist
            (ID, Seq, Name, ImagePath, Attribute, ParentID, SmartList, UUID,
             rb_data_status, rb_local_data_status, rb_local_deleted,
             rb_local_synced, usn, rb_local_usn, created_at, updated_at)
        VALUES
            (:id, :seq, :name, NULL, 1, :pid, NULL, :uuid,
             0, 0, 0,
             1, NULL, :usn, :now, :now)
    """), {"id": new_id, "seq": seq, "name": name, "pid": parent_id,
           "uuid": _new_uuid(), "usn": usn, "now": now})
    return new_id


def build_smart_list_xml(year: int, month: int) -> str:
    """
    Generiert den SmartList-XML-String für eine Monats-Smart-Playlist.

    Format (aus bestehenden Rekordbox-Einträgen reverse-engineered):
      Operator="3" = stockDate >= ValueLeft  (on or after)
      Operator="4" = stockDate <= ValueLeft  (on or before)

    Args:
        year:  Zieljahr
        month: Zielmonat (1–12)

    Returns:
        XML-String für die SmartList-Spalte in djmdPlaylist
    """
    first_day = f"{year}-{month:02d}-01"
    last_day  = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    node_id   = _rand_node_id()

    return (
        f'<?xml version="1.0" encoding="UTF-8"?> '
        f'<NODE Id="{node_id}" LogicalOperator="1" AutomaticUpdate="0">'
        f'<CONDITION PropertyName="stockDate" Operator="3" '
        f'ValueUnit="" ValueLeft="{first_day}" ValueRight=""/>'
        f'<CONDITION PropertyName="stockDate" Operator="4" '
        f'ValueUnit="" ValueLeft="{last_day}" ValueRight=""/>'
        f'</NODE>'
    )


def create_smart_playlist_db(conn, name: str, parent_id: str,
                              smart_xml: str) -> str:
    """
    Legt eine neue Smart-Playlist (Attribute=4) an.
    Gibt die neue ID zurück.
    """
    from sqlalchemy import text
    new_id  = _rand_id()
    seq     = _next_seq(conn, parent_id)
    usn     = _next_usn(conn)
    now     = _now_str()

    conn.execute(text("""
        INSERT INTO djmdPlaylist
            (ID, Seq, Name, ImagePath, Attribute, ParentID, SmartList, UUID,
             rb_data_status, rb_local_data_status, rb_local_deleted,
             rb_local_synced, usn, rb_local_usn, created_at, updated_at)
        VALUES
            (:id, :seq, :name, NULL, 4, :pid, :smartlist, :uuid,
             0, 0, 0,
             1, NULL, :usn, :now, :now)
    """), {"id": new_id, "seq": seq, "name": name, "pid": parent_id,
           "smartlist": smart_xml, "uuid": _new_uuid(),
           "usn": usn, "now": now})
    return new_id


def delete_smart_playlist_db(conn, playlist_id: str) -> None:
    """
    Löscht eine Smart-Playlist und ihre djmdSongPlaylist-Einträge.
    """
    from sqlalchemy import text
    conn.execute(
        text("DELETE FROM djmdSongPlaylist WHERE PlaylistID=:pid"),
        {"pid": playlist_id}
    )
    conn.execute(
        text("DELETE FROM djmdPlaylist WHERE ID=:id"),
        {"id": playlist_id}
    )


def _prompt_overwrite(name: str) -> bool:
    """
    Fragt den User ob eine existierende Playlist überschrieben werden soll.
    Gibt True zurück wenn ja.
    """
    while True:
        ans = input(f"  Playlist [{name}] existiert bereits. Überschreiben? (y/n): ").strip().lower()
        if ans in ("y", "j", "yes", "ja"):
            return True
        if ans in ("n", "no", "nein"):
            return False
        print("  ⚠  Bitte 'y' oder 'n' eingeben.")


def _ensure_year_folder(conn) -> str:
    """
    Stellt sicher dass der 'Year'-Ordner existiert.
    Gibt seine ID zurück.
    """
    existing = find_folder_db(conn, _YEAR_FOLDER_NAME, _ROOT_PARENT_ID)
    if existing:
        return existing
    # Neu anlegen
    print(f"  → Erstelle Ordner '{_YEAR_FOLDER_NAME}' …")
    return create_folder_db(conn, _YEAR_FOLDER_NAME, _ROOT_PARENT_ID)


def _ensure_year_subfolder(conn, year: int, year_folder_id: str) -> str:
    """
    Stellt sicher dass der Jahr-Unterordner (z.B. '2027') existiert.
    Gibt seine ID zurück.
    """
    name = str(year)
    existing = find_folder_db(conn, name, year_folder_id)
    if existing:
        return existing
    print(f"  → Erstelle Unterordner '{name}' …")
    return create_folder_db(conn, name, year_folder_id)


def create_yearly_playlists_db(year: int) -> dict:
    """
    Erstellt 12 Smart-Playlisten für das angegebene Jahr direkt in master.db.

    Ablauf:
      1. 'Year'-Ordner suchen / anlegen
      2. Jahr-Unterordner suchen / anlegen
      3. 12 Monats-Smart-Playlisten anlegen (mit Überschreib-Schutz)

    Returns:
        Statistik-Dict: {year, created, skipped, overwritten}
    """
    engine = _get_engine()

    with engine.begin() as conn:  # begin() = automatischer COMMIT/ROLLBACK
        year_folder_id = _ensure_year_folder(conn)
        year_sub_id    = _ensure_year_subfolder(conn, year, year_folder_id)

        created = skipped = overwritten = 0

        for month in range(1, 13):
            month_name  = f"{month:02d}-{year}"
            month_label = calendar.month_name[month]

            existing = find_smart_playlist_db(conn, month_name, year_sub_id)

            if existing:
                if not _prompt_overwrite(month_name):
                    print(f"    {month:02d}  {month_label:<12}  übersprungen")
                    skipped += 1
                    continue
                delete_smart_playlist_db(conn, existing["id"])
                overwritten += 1

            smart_xml = build_smart_list_xml(year, month)
            create_smart_playlist_db(conn, month_name, year_sub_id, smart_xml)
            print(f"    {month:02d}  {month_label:<12}  ✓  {month_name}")

            if not existing:
                created += 1

    return {
        "year": year, "mode": "db",
        "created": created, "overwritten": overwritten, "skipped": skipped,
    }


# ═══════════════════════════════════════════════════════════════
#  XML-Modus: Export ins ./export/-Verzeichnis
# ═══════════════════════════════════════════════════════════════

_XML_PRODUCT_NAME = "rekordbox"
_XML_PRODUCT_VER  = "6.8.5"
_XML_PRODUCT_CO   = "AlphaTheta"


def _parse_stock_date(raw) -> str | None:
    """Normalisiert ein Rekordbox-Datumsfeld auf 'YYYY-MM-DD'."""
    if not raw:
        return None
    if isinstance(raw, str):
        return raw[:10] if len(raw) >= 10 else None
    if isinstance(raw, (datetime,)):
        return raw.strftime("%Y-%m-%d")
    return None


def query_tracks_by_month(db, year: int, month: int) -> list[dict]:
    """
    Gibt alle Tracks zurück, die im angegebenen Monat/Jahr
    zur Bibliothek hinzugefügt wurden (StockDate).
    """
    first = f"{year}-{month:02d}-01"
    last  = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"

    result = []
    for content in db.get_content().all():
        raw = getattr(content, "StockDate", None) \
           or getattr(content, "DateCreated", None)
        date_str = _parse_stock_date(raw)
        if not date_str or not (first <= date_str <= last):
            continue

        artist = ""
        try:
            if content.Artist:
                artist = content.Artist.Name or ""
        except Exception:
            pass

        bpm_raw = content.BPM or 0
        result.append({
            "id":         str(content.ID),
            "title":      content.Title or "Unbekannt",
            "artist":     artist,
            "bpm":        f"{bpm_raw / 100:.2f}" if bpm_raw else "0.00",
            "date_added": date_str,
            "path":       content.FolderPath or "",
            "duration":   str(content.Length or 0),
            "rating":     str(content.Rating or 0),
        })
    return result


def _path_to_uri(path: str) -> str:
    """Konvertiert Dateipfad zu file://-URI."""
    if not path:
        return ""
    return "file://" + path.replace(" ", "%20")


def _add_track_to_collection(parent: Element, track: dict) -> None:
    """Fügt ein <TRACK>-Element zur COLLECTION hinzu."""
    t = SubElement(parent, "TRACK")
    t.set("TrackID",    track["id"])
    t.set("Name",       track["title"])
    t.set("Artist",     track["artist"])
    t.set("Composer",   "")
    t.set("Album",      "")
    t.set("Grouping",   "")
    t.set("Genre",      "")
    t.set("Kind",       "MP3 File")
    t.set("Size",       "0")
    t.set("TotalTime",  track["duration"])
    t.set("DiscNumber", "0")
    t.set("TrackNumber","0")
    t.set("Year",       "")
    t.set("AverageBpm", track["bpm"])
    t.set("DateAdded",  track["date_added"])
    t.set("BitRate",    "320")
    t.set("SampleRate", "44100")
    t.set("Comments",   "")
    t.set("PlayCount",  "0")
    t.set("Rating",     track["rating"])
    t.set("Location",   _path_to_uri(track["path"]))
    t.set("Remixer",    "")
    t.set("Tonality",   "")
    t.set("Label",      "")
    t.set("Mix",        "")


def build_xml(year: int, monthly_tracks: dict[int, list[dict]]) -> Element:
    """Baut das Rekordbox-XML-Dokument für den XML-Export."""
    all_tracks: dict[str, dict] = {}
    for tracks in monthly_tracks.values():
        for t in tracks:
            all_tracks.setdefault(t["id"], t)

    root = Element("DJ_PLAYLISTS")
    root.set("Version", "1.0.0")

    prod = SubElement(root, "PRODUCT")
    prod.set("Name",    _XML_PRODUCT_NAME)
    prod.set("Version", _XML_PRODUCT_VER)
    prod.set("Company", _XML_PRODUCT_CO)

    coll = SubElement(root, "COLLECTION")
    coll.set("Entries", str(len(all_tracks)))
    for track in all_tracks.values():
        _add_track_to_collection(coll, track)

    playlists = SubElement(root, "PLAYLISTS")
    root_node = SubElement(playlists, "NODE")
    root_node.set("Type", "0"); root_node.set("Name", "ROOT"); root_node.set("Count", "1")

    year_folder = SubElement(root_node, "NODE")
    year_folder.set("Type", "0"); year_folder.set("Name", "Year"); year_folder.set("Count", "1")

    year_node = SubElement(year_folder, "NODE")
    year_node.set("Type", "0"); year_node.set("Name", str(year)); year_node.set("Count", "12")

    for month in range(1, 13):
        tracks      = monthly_tracks.get(month, [])
        month_label = f"{month:02d}-{year}"
        mn = SubElement(year_node, "NODE")
        mn.set("Type", "1"); mn.set("Name", month_label)
        mn.set("KeyType", "0"); mn.set("Rows", str(len(tracks)))
        for track in tracks:
            ref = SubElement(mn, "TRACK")
            ref.set("Key", track["id"])

    return root


def _save_xml(root: Element, output_path: str) -> None:
    """Speichert XML-Baum als UTF-8-Datei mit XML-Deklaration."""
    tree = ElementTree(root)
    try:
        indent(tree, space="  ")
    except AttributeError:
        pass
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


def create_yearly_playlists_xml(year: int, export_dir: Path) -> dict:
    """
    Erstellt 12 Monats-Playlisten als Rekordbox-XML.
    Exportiert nach export_dir/rekordbox_YYYY.xml.

    Returns:
        Statistik-Dict: {year, total_tracks, monthly, output}
    """
    from app.beatgrid import open_db

    export_dir.mkdir(parents=True, exist_ok=True)
    output_path = export_dir / f"rekordbox_{year}.xml"

    # Überschreib-Schutz für XML-Datei
    if output_path.exists():
        if not _prompt_overwrite(str(output_path.name)):
            print(f"  Übersprungen.")
            return {"year": year, "mode": "xml", "total_tracks": 0,
                    "skipped": True, "output": str(output_path)}

    print(f"\n  Öffne Rekordbox-Datenbank …")
    db = open_db()

    monthly_tracks: dict[int, list[dict]] = {}
    total = 0

    print(f"  Filtere Tracks für {year}:")
    print()
    for month in range(1, 13):
        tracks = query_tracks_by_month(db, year, month)
        monthly_tracks[month] = tracks
        total += len(tracks)
        bar = "█" * min(len(tracks), 40) if tracks else "–"
        print(f"    {month:02d}  {calendar.month_name[month]:<12}  {len(tracks):5d}  {bar}")

    print()
    root = build_xml(year, monthly_tracks)
    _save_xml(root, str(output_path))

    return {
        "year": year, "mode": "xml",
        "total_tracks": total,
        "monthly": {m: len(t) for m, t in monthly_tracks.items()},
        "output": str(output_path),
        "skipped": False,
    }


# ═══════════════════════════════════════════════════════════════
#  Terminal-Menü
# ═══════════════════════════════════════════════════════════════

def _div(char: str = "─", w: int = 54) -> str:
    return "  " + char * w


def _prompt_year(default: int | None = None) -> int:
    """Fragt nach der Jahreszahl (Default: aktuelles Jahr)."""
    current = default or datetime.now().year
    raw = input(f"  Jahreszahl [{current}]: ").strip()
    if not raw:
        return current
    try:
        y = int(raw)
        if 2000 <= y <= 2100:
            return y
        print(f"  ⚠  Ungültige Jahreszahl — verwende {current}.")
    except ValueError:
        print(f"  ⚠  Keine Zahl — verwende {current}.")
    return current


def _action_yearly(mode: int, prefilled_year: int | None = None) -> None:
    """
    Aktion: Jahres-Playlisten erstellen.
    mode: 1=DB, 2=XML
    prefilled_year: wenn gesetzt, wird die Jahres-Eingabe übersprungen.
    """
    print()
    print(_div())
    print(f"  Jahres-Playlisten  ({'DB-Modus' if mode == 1 else 'XML-Export'})")
    print(_div())
    print()

    year = prefilled_year if prefilled_year is not None else _prompt_year()
    if prefilled_year is not None:
        print(f"  Jahreszahl: {year}")

    print()

    try:
        if mode == 1:
            stats = create_yearly_playlists_db(year)
            print()
            print(_div("═"))
            print(f"  ✓  Fertig!  {stats['created']} erstellt, "
                  f"{stats['overwritten']} überschrieben, "
                  f"{stats['skipped']} übersprungen")
            print(_div("═"))
            print()
            print("  → Rekordbox neu starten um Playlisten zu sehen.")

        else:  # XML
            stats = create_yearly_playlists_xml(year, _EXPORT_DIR)
            if not stats.get("skipped"):
                print()
                print(_div("═"))
                print(f"  ✓  Fertig!  {stats['total_tracks']} Tracks → {stats['output']}")
                print(_div("═"))
                print()
                print("  → Import: Rekordbox → Datei → Bibliothek → XML-Bibliothek importieren")

    except FileNotFoundError as e:
        print(f"\n  ✗  Datei nicht gefunden: {e}")
    except Exception as e:
        print(f"\n  ✗  Fehler: {e}")
        log.exception("Fehler in _action_yearly")

    print()


def _function_menu(mode: int) -> None:
    """
    Schritt 2: Funktion wählen (Jahres-Playlisten oder Platzhalter).
    Unterstützt Direkteingabe von Jahreszahlen.
    """
    mode_label = "DB-Modus (direkt schreiben)" if mode == 1 else "XML-Export (./export/)"
    print()
    print(_div("─"))
    print(f"  Funktion  ·  {mode_label}")
    print(_div("─"))
    print()
    print("  1   Jahres-Playlisten  (01-YYYY … 12-YYYY)")
    print("  ↳   oder Jahreszahl direkt eingeben  (z.B. 2027)")
    # Platzhalter für zukünftige Optionen:
    # print("  2   Genre-Playlisten")
    # print("  3   BPM-Range-Playlisten")
    print()
    print("  0   Zurück zum Modus-Menü")
    print()

    while True:
        try:
            choice = input("  Auswahl: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Abgebrochen.\n")
            sys.exit(0)

        # Direkte Jahreszahl?
        direct_year: int | None = None
        try:
            n = int(choice)
            if 2000 <= n <= 2100:
                direct_year = n
        except ValueError:
            pass

        if direct_year is not None:
            _action_yearly(mode, prefilled_year=direct_year)
            _print_function_options(mode_label)
            continue

        match choice:
            case "1":
                _action_yearly(mode)
                _print_function_options(mode_label)
            case "0" | "q" | "back" | "zurück":
                return
            case "":
                pass
            case _:
                print(f"  ⚠  Ungültige Auswahl: '{choice}'")


def _print_function_options(mode_label: str) -> None:
    """Zeigt Funktions-Optionen nach einer abgeschlossenen Aktion."""
    print(_div())
    print(f"  Funktion  ·  {mode_label}")
    print()
    print("  1   Jahres-Playlisten  (01-YYYY … 12-YYYY)")
    print("  ↳   oder Jahreszahl direkt eingeben  (z.B. 2027)")
    print()
    print("  0   Zurück zum Modus-Menü")
    print()


def main_menu() -> None:
    """
    Schritt 1: Arbeitsmodus wählen.
      1 = Direkt in master.db schreiben
      2 = XML-Export ins ./export/-Verzeichnis
      0 = Beenden
    """
    print()
    print(_div("═"))
    print("  AutoCue — Rekordbox Playlist Generator")
    print(_div("═"))
    print()
    print("  1   Direkt in Datenbank schreiben  (master.db)")
    print("  2   XML exportieren                 (./export/rekordbox_YYYY.xml)")
    print()
    print("  0   Beenden")
    print()

    while True:
        try:
            choice = input("  Modus: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Abgebrochen.\n")
            sys.exit(0)

        match choice:
            case "1":
                _function_menu(mode=1)
                # Zurück: Modus-Optionen wiederholen
                print(_div())
                print("  1   Direkt in Datenbank schreiben  (master.db)")
                print("  2   XML exportieren                 (./export/rekordbox_YYYY.xml)")
                print()
                print("  0   Beenden")
                print()

            case "2":
                _function_menu(mode=2)
                print(_div())
                print("  1   Direkt in Datenbank schreiben  (master.db)")
                print("  2   XML exportieren                 (./export/rekordbox_YYYY.xml)")
                print()
                print("  0   Beenden")
                print()

            case "0" | "q" | "quit" | "exit":
                print("\n  Auf Wiedersehen!\n")
                sys.exit(0)

            case "":
                pass

            case _:
                print(f"  ⚠  Ungültige Auswahl: '{choice}'")


# ═══════════════════════════════════════════════════════════════
#  Einstiegspunkt
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """Einstiegspunkt: macOS-Prozess-Check → Modus-Menü."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)

    # --- macOS Prozess-Schutz ---
    if is_rekordbox_running():
        print()
        print("  ⚠   Rekordbox ist aktiv.")
        print("      Bitte mit Cmd+Q beenden und Skript neu starten.")
        print()
        sys.exit(1)

    main_menu()


if __name__ == "__main__":
    main()
