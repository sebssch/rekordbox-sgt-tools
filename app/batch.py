"""
v26 Playlist-Automator
Verarbeitet alle Tracks einer Rekordbox-Playlist automatisch.
Sucht nach der konfigurierten Analyse-Playlist, prueft ob Tracks
bereits analysiert sind, und fuehrt die v26-Pipeline batch-weise aus.

Nutzung:
    python -m app.batch                      # Dry-Run (standard)
    python -m app.batch --live               # Wirklich schreiben
    python -m app.batch --live --clean       # Alte Cues vorher entfernen
    python -m app.batch --live --force       # Auch bereits analysierte neu verarbeiten
    python -m app.batch --playlist "MyList"  # Andere Playlist
"""

import argparse
import logging
import sys
import time
import traceback

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.markup import escape

from app import config as _cfg
from app.beatgrid import open_db, _MASTER_DB_PATH
from app.writer import (
    check_rekordbox_running,
    backup_database,
    process_track,
    ProcessResult,
)
from app.validator import build_status_report

# --- Logging + Console ---
console = Console()
log = logging.getLogger("autocue.batch")


# --- Playlist-Lookup ---

def find_playlist(db, name: str):
    """
    Sucht eine Playlist nach Name in der Rekordbox-DB.
    Gibt das DjmdPlaylist-Objekt zurueck oder None.
    """
    playlists = db.get_playlist(Name=name)

    # get_playlist gibt Query zurueck wenn kein ID angegeben
    results = playlists.all() if hasattr(playlists, "all") else [playlists]

    for pl in results:
        if pl is not None and pl.is_playlist:
            return pl

    return None


def get_playlist_tracks(db, playlist) -> list:
    """
    Gibt alle DjmdContent-Objekte einer Playlist zurueck.
    Sortiert nach TrackNo (Reihenfolge in der Playlist).
    """
    contents = db.get_playlist_contents(playlist)

    if hasattr(contents, "all"):
        return contents.all()
    return list(contents)


# --- Already-Processed Check ---

def is_already_processed(content) -> bool:
    """
    Prueft ob ein Track bereits AutoCue-Cues hat.
    Erkennt aktuellen Prefix (z.B. 'A:') sowie Legacy 'v3:' und 'v26:'.
    """
    for cue in content.Cues:
        comment = cue.Comment or ""
        if _cfg.is_autocue_comment(comment):
            return True
    return False


# --- Batch-Verarbeitung ---

def run_playlist(
    playlist_name: str | None = None,
    dry_run: bool = True,
    clean: bool = False,
    force: bool = False,
) -> dict:
    """
    Verarbeitet alle Tracks einer Rekordbox-Playlist.

    Args:
        playlist_name: Name der Playlist in Rekordbox (None = aus config.yaml)
        dry_run: True = nur anzeigen (default!), False = wirklich schreiben
        clean: True = bestehende AutoCue-Cues vorher entfernen
        force: True = auch bereits analysierte Tracks neu verarbeiten

    Returns:
        dict mit Statistiken: processed, skipped, errors, total
    """
    # Setup: Logging + Config
    _cfg.setup_logging()
    cfg = _cfg.load_config()

    # Pyrekordbox-Warnungen unterdrücken (manuell erstellte Playlists sind nicht
    # in masterPlaylists6.xml — diese Warnung ist harmlos und zu gesprächig)
    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)
    version = cfg.get("version", "26")

    if playlist_name is None:
        playlist_name = cfg.get("analyse_playlist", "--analyse-tracks")

    mode_label = "[bold red]LIVE[/bold red]" if not dry_run else "[bold yellow]DRY-RUN[/bold yellow]"
    mode_str   = "LIVE" if not dry_run else "DRY-RUN"

    # --- Header ---
    console.print()
    console.print(Panel(
        f"[bold cyan]AutoCue v{version}[/bold cyan]  {mode_label}\n"
        f"[dim]Playlist:[/dim] [white]{escape(playlist_name)}[/white]",
        title="[bold]Rekordbox AutoCue[/bold]",
        border_style="cyan",
    ))

    log.info("Batch gestartet — Playlist=%s  mode=%s", playlist_name, mode_str)

    # --- Sicherheitschecks ---
    if not dry_run:
        if check_rekordbox_running():
            console.print("[bold red]FEHLER:[/bold red] Rekordbox läuft noch! Bitte erst beenden.")
            log.error("Rekordbox laeuft — Abbruch")
            sys.exit(1)
        console.print("[green]✓[/green] Sicherheitscheck: Rekordbox nicht aktiv")
        backup_path = backup_database()
        log.info("Backup erstellt: %s", backup_path)

    # --- DB oeffnen ---
    console.print("[dim]Öffne Datenbank...[/dim]")
    db = open_db()

    # --- Review & Learn: Automatischer Sync-Lauf VOR der Analyse ---
    if not dry_run:
        try:
            from app.learning_db import review_and_learn
            console.print("[dim]Review & Learn (Sync-Lauf)...[/dim]")
            review_and_learn(rekordbox_db=db)
            log.info("Review & Learn abgeschlossen")
        except Exception as e:
            console.print(f"[yellow]⚠ Review & Learn fehlgeschlagen:[/yellow] {e}")
            log.warning("Review & Learn fehlgeschlagen: %s", e)

    # --- Playlist finden ---
    playlist = find_playlist(db, playlist_name)
    if playlist is None:
        console.print(
            f"[bold red]FEHLER:[/bold red] Playlist [bold]{escape(playlist_name)}[/bold] nicht gefunden!\n"
            "[dim]Erstelle in Rekordbox eine Playlist mit diesem Namen "
            "und füge die zu analysierenden Tracks hinzu.[/dim]"
        )
        log.error("Playlist nicht gefunden: %s", playlist_name)
        return {"processed": 0, "skipped": 0, "errors": 0, "total": 0}

    console.print(f"[green]✓[/green] Playlist: [bold]{escape(playlist.Name)}[/bold] (ID={playlist.ID})")

    # --- Tracks laden ---
    tracks = get_playlist_tracks(db, playlist)
    total = len(tracks)
    console.print(f"  [dim]{total} Tracks geladen[/dim]")

    if total == 0:
        console.print("\n[yellow]Keine Tracks zum Verarbeiten.[/yellow]")
        return {"processed": 0, "skipped": 0, "errors": 0, "total": 0}

    # --- Already-Processed zaehlen ---
    already = sum(1 for t in tracks if is_already_processed(t))
    if already > 0 and not force:
        console.print(f"  [dim]Bereits analysiert: {already} (werden übersprungen)[/dim]")
        console.print(f"  [dim]Zu verarbeiten: {total - already}[/dim]")
    elif already > 0 and force:
        console.print(f"  [yellow]⚡ force=True — {already} bereits analysierte Tracks werden neu verarbeitet[/yellow]")

    # --- Batch-Loop mit rich.progress ---
    stats = {"processed": 0, "skipped": 0, "errors": 0, "total": total}
    error_tracks: list[tuple[str, str]] = []
    start_time = time.time()

    console.print()

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=30, style="cyan", complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"AutoCue v{version}", total=total)

        for i, content in enumerate(tracks):
            title  = content.Title or "Unbekannt"
            artist = content.Artist.Name if hasattr(content, 'Artist') and content.Artist else ""
            label  = f"{artist} – {title}" if artist else title
            audio_path = content.FolderPath

            progress.update(task, description=f"[cyan]{escape(label[:50])}[/cyan]")

            # Skip wenn bereits verarbeitet
            if is_already_processed(content) and not force:
                stats["skipped"] += 1
                log.debug("Uebersprungen (bereits analysiert): %s", title)
                progress.advance(task)
                continue

            # Skip wenn kein Audio-Pfad
            if not audio_path:
                console.print(f"  [yellow]⚠[/yellow] Kein Dateipfad: {escape(title)}")
                log.warning("Kein Dateipfad: %s", title)
                stats["errors"] += 1
                error_tracks.append((title, "Kein Dateipfad"))
                progress.advance(task)
                continue

            log.info("[%d/%d] %s", i + 1, total, title)

            try:
                result = process_track(
                    audio_path=audio_path,
                    dry_run=dry_run,
                    clean=clean,
                    db=db,
                    content=content,
                    skip_safety=True,
                    verbose=False,
                )

                result_cues = result.cues

                # Kompakt-Ausgabe: Hot Cues
                hot_a = next((c for c in result_cues if c.kind == 1), None)
                hot_b = next((c for c in result_cues if c.kind == 2), None)
                hot_c = next((c for c in result_cues if c.kind == 3), None)
                mem_n = sum(1 for c in result_cues if c.kind == 0)

                def _fmt(c):
                    if c is None:
                        return "[dim]—[/dim]"
                    m = int(c.time_sec // 60)
                    s = c.time_sec % 60
                    return f"[green]{m}:{s:05.2f}[/green]"

                console.print(
                    f"  [green]✓[/green] {escape(label[:45]):<48}"
                    f"  A:{_fmt(hot_a)}  B:{_fmt(hot_b)}  C:{_fmt(hot_c)}"
                    f"  Mem:[cyan]{mem_n}[/cyan]"
                )

                # Per-Cue Triple-Check Status-Report
                for line in build_status_report(result.decisions):
                    console.print(line)

                # Memory Cue Details
                mem_cues = [c for c in result_cues if c.kind == 0]
                if mem_cues:
                    parts = []
                    for mc in mem_cues:
                        m = int(mc.time_sec // 60)
                        s = mc.time_sec % 60
                        parts.append(f"[dark_orange]{m}:{s:05.2f}[/dark_orange] {escape(mc.comment[:20])}")
                    console.print(f"    [dark_orange]Mem:[/dark_orange] " + "  |  ".join(parts))

                stats["processed"] += 1
                log.info("OK: %s  Cues=%d", title, len(result_cues))

            except FileNotFoundError as e:
                console.print(f"  [red]✗[/red] Datei nicht gefunden: {escape(str(e)[:60])}")
                log.error("FileNotFoundError: %s — %s", title, e)
                stats["errors"] += 1
                error_tracks.append((title, f"FileNotFoundError: {e}"))

            except Exception as e:
                console.print(f"  [red]✗[/red] Fehler bei '{escape(title[:40])}': {escape(str(e)[:60])}")
                log.error("Fehler: %s — %s", title, e, exc_info=True)
                stats["errors"] += 1
                error_tracks.append((title, str(e)))

            finally:
                progress.advance(task)

    # --- Zusammenfassung ---
    elapsed  = time.time() - start_time
    minutes  = int(elapsed // 60)
    seconds  = elapsed % 60

    console.print()

    summary = Table(title=f"Zusammenfassung · AutoCue v{version} · {mode_str}", show_header=False)
    summary.add_column("", style="dim")
    summary.add_column("", justify="right")
    summary.add_row("Gesamt",         f"[white]{stats['total']}[/white]")
    summary.add_row("Verarbeitet",    f"[green]{stats['processed']}[/green]")
    summary.add_row("Übersprungen",   f"[dim]{stats['skipped']}[/dim]")
    summary.add_row("Fehler",         f"[red]{stats['errors']}[/red]" if stats["errors"] else f"[dim]{stats['errors']}[/dim]")
    summary.add_row("Dauer",          f"[cyan]{minutes}:{seconds:05.2f}[/cyan]")
    console.print(summary)

    if error_tracks:
        console.print("\n[bold red]Fehlerhafte Tracks:[/bold red]")
        for t_title, err in error_tracks:
            console.print(f"  [red]–[/red] {escape(t_title)}: [dim]{escape(err[:80])}[/dim]")

    log.info(
        "Batch abgeschlossen: processed=%d skipped=%d errors=%d elapsed=%.1fs",
        stats["processed"], stats["skipped"], stats["errors"], elapsed,
    )
    console.print()
    return stats


# --- CLI Entrypoint ---

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.batch",
        description="AutoCue v26 — Rekordbox Playlist Automator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m app.batch                    # Dry-Run (Standard, nichts wird geschrieben)
  python -m app.batch --live             # Cues wirklich in Rekordbox schreiben
  python -m app.batch --live --clean     # Alte AutoCue-Cues zuerst entfernen
  python -m app.batch --live --force     # Auch bereits analysierte Tracks neu verarbeiten
  python -m app.batch --playlist "MyPl"  # Andere Playlist aus Rekordbox verwenden
        """,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Cues wirklich schreiben (Standard: Dry-Run)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Bestehende AutoCue-Cues vorher entfernen"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Auch bereits analysierte Tracks neu verarbeiten"
    )
    parser.add_argument(
        "--playlist",
        metavar="NAME",
        default=None,
        help="Name der Rekordbox-Playlist (Standard: aus config.yaml)"
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_playlist(
        playlist_name=args.playlist,
        dry_run=not args.live,
        clean=args.clean,
        force=args.force,
    )
