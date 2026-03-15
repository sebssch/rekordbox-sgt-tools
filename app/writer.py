"""
v26 Database Writer
Schreibt generierte CuePoints in die Rekordbox master.db.
Sicherheitsfeatures: Rekordbox-Laufcheck, Backup, Duplikat-Erkennung,
optionale Bereinigung bestehender AutoCue-Eintraege.
"""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@contextlib.contextmanager
def _suppress_c_stderr():
    """
    Unterdrückt C-Library stderr-Output (z.B. libmpg123 ID3-Warnungen).
    Leitet File-Descriptor 2 temporaer nach /dev/null um — noetig weil
    C-Libraries Python's sys.stderr umgehen.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        os.close(devnull_fd)

from sqlalchemy import func

from app.beatgrid import open_db, find_content, _MASTER_DB_PATH
from app.cue_logic import CuePoint
from app import config as _cfg

log = logging.getLogger("autocue.writer")


@dataclass
class ProcessResult:
    """Ergebnis der kompletten Track-Verarbeitung."""
    cues: list       # list[CuePoint]
    decisions: list  # list[CueDecision] — fuer Triple-Check Status-Report
    grid: object     # BeatGrid — fuer Zeitformatierung
    n_written: int = 0

from pyrekordbox.db6.database import Rekordbox6Database
from pyrekordbox.db6.tables import DjmdCue


# --- Sicherheitsfunktionen ---

def check_rekordbox_running() -> bool:
    """Prueft ob Rekordbox laeuft. True = laeuft (NICHT schreiben!)."""
    # -x = exakter Prozessname, -i = case-insensitive
    # NICHT -f verwenden (matcht sonst pgrep selbst wegen "rekordbox" im Argument)
    result = subprocess.run(
        ["pgrep", "-xi", "rekordbox"],
        capture_output=True,
    )
    return result.returncode == 0


def _prune_backups(db_path: str, keep: int = 5) -> list[str]:
    """
    Loescht die aeltesten Backups der master.db, sodass hoechstens *keep*
    Backups erhalten bleiben. Gibt die Liste der geloeschten Pfade zurueck.

    Backup-Namensschema: <db_path>.backup_YYYYMMDD_HHMMSS
    """
    db_path = Path(db_path)
    pattern = db_path.name + ".backup_*"
    backups = sorted(
        db_path.parent.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    to_delete = backups[: max(0, len(backups) - keep)]
    deleted = []
    for old in to_delete:
        old.unlink()
        deleted.append(str(old))
        log.info("Altes Backup geloescht: %s", old)
    return deleted


def backup_database(db_path: str = None, keep: int = 5) -> str:
    """
    Erstellt ein Backup der master.db. Gibt den Backup-Pfad zurueck.
    Prueft anschliessend, dass hoechstens *keep* Backups erhalten bleiben
    (aelteste werden automatisch geloescht).
    """
    if db_path is None:
        db_path = _MASTER_DB_PATH

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    shutil.copy2(db_path, backup_path)
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    print(f"  Backup: {backup_path} ({size_mb:.1f} MB)")

    deleted = _prune_backups(db_path, keep=keep)
    if deleted:
        for d in deleted:
            print(f"  [alt] Backup entfernt: {Path(d).name}")

    return backup_path


# --- Duplikat-Erkennung ---

def _get_existing_cue_times(content) -> list[tuple[int, int]]:
    """Gibt bestehende Cue-Zeiten als (InMsec, Kind) zurueck."""
    return [(cue.InMsec, cue.Kind) for cue in content.Cues]


def _is_duplicate(new_ms: int, new_kind: int,
                  existing: list[tuple[int, int]],
                  tolerance_ms: int = 500) -> bool:
    """
    Prueft ob ein Cue-Punkt ein Duplikat ist.
    Hot Cues: exakter Kind-Match + Zeitnaehe.
    Memory Cues: nur Zeitnaehe (Kind=0).
    """
    for ex_ms, ex_kind in existing:
        if abs(new_ms - ex_ms) <= tolerance_ms:
            # Hot Cue: nur Duplikat wenn gleicher Slot
            if new_kind > 0 and ex_kind > 0:
                if new_kind == ex_kind:
                    return True
            # Memory Cue: Duplikat bei Zeitnaehe
            elif new_kind == 0 and ex_kind == 0:
                return True
            # Hot Cue am gleichen Ort wie Memory Cue: kein Duplikat
    return False


# --- Bestehende AutoCues entfernen ---

def remove_autocues(db: Rekordbox6Database, content,
                    dry_run: bool = False) -> int:
    """
    Entfernt alle AutoCue-Cues (erkennt aktuellen Prefix + Legacy 'v3:' / 'AutoCue').
    Gibt die Anzahl der entfernten Cues zurueck.
    """
    to_remove = []
    for cue in content.Cues:
        comment = cue.Comment or ""
        if _cfg.is_autocue_comment(comment):
            to_remove.append(cue)

    if not to_remove:
        return 0

    if dry_run:
        for cue in to_remove:
            kind_str = "Mem" if cue.Kind == 0 else f"Hot {chr(64 + cue.Kind)}"
            print(f"    [DRY-RUN] Wuerde entfernen: {kind_str} @ "
                  f"{cue.InMsec}ms \"{cue.Comment}\"")
        return len(to_remove)

    for cue in to_remove:
        db.session.delete(cue)
    db.commit()

    return len(to_remove)


# --- CuePoint → DjmdCue ---

def _next_cue_id(db: Rekordbox6Database, offset: int = 0) -> str:
    """Generiert die naechste freie DjmdCue.ID."""
    with db.session.no_autoflush:
        max_id = db.session.query(func.max(DjmdCue.ID)).scalar()
    return str(int(max_id or 0) + 1 + offset)


def _create_djmd_cue(db: Rekordbox6Database, content,
                     cue: CuePoint, id_offset: int) -> DjmdCue:
    """Konvertiert einen CuePoint in ein DjmdCue-Objekt."""
    now = datetime.now()
    cue_id = _next_cue_id(db, id_offset)

    # Konfigurierter Prefix (z.B. "A:") + beschreibender Kommentar
    prefix = _cfg.get_cue_prefix()
    comment = f"{prefix} {cue.comment}"

    # Farbe: CuePoint-Wert hat Vorrang; sonst aus config.yaml
    _color_map = {0: "memory", 1: "hot_a", 2: "hot_b", 3: "hot_c"}
    colors_cfg = _cfg.get("colors", {})
    color_key = _color_map.get(cue.kind, "memory")
    color = cue.color if cue.color != -1 else colors_cfg.get(color_key, -1)

    return DjmdCue(
        ID=cue_id,
        ContentID=str(content.ID),
        InMsec=cue.time_ms,
        InFrame=cue.time_frame,
        InMpegFrame=0,
        InMpegAbs=0,
        OutMsec=-1,
        OutFrame=0,
        OutMpegFrame=0,
        OutMpegAbs=0,
        Kind=cue.kind,
        Color=color,
        ColorTableIndex=0,
        ActiveLoop=0,
        Comment=comment,
        BeatLoopSize=0,
        CueMicrosec=0,
        InPointSeekInfo="",
        OutPointSeekInfo="",
        ContentUUID=str(content.UUID),
        UUID=str(_uuid.uuid4()),
        rb_data_status=0,
        rb_local_data_status=0,
        rb_local_deleted=0,
        rb_local_synced=0,
        usn=0,
        rb_local_usn=0,
        created_at=now,
        updated_at=now,
    )


# --- Hauptfunktion ---

def write_cues(audio_path: str,
               cues: list[CuePoint],
               db: Rekordbox6Database = None,
               content=None,
               dry_run: bool = False,
               clean: bool = False,
               skip_safety: bool = False) -> int:
    """
    Schreibt CuePoints in die Rekordbox master.db.

    Args:
        audio_path: Pfad zur Audio-Datei (muss in Rekordbox importiert sein)
        cues: Liste von CuePoint-Objekten aus cue_logic
        db: Offene DB-Verbindung (oder None fuer auto-open)
        content: DjmdContent-Objekt (oder None fuer auto-lookup)
        dry_run: Nur anzeigen, nicht schreiben
        clean: Bestehende AutoCue-Eintraege vorher entfernen
        skip_safety: True = kein Rekordbox-Check/Backup (wenn batch.py das uebernimmt)

    Returns:
        Anzahl der geschriebenen Cues
    """
    # --- Sicherheitschecks ---
    if not dry_run and not skip_safety:
        if check_rekordbox_running():
            print("FEHLER: Rekordbox laeuft noch! Bitte erst beenden.")
            sys.exit(1)

        print("  Sicherheitscheck: Rekordbox nicht aktiv ✓")
        backup_database()

    # --- DB oeffnen ---
    if db is None:
        db = open_db()

    # --- Track finden ---
    if content is None:
        content = find_content(db, audio_path)
    if content is None:
        print(f"FEHLER: Track nicht in Rekordbox-DB: {audio_path}")
        print("Importiere den Track zuerst in Rekordbox.")
        return 0

    print(f"  Track: {content.Title} (ID={content.ID})")

    # --- Optional: alte AutoCues entfernen ---
    if clean:
        removed = remove_autocues(db, content, dry_run=dry_run)
        if removed > 0:
            print(f"  {removed} alte AutoCues {'wuerden entfernt' if dry_run else 'entfernt'}")
            # Existing-Liste neu laden nach Bereinigung
            if not dry_run:
                db.session.expire(content)

    # --- Bestehende Cues pruefen ---
    existing = _get_existing_cue_times(content)
    if existing:
        print(f"  Bestehende Cues: {len(existing)}")

    # --- Cues schreiben ---
    added = 0
    skipped = 0

    for cue in cues:
        if _is_duplicate(cue.time_ms, cue.kind, existing):
            if dry_run:
                kind_str = "Mem" if cue.kind == 0 else f"Hot {chr(64 + cue.kind)}"
                m = int(cue.time_sec // 60)
                s = cue.time_sec % 60
                print(f"    Duplikat: {kind_str} [{m}:{s:05.2f}] — {cue.comment}")
            skipped += 1
            continue

        if dry_run:
            kind_str = "Mem" if cue.kind == 0 else f"Hot {chr(64 + cue.kind)}"
            m = int(cue.time_sec // 60)
            s = cue.time_sec % 60
            print(f"    [DRY-RUN] {kind_str} [{m}:{s:05.2f}] — {cue.comment}")
        else:
            djmd_cue = _create_djmd_cue(db, content, cue, added)
            db.session.add(djmd_cue)
            existing.append((cue.time_ms, cue.kind))

        added += 1

    # --- Commit ---
    if not dry_run and added > 0:
        db.commit()

    # --- Zusammenfassung ---
    action = "wuerden geschrieben" if dry_run else "geschrieben"
    print(f"\n  Ergebnis: {added} Cues {action}, {skipped} Duplikate uebersprungen")

    return added


# --- Komplette v3-Pipeline ---

def process_track(audio_path: str,
                  dry_run: bool = True,
                  clean: bool = False,
                  db: Rekordbox6Database = None,
                  content=None,
                  skip_safety: bool = False,
                  verbose: bool = True,
                  use_cbr: bool = True) -> "ProcessResult":
    """
    Komplette v27-Pipeline: Analyse → CBR → PWAV-ML → Quad-Check → Cue-Schreiben.
    Falls kein PWAV-Modell vorhanden: Fallback auf Triple-Check.
    Standardmaessig im Dry-Run-Modus!

    Args:
        audio_path: Pfad zur Audio-Datei
        dry_run: True = nur anzeigen (default!), False = wirklich schreiben
        clean: True = alte AutoCues vorher entfernen
        db: Offene DB-Verbindung (oder None fuer auto-open)
        content: DjmdContent-Objekt (oder None fuer auto-lookup)
        skip_safety: True = kein Rekordbox-Check/Backup (batch.py uebernimmt)
        verbose: True = ausfuehrliche Ausgabe, False = kompakt
        use_cbr: True = Case-Based Reasoning nutzen (adaptive Cues)

    Returns:
        ProcessResult mit cues, decisions, grid, n_written
    """
    import warnings
    warnings.filterwarnings("ignore")
    import librosa

    from app.beatgrid import get_anlz_path, read_beatgrid
    from app.segments import analyze_track
    from app.cue_logic import generate_cues, print_cues
    from app.phrase_reader import read_phrases
    from app.validator import build_status_report

    resolved = str(Path(audio_path).resolve())
    filename = Path(audio_path).stem
    mode = "[DRY-RUN]" if dry_run else "[LIVE]"
    version = _cfg.get("version", "26")

    log.info("Verarbeite: %s  %s", filename, mode)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  AutoCue v{version} {mode}: {filename}")
        print(f"{'='*60}")

    # --- DB + Beatgrid ---
    if db is None:
        db = open_db()

    if verbose:
        print("\n1. Beatgrid laden...")
    dat_path, ext_path = get_anlz_path(db, resolved)
    grid = read_beatgrid(dat_path)
    if verbose:
        print(f"   {grid.count} Beats, {grid.bpm:.0f} BPM, {grid.bar_count} Bars")

    # --- Rekordbox-Phrasen (PSSI aus .EXT-Datei) ---
    phrases = read_phrases(ext_path, grid)
    if verbose:
        if phrases:
            print(f"   Phrasen: {len(phrases)} PSSI-Segmente "
                  f"({', '.join(sorted({p.kind_name for p in phrases}))})")
        else:
            print("   Phrasen: keine PSSI-Daten (akust. Segmente als Fallback)")

    # --- Audio laden ---
    if verbose:
        print("2. Audio laden...")
    with _suppress_c_stderr():
        y, sr = librosa.load(resolved, sr=22050, mono=True)
    duration = len(y) / sr
    if verbose:
        print(f"   {duration:.0f}s ({int(duration//60)}:{duration%60:05.2f})")

    # --- Analyse ---
    if verbose:
        print("3. Akustische Analyse...")
    analysis = analyze_track(y, sr, grid)
    if verbose:
        print(f"   Komplexitaet: {analysis.complexity:.3f}")
        print(f"   {len(analysis.segments)} Segmente erkannt:")
        for i, seg in enumerate(analysis.segments, 1):
            m = int(seg.start_time // 60)
            s = seg.start_time % 60
            print(f"     {i}. [{m}:{s:05.2f}] {seg.kind:8s} E={seg.energy_mean:.2f}")

    # --- Content auto-resolve (fuer MIK-Lookup und spaetere Schritte) ---
    if content is None:
        content = find_content(db, resolved)

    # --- MIK-Hotspots: Externe Cue-Positionen aus Mixed In Key ---
    try:
        from app.mik_scraper import get_mik_data
        mik_artist = (getattr(content, 'ArtistName', '') or '') if content else ''
        mik_title  = (getattr(content, 'Title',      '') or '') if content else ''
        mik_data = get_mik_data(resolved, mik_artist, mik_title)
        if mik_data and mik_data.cue_times:
            analysis.mik_hotspots = mik_data.cue_times
            if verbose:
                times_str = ", ".join(f"{t:.1f}s" for t in mik_data.cue_times)
                print(f"   MIK [{mik_data.source}]: {len(mik_data.cue_times)} Cues"
                      f" (Energy={mik_data.energy}) → {times_str}")
        else:
            if verbose:
                print("   MIK: kein Treffer")
    except Exception as _mik_err:
        if verbose:
            print(f"   [WARN] MIK-Lookup: {_mik_err}")

    # --- CBR: Zwillinge finden + adaptive Parameter ---
    cbr_result = None
    if use_cbr:
        try:
            from app.cbr import run_cbr, print_cbr_result
            if verbose:
                print("3b. Case-Based Reasoning...")
            cbr_result = run_cbr(analysis, resolved, db=db)
            if verbose and cbr_result.twins:
                print_cbr_result(cbr_result)
            elif verbose:
                print(f"   {cbr_result.explanation}")
        except FileNotFoundError:
            if verbose:
                print("   CBR: Keine Vektor-DB. Nutze regelbasierte Logik.")
        except Exception as e:
            if verbose:
                print(f"   CBR-Fehler: {e}. Nutze regelbasierte Logik.")

    # --- Recursive Intelligence: Auto-Korrektur-Offsets ---
    # Genre + Key fuer Fingerprint auflösen (content koennte None sein)
    _fp_content = content or find_content(db, resolved)
    _fp_genre, _fp_key = "", ""
    if _fp_content is not None:
        try:
            _fp_genre = (getattr(_fp_content, "GenreName", None) or "").strip().lower()
        except Exception:
            pass
        try:
            _fp_key = (getattr(_fp_content, "KeyName", None) or "").strip()
        except Exception:
            pass

    learned_offsets = {"hot_a_offset_ms": 0, "hot_c_offset_ms": 0}
    _use_learned = _cfg.get("use_learned_offsets", True)
    if _use_learned:
        try:
            from app.learning_db import get_db as _get_ldb, get_auto_correction_offsets
            _lconn = _get_ldb()
            _conf_threshold = _cfg.get("auto_correction_confidence_threshold", 0.80)
            learned_offsets = get_auto_correction_offsets(
                _lconn, grid.bpm,
                genre=_fp_genre,
                duration_s=float(duration),
                key=_fp_key,
                confidence_threshold=float(_conf_threshold),
            )
            _lconn.close()
            if verbose and (learned_offsets.get("hot_a_offset_ms")
                            or learned_offsets.get("hot_c_offset_ms")):
                print(f"   Auto-Korrektur: "
                      f"A={learned_offsets['hot_a_offset_ms']}ms  "
                      f"C={learned_offsets['hot_c_offset_ms']}ms")
        except Exception:
            pass  # Kein Abbruch bei fehlendem learning_db
    elif verbose:
        print("   Learned Offsets: deaktiviert (use_learned_offsets: false)")

    # --- ML Prediction (LightGBM, ersetzt PWAV Random Forest) ---
    pwav_candidates = None
    _conf = _cfg.load_config()
    if _conf.get("cue_engine", "auto") in ("ml", "pwav", "auto"):
        try:
            from app.ml_predictor import predict_cue_positions
            _mik_for_ml = getattr(analysis, '_mik_data_raw', None)
            try:
                from app.mik_scraper import get_mik_data as _get_mik_ml
                _mik_for_ml = _get_mik_ml(
                    resolved,
                    (getattr(content, 'ArtistName', '') or '') if content else '',
                    (getattr(content, 'Title', '') or '') if content else '',
                )
            except Exception:
                pass
            pwav_candidates = predict_cue_positions(
                content, dat_path, ext_path, grid, db,
                mik_data=_mik_for_ml,
            )
            if verbose and pwav_candidates:
                n = sum(len(v) for v in pwav_candidates.values())
                print(f"   ML-Predictor: {n} Kandidaten generiert")
        except FileNotFoundError:
            pass  # Kein trainiertes Modell → Fallback auf Triple-Check
        except Exception as e:
            log.warning("ML-Predictor Fehler: %s", e)

    # --- Cue-Generierung (Quad-Check) ---
    check_label = "Quad-Check" if pwav_candidates else "Triple-Check"
    if verbose:
        print(f"4. Cues generieren ({check_label})...")
    cues, decisions = generate_cues(
        analysis,
        cbr=cbr_result,
        phrases=phrases,
        learned_offsets=learned_offsets,
        pwav_candidates=pwav_candidates,
    )
    if verbose:
        print(f"\n   {check_label} Ergebnis:")
        for line in build_status_report(decisions):
            print(line)
        print_cues(cues, grid)

    # --- Schreiben ---
    if verbose:
        print("5. In Datenbank schreiben...")
    n_written = write_cues(resolved, cues, db=db, content=content,
                           dry_run=dry_run, clean=clean, skip_safety=skip_safety)

    # --- Prediction Log (nur bei echtem Schreiben) ---
    if not dry_run:
        _content = content
        if _content is None:
            _content = find_content(db, resolved)
        if _content is not None:
            _log_prediction(_content, cues, cbr_result)
            _log_prediction_db(_content, cues, cbr_result)

    return ProcessResult(cues=cues, decisions=decisions, grid=grid, n_written=n_written)


# --- Prediction Log ---

_PREDICTIONS_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "predictions.jsonl"
)


def _log_prediction(content, cues: list[CuePoint], cbr_result=None) -> None:
    """
    Speichert die v3-Vorhersagen in predictions.jsonl (append-only).
    Dient als Referenz fuer den Feedback-Loop: Was hat v3 vorhergesagt
    vs. was hat der User spaeter in Rekordbox geaendert.
    """
    cbr_data = {}
    if cbr_result is not None and cbr_result.twins:
        best = cbr_result.twins[0]
        cbr_data = {
            "cue_spacing": cbr_result.cue_spacing,
            "structure_label": cbr_result.structure_label,
            "best_twin": best.get("title", ""),
            "best_twin_artist": best.get("artist", ""),
            "twin_similarity": round(best.get("similarity", 0), 3),
        }
        if cbr_result.hot_cue_pattern:
            cbr_data["hot_cue_pattern"] = cbr_result.hot_cue_pattern
        if cbr_result.energy_match:
            cbr_data["energy_match"] = {
                k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in cbr_result.energy_match.items()
            }

    entry = {
        "content_id": str(content.ID),
        "title": content.Title or "",
        "bpm": (content.BPM or 0) / 100.0,
        "duration": content.Length or 0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cbr": cbr_data,
        "cues": [
            {
                "time_ms": c.time_ms,
                "kind": c.kind,
                "name": c.name,
                "comment": c.comment,
            }
            for c in cues
        ],
    }

    os.makedirs(os.path.dirname(_PREDICTIONS_LOG), exist_ok=True)

    with open(_PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_prediction_db(content, cues: list[CuePoint], cbr_result=None) -> None:
    """
    Speichert Vorhersagen in der agent_learning.db (SQLite).
    Ergaenzt den JSONL-Log fuer strukturierte Abfragen und Inferenz.
    """
    try:
        from app.learning_db import get_db, upsert_track, log_predictions

        conn = get_db()

        # Track-Metadaten speichern (inkl. Genre + Key aus Rekordbox = MP3-Tag)
        bpm = (content.BPM or 0) / 100.0
        try:
            genre = (getattr(content, "GenreName", None) or "").strip().lower()
        except Exception:
            genre = ""
        try:
            key = (getattr(content, "KeyName", None) or "").strip()
        except Exception:
            key = ""
        upsert_track(
            conn,
            content_id=str(content.ID),
            title=content.Title or "",
            artist=content.Artist.Name if hasattr(content, 'Artist') and content.Artist else "",
            bpm=bpm,
            duration=content.Length or 0,
            genre=genre,
            key=key,
        )

        # CBR-Daten extrahieren
        cbr_spacing = 32
        twin_id = None
        twin_sim = 0.0
        if cbr_result is not None and cbr_result.twins:
            cbr_spacing = cbr_result.cue_spacing
            best = cbr_result.twins[0]
            twin_id = best.get("content_id", "")
            twin_sim = best.get("similarity", 0.0)

        # Cues als Dicts fuer learning_db
        cue_dicts = [
            {"time_ms": c.time_ms, "kind": c.kind, "comment": c.comment}
            for c in cues
        ]

        log_predictions(
            conn, str(content.ID), cue_dicts,
            cbr_spacing=cbr_spacing,
            cbr_twin_id=twin_id,
            cbr_twin_similarity=twin_sim,
        )

        conn.close()
    except Exception as e:
        # Kein Abbruch bei Logging-Fehler
        print(f"  [WARN] learning_db Log fehlgeschlagen: {e}")
