"""
AutoCue v27 — Prediction vs. Manuell Vergleich

Liest 5 Tracks mit manuell gesetzten Hot Cues aus der Rekordbox-DB,
fuehrt einen Dry-Run durch und vergleicht die Vorhersagen mit den
manuellen Tags. Ergebnis wird als Markdown-Log geschrieben.

KEINE Datenbank-Schreibvorgaenge. Rein analytisch.

Aufruf:
    .venv/bin/python tools/compare_predictions.py
"""

from __future__ import annotations

import sys
import os
from datetime import date
from pathlib import Path

# Projektroot im PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.beatgrid import open_db
from app import config as _cfg


# Content-IDs der bereits analysierten 10 Tracks (Batch 1 + 2) ausschliessen
_SKIP_IDS = {
    # Batch 1
    "60773607",    # In My Mind Trap Transition (70 BPM)
    "261663350",   # Mobbin — Grandtheft (90 BPM)
    "172847123",   # Get Low — Reid Stefan (110 BPM)
    "197296240",   # I Want You To Know — Zedd (130 BPM)
    "12591733",    # The Dopest — Moksi (150 BPM)
    # Batch 2
    "57000017",    # GDFR — Flo Rida (73 BPM)
    "1765724",     # Closer — Chainsmokers (95 BPM)
    "57093172",    # On Trees and Birds and Fire — I Am Oak (117 BPM)
    "56422182",    # Time — Karla Blum (139 BPM)
    "261102615",   # Friesenjung — Ski Aggu (161 BPM)
    # Batch 3
    "49311865",    # Inside Out — Zedd (73 BPM)
    "159106036",   # Faded — ZHU (97 BPM)
    "174402019",   # Show Me Love — Sam Feldt (121 BPM)
    "164245259",   # Children Of A Miracle — Don Diablo (145 BPM)
    "65356661",    # About You Now — Niklas Dee (169 BPM)
    # Batch 4
    "167808883",   # Lookout Weekend — Reid Stefan (75 BPM)
    "90283934",    # Tuesday — Burak Yeter (99 BPM)
    "112931934",   # Givin' My Love — Tobi Kramer (123 BPM)
    "209805835",   # Phat Beat — Dr Phunk (147 BPM)
    "164435357",   # Blinding Lights — The Weeknd (171 BPM)
    # Batch 5
    "166380737",   # Chargé — Kaaris x Mr Carmack (75 BPM)
    "84475010",    # Don't Call Me Up — Mabel (99 BPM)
    "213136726",   # Parachute — Otto Knows (123 BPM)
    "152691798",   # Move It 2 The Drum — Maddix (147 BPM)
    "198983193",   # Remember — Netsky (173 BPM)
    # Batch 6
    "110536390",   # Lighthouse — Ookay & Fox Stevenson (75 BPM)
    "64994532",    # 0 to 100 — Drake (100 BPM)
    "46721060",    # Holding On — Disclosure (124 BPM)
    "187544640",   # Shine a Light — Hardwell & Wildstylez (148 BPM)
    "211531104",   # Stay (LUUDE Remix) (174 BPM)
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _fmt_time(ms: int | float | None) -> str:
    """Millisekunden → 'M:SS.ss'"""
    if ms is None:
        return "—"
    total_s = ms / 1000.0
    m = int(total_s // 60)
    s = total_s % 60
    return f"{m}:{s:05.2f}"


def _status_symbol(delta_beats: float | None, skipped: bool) -> str:
    """Delta in Beats → Status-Symbol"""
    if skipped:
        return "⬛ skip"
    if delta_beats is None:
        return "⬛ —"
    ab = abs(delta_beats)
    if ab <= 2:
        return "✅"
    elif ab <= 8:
        return "⚠️ nah"
    else:
        return "❌ falsch"


def _pick_diverse(candidates: list[tuple[float, object]], n: int = 5, bpm_gap: float = 20.0) -> list:
    """
    Greedy-Auswahl: Waehlt n Tracks mit moeglichst grossem BPM-Abstand.
    candidates: Liste von (bpm, content) sortiert nach BPM aufsteigend.
    """
    if len(candidates) <= n:
        return [c for _, c in candidates]

    selected = []
    last_bpm = -999.0
    for bpm, content in candidates:
        if bpm - last_bpm >= bpm_gap:
            selected.append(content)
            last_bpm = bpm
            if len(selected) >= n:
                break

    # Falls nicht genug gefunden: Rest auffuellen
    if len(selected) < n:
        used_ids = {c.ID for c in selected}
        for bpm, content in candidates:
            if content.ID not in used_ids:
                selected.append(content)
                if len(selected) >= n:
                    break

    return selected[:n]


def _compare_hot_cues(
    decisions: list,
    actual_cues: list,
    beat_ms: float,
) -> list[dict]:
    """
    Vergleicht vorhergesagte Hot-Cue-Entscheidungen mit manuellen DB-Cues.

    Returns: Liste von Vergleichs-Dicts pro Hot Cue (A, B, C).
    """
    results = []
    for dec in decisions:
        if dec.kind == 0:
            continue  # Memory: separat

        kind_label = f"Hot {chr(64 + dec.kind)}"

        # Vorhergesagter Cue
        pred_ms = dec.time_sec * 1000 if dec.time_sec is not None else None
        skipped = (dec.action == "skip")

        # Manueller Cue (gleicher Kind, kein AutoCue-Prefix)
        actual = next(
            (c for c in actual_cues
             if c.Kind == dec.kind
             and not _cfg.is_autocue_comment(c.Comment or "")),
            None,
        )
        actual_ms = actual.InMsec if actual else None

        # Delta
        if pred_ms is not None and actual_ms is not None:
            delta_ms = pred_ms - actual_ms
            delta_beats = delta_ms / beat_ms
        else:
            delta_ms = None
            delta_beats = None

        results.append({
            "label":        kind_label,
            "pred_ms":      pred_ms,
            "actual_ms":    actual_ms,
            "delta_ms":     delta_ms,
            "delta_beats":  round(delta_beats, 1) if delta_beats is not None else None,
            "sources":      dec.sources_str if dec.sources else "—",
            "skipped":      skipped,
            "action":       dec.action,
        })

    return results


def _compare_memory_cues(
    pred_cues: list,
    actual_cues: list,
    beat_ms: float,
) -> list[dict]:
    """
    Vergleicht vorhergesagte Memory Cues mit manuellen Memory Cues.
    Fuer jeden manuellen Memory Cue wird der naechstgelegene Predicted-Cue gesucht.
    """
    manual_mems = [c for c in actual_cues
                   if c.Kind == 0
                   and not _cfg.is_autocue_comment(c.Comment or "")]
    pred_mems   = [c for c in pred_cues if c.kind == 0]

    if not manual_mems:
        return []

    results = []
    for i, m in enumerate(manual_mems, 1):
        actual_ms = m.InMsec

        if pred_mems:
            closest = min(pred_mems, key=lambda p: abs(p.time_ms - actual_ms))
            pred_ms = closest.time_ms
            delta_ms = pred_ms - actual_ms
            delta_beats = delta_ms / beat_ms
        else:
            pred_ms = None
            delta_ms = None
            delta_beats = None

        results.append({
            "label":       f"Mem {i}",
            "pred_ms":     pred_ms,
            "actual_ms":   actual_ms,
            "delta_ms":    delta_ms,
            "delta_beats": round(delta_beats, 1) if delta_beats is not None else None,
            "sources":     "—",
            "skipped":     pred_ms is None,
            "action":      "set" if pred_ms is not None else "skip",
        })

    return results


def _render_table(rows: list[dict]) -> str:
    """Rendert eine Markdown-Tabelle aus Vergleichs-Rows."""
    header = "| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |"
    sep    = "|--------|----------|-------------|---------|---------|--------------|-----------------|"
    lines  = [header, sep]

    for r in rows:
        status     = _status_symbol(r["delta_beats"], r["skipped"])
        actual_str = _fmt_time(r["actual_ms"])
        pred_str   = _fmt_time(r["pred_ms"]) if not r["skipped"] else "skip"
        db_str     = f"{r['delta_beats']:+.1f}" if r["delta_beats"] is not None else "—"
        ds_str     = f"{r['delta_ms']/1000:+.2f}s" if r["delta_ms"] is not None else "—"
        src_str    = r["sources"]

        lines.append(
            f"| {r['label']:<6} | {actual_str:<8} | {pred_str:<11} | {db_str:<7} | {ds_str:<7} | {status:<12} | {src_str:<15} |"
        )

    return "\n".join(lines)


def _derive_optimizations(track_results: list[dict]) -> str:
    """
    Analysiert alle Track-Ergebnisse und leitet Optimierungsvorschlaege ab.
    """
    hot_a_deltas  = []
    hot_c_deltas  = []
    hot_a_sources = []
    hot_c_sources = []
    hot_a_skips   = 0
    hot_c_skips   = 0
    stage1_count  = 0
    total_tracks  = len(track_results)

    for tr in track_results:
        for row in tr["hot_rows"]:
            if row["label"] == "Hot A":
                if row["skipped"]:
                    hot_a_skips += 1
                elif row["delta_beats"] is not None:
                    hot_a_deltas.append(row["delta_beats"])
                hot_a_sources.append(row["sources"])
                if "MIK" in row["sources"] and "PHRASE" in row["sources"]:
                    stage1_count += 1
            elif row["label"] == "Hot C":
                if row["skipped"]:
                    hot_c_skips += 1
                elif row["delta_beats"] is not None:
                    hot_c_deltas.append(row["delta_beats"])
                hot_c_sources.append(row["sources"])

    lines = ["## Optimierungsempfehlungen\n"]

    # Hot A Analyse
    if hot_a_deltas:
        avg_a = sum(hot_a_deltas) / len(hot_a_deltas)
        avg_abs_a = sum(abs(d) for d in hot_a_deltas) / len(hot_a_deltas)
        lines.append(f"### Hot A")
        lines.append(f"- Ø Fehler: **{avg_a:+.1f} Beats** | Ø |Fehler|: **{avg_abs_a:.1f} Beats**")
        if avg_a < -4:
            lines.append("- ❌ **Systematisch zu früh** — MIK-Kandidaten oder CBR-Position könnte zu früh im Track liegen")
            lines.append("  → Prüfe: `hot_a_relative_pos` im CBR-Ergebnis vs. tatsächliche Breaks")
        elif avg_a > 4:
            lines.append("- ❌ **Systematisch zu spät** — Möglicher Offset-Fehler oder Phrase-Alignment greift zu spät")
        else:
            lines.append("- ✅ Kein systematischer Offset erkennbar")
        correct_a = sum(1 for d in hot_a_deltas if abs(d) <= 2)
        near_a    = sum(1 for d in hot_a_deltas if 2 < abs(d) <= 8)
        lines.append(f"- Treffer (±2 Beats): **{correct_a}/{len(hot_a_deltas)}** | Nah (±8 Beats): {near_a}/{len(hot_a_deltas)}")

    if hot_a_skips:
        lines.append(f"- ⬛ **{hot_a_skips}/{total_tracks} Tracks: Hot A skip** — kein Konsens erreicht")
        lines.append("  → Stage 1 (MIK×Phrase) und Stage 2 (CBR) liefern keinen Match. PWAV-Stage 2.5 prüfen.")
    lines.append("")

    # Hot C Analyse
    if hot_c_deltas:
        avg_c = sum(hot_c_deltas) / len(hot_c_deltas)
        avg_abs_c = sum(abs(d) for d in hot_c_deltas) / len(hot_c_deltas)
        lines.append(f"### Hot C")
        lines.append(f"- Ø Fehler: **{avg_c:+.1f} Beats** | Ø |Fehler|: **{avg_abs_c:.1f} Beats**")
        if avg_c < -4:
            lines.append("- ❌ **Systematisch zu früh** — CBR wählt den falschen Drop oder PWAV-Kandidat ist zu früh")
            lines.append("  → Prüfe: Zweiter Drop erkannt? Oder nur erster Drop?")
        elif avg_c > 4:
            lines.append("- ❌ **Systematisch zu spät** — Outro zu früh erkannt, verdrängt Hot C?")
        else:
            lines.append("- ✅ Kein systematischer Offset erkennbar")
        correct_c = sum(1 for d in hot_c_deltas if abs(d) <= 2)
        near_c    = sum(1 for d in hot_c_deltas if 2 < abs(d) <= 8)
        lines.append(f"- Treffer (±2 Beats): **{correct_c}/{len(hot_c_deltas)}** | Nah (±8 Beats): {near_c}/{len(hot_c_deltas)}")

    if hot_c_skips:
        lines.append(f"- ⬛ **{hot_c_skips}/{total_tracks} Tracks: Hot C skip** — kein Konsens erreicht")
    lines.append("")

    # Stage-Analyse
    lines.append("### Stage-Aktivierung (Hot A)")
    dominant_sources_a = {}
    for s in hot_a_sources:
        dominant_sources_a[s] = dominant_sources_a.get(s, 0) + 1
    for src, count in sorted(dominant_sources_a.items(), key=lambda x: -x[1]):
        lines.append(f"  - `{src}`: {count}x")

    if stage1_count < total_tracks // 2:
        lines.append(f"- ⚠️ **Stage 1 (MIK×PHRASE) greift nur bei {stage1_count}/{total_tracks} Tracks**")
        lines.append("  → MIK-Daten fehlen oder Phrasen-Detektion findet keine Breaks im richtigen Bereich")
    else:
        lines.append(f"- ✅ Stage 1 greift bei {stage1_count}/{total_tracks} Tracks")
    lines.append("")

    lines.append("### Hot C Quellen")
    dominant_sources_c = {}
    for s in hot_c_sources:
        dominant_sources_c[s] = dominant_sources_c.get(s, 0) + 1
    for src, count in sorted(dominant_sources_c.items(), key=lambda x: -x[1]):
        lines.append(f"  - `{src}`: {count}x")

    pwav_c_count = sum(1 for s in hot_c_sources if "PWAV" in s)
    if pwav_c_count > 0:
        lines.append(f"- ✅ PWAV trägt zu **{pwav_c_count}/{total_tracks}** Hot C Entscheidungen bei")
    else:
        lines.append("- ⬛ PWAV nicht in Hot C Quellen — Stage 1.5 oder Stage 2.5 greift nicht")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Haupt-Logik
# ---------------------------------------------------------------------------

def main(genre_filter: str | None = None, n_tracks: int = 5):
    print("=" * 60)
    print("  AutoCue v27 — Prediction vs. Manuell Vergleich")
    if genre_filter:
        print(f"  Genre-Filter: {genre_filter}")
    print("=" * 60)

    # 1. Datenbank oeffnen
    print("\n1. Rekordbox-DB oeffnen...")
    db = open_db()
    all_content = db.get_content().all()
    print(f"   {len(all_content)} Tracks in Bibliothek")

    # 2. Tracks mit manuellen Hot Cues filtern
    print("2. Tracks mit manuellen Hot Cues suchen...")
    candidates: list[tuple[float, object]] = []

    for content in all_content:
        if str(content.ID) in _SKIP_IDS:
            continue
        if not content.Cues or not content.Length:
            continue
        folder = content.FolderPath or ""
        if not folder or not Path(folder).exists():
            continue

        cues = list(content.Cues)
        hot_a = next(
            (c for c in cues
             if c.Kind == 1 and not _cfg.is_autocue_comment(c.Comment or "")),
            None,
        )
        hot_c = next(
            (c for c in cues
             if c.Kind == 3 and not _cfg.is_autocue_comment(c.Comment or "")),
            None,
        )

        if hot_a and hot_c:
            bpm = (content.BPM or 0) / 100.0
            if bpm > 0:
                # Genre-Filter
                if genre_filter:
                    g = (getattr(content, 'GenreName', '') or '').lower()
                    if genre_filter.lower() not in g:
                        continue

                # MIK-Filter: Nur Tracks mit MIK-Daten
                from app.mik_scraper import get_mik_data
                _artist = content.Artist.Name if content.Artist else ""
                _title  = content.Title or ""
                _mik    = get_mik_data(folder, _artist, _title)
                if _mik and _mik.cue_times:
                    candidates.append((bpm, content))

    print(f"   {len(candidates)} Tracks mit manuellen Hot A + Hot C gefunden (Batch 1+2 ausgeschlossen)")

    if not candidates:
        print("   ❌ Keine Tracks mit manuellen Hot Cues gefunden. Abbruch.")
        return

    # Sortiert nach BPM, diverse Auswahl
    candidates.sort(key=lambda x: x[0])
    selected = _pick_diverse(candidates, n=n_tracks, bpm_gap=20.0)
    print(f"   → {len(selected)} Tracks ausgewaehlt (BPM-Diversitaet)")

    for c in selected:
        bpm = (c.BPM or 0) / 100.0
        artist = c.Artist.Name if c.Artist else "Unbekannt"
        print(f"     • {c.Title} — {artist} ({bpm:.0f} BPM)")

    # 3. Dry-Run + Vergleich
    print("\n3. Analyse laeuft...\n")

    from app.writer import process_track

    track_results = []
    report_sections = []

    for idx, content in enumerate(selected, 1):
        bpm = (content.BPM or 0) / 100.0
        artist = content.Artist.Name if content.Artist else "Unbekannt"
        title = content.Title or "Unbekannt"
        duration_s = content.Length or 0

        dur_m = int(duration_s // 60)
        dur_s = duration_s % 60
        dur_str = f"{dur_m}:{dur_s:04.1f}"

        print(f"[{idx}/{len(selected)}] {title} — {artist} ({bpm:.0f} BPM)...")

        try:
            result = process_track(
                content.FolderPath,
                dry_run=True,
                verbose=False,
            )
        except Exception as exc:
            print(f"  ❌ Fehler: {exc}")
            report_sections.append(
                f"## {idx}. {title} — {artist}\n\n❌ Fehler bei Analyse: `{exc}`\n"
            )
            track_results.append({
                "title": title,
                "artist": artist,
                "bpm": bpm,
                "hot_rows": [],
                "mem_rows": [],
            })
            continue

        beat_ms = 60_000.0 / max(result.grid.bpm, 1.0)
        cues_in_db = list(content.Cues)

        hot_rows = _compare_hot_cues(result.decisions, cues_in_db, beat_ms)
        mem_rows = _compare_memory_cues(result.cues, cues_in_db, beat_ms)

        track_results.append({
            "title":    title,
            "artist":   artist,
            "bpm":      bpm,
            "hot_rows": hot_rows,
            "mem_rows": mem_rows,
        })

        # Welche Quellen wurden benutzt?
        hot_a_dec = next((d for d in result.decisions if d.kind == 1), None)
        hot_c_dec = next((d for d in result.decisions if d.kind == 3), None)
        hot_a_src = hot_a_dec.sources_str if hot_a_dec and hot_a_dec.sources else "skip"
        hot_c_src = hot_c_dec.sources_str if hot_c_dec and hot_c_dec.sources else "skip"

        section_lines = [
            f"## {idx}. {title} — {artist}",
            f"BPM: {bpm:.0f} | Dauer: {dur_str} | Hot A: `{hot_a_src}` | Hot C: `{hot_c_src}`",
            "",
            _render_table(hot_rows + mem_rows),
            "",
        ]
        report_sections.append("\n".join(section_lines))

        # Kurzfassung auf Terminal
        for r in hot_rows:
            status = _status_symbol(r["delta_beats"], r["skipped"])
            db_str = f"{r['delta_beats']:+.1f}b" if r["delta_beats"] is not None else "—"
            print(f"  {r['label']}: manuell={_fmt_time(r['actual_ms'])} pred={_fmt_time(r['pred_ms'])} Δ={db_str} {status} [{r['sources']}]")

    # 4. Optimierungsanalyse
    print("\n4. Optimierungsanalyse erstellen...")
    optimizations = _derive_optimizations(track_results)

    # 5. Gesamt-Statistik
    all_hot_a = [r for tr in track_results for r in tr["hot_rows"] if r["label"] == "Hot A"]
    all_hot_c = [r for tr in track_results for r in tr["hot_rows"] if r["label"] == "Hot C"]

    def _accuracy(rows, tol=2):
        valid = [r for r in rows if not r["skipped"] and r["delta_beats"] is not None]
        if not valid:
            return 0, len(rows)
        return sum(1 for r in valid if abs(r["delta_beats"]) <= tol), len(rows)

    a_ok, a_tot = _accuracy(all_hot_a)
    c_ok, c_tot = _accuracy(all_hot_c)
    avg_abs_a = (
        sum(abs(r["delta_beats"]) for r in all_hot_a if r["delta_beats"] is not None)
        / max(1, sum(1 for r in all_hot_a if r["delta_beats"] is not None))
    )
    avg_abs_c = (
        sum(abs(r["delta_beats"]) for r in all_hot_c if r["delta_beats"] is not None)
        / max(1, sum(1 for r in all_hot_c if r["delta_beats"] is not None))
    )

    summary = "\n".join([
        f"## Gesamt-Auswertung ({len(selected)} Tracks)\n",
        "| Metrik                     | Wert     |",
        "|---------------------------|----------|",
        f"| Tracks analysiert          | {len(selected)}        |",
        f"| Hot A ✅ Treffer (±2 Beats) | {a_ok}/{a_tot}      |",
        f"| Hot C ✅ Treffer (±2 Beats) | {c_ok}/{c_tot}      |",
        f"| Ø |Δ Hot A| (Beats)        | {avg_abs_a:.1f}    |",
        f"| Ø |Δ Hot C| (Beats)        | {avg_abs_c:.1f}    |",
        "",
    ])

    # 6. Report schreiben
    report_path = Path(__file__).parent / "compare_report.md"
    report = "\n---\n\n".join([
        f"# AutoCue v27 — Prediction vs. Manuell\nDatum: {date.today()} | Algorithmus: Quad-Check + PWAV-ML\n",
        *report_sections,
        summary,
        optimizations,
    ])

    report_path.write_text(report, encoding="utf-8")
    print(f"\n✅ Report geschrieben: {report_path}")
    print(f"\n{'='*60}")
    print(f"  Hot A Treffer (±2 Beats): {a_ok}/{a_tot}")
    print(f"  Hot C Treffer (±2 Beats): {c_ok}/{c_tot}")
    print(f"  Ø |Δ Hot A|: {avg_abs_a:.1f} Beats")
    print(f"  Ø |Δ Hot C|: {avg_abs_c:.1f} Beats")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", default=None, help="Genre-Filter (substring, case-insensitive)")
    parser.add_argument("-n", type=int, default=5, help="Anzahl Tracks")
    args = parser.parse_args()
    main(genre_filter=args.genre, n_tracks=args.n)
