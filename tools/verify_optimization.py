"""
AutoCue v27 — Optimierungsverifikation: Hot C Drop-Auswahl

Prueft ob der vorgeschlagene Fix (Stage 1 in validate_hot_c() sortiert
nach CBR-Naehe statt nach Zeit) bei 5 neuen Tracks wuerde greifen.

Methode:
  1. 5 neue Tracks (andere BPM-Bereiche als Vergleich 1)
  2. Dry-Run mit verbose=True → CBR-Position aus Output parsen
  3. Fuer jedes falsch vorhergesagte Hot C:
     - CBR-Anker berechnen
     - Pruefen: liegt CBR naeher am manuellen Ziel als die Vorhersage?
     - Wenn ja → Fix wuerde greifen

Keine Datenbankschreibvorgaenge. Rein analytisch.

Aufruf:
    .venv/bin/python tools/verify_optimization.py
"""
from __future__ import annotations

import io
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.beatgrid import open_db
from app import config as _cfg

# Content-IDs der bereits analysierten 5 Tracks (aus Vergleich 1) ausschliessen
_SKIP_IDS = {
    "60773607",   # In My Mind Trap Transition (70 BPM)
    "261663350",  # Mobbin — Grandtheft (90 BPM)
    "172847123",  # Get Low — Reid Stefan (110 BPM)
    "197296240",  # I Want You To Know — Zedd (130 BPM)
    "12591733",   # The Dopest — Moksi (150 BPM)
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen (aus compare_predictions.py uebernommen)
# ---------------------------------------------------------------------------

def _fmt_time(ms: int | float | None) -> str:
    if ms is None:
        return "—"
    total_s = ms / 1000.0
    m = int(total_s // 60)
    s = total_s % 60
    return f"{m}:{s:05.2f}"


def _status(delta_beats: float | None, skipped: bool) -> str:
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


def _pick_diverse(candidates: list[tuple[float, object]], n: int = 5,
                  bpm_gap: float = 20.0) -> list:
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
    if len(selected) < n:
        used = {c.ID for c in selected}
        for bpm, content in candidates:
            if content.ID not in used:
                selected.append(content)
                if len(selected) >= n:
                    break
    return selected[:n]


def _parse_cbr_hot_c_pct(verbose_output: str) -> float | None:
    """Extrahiert die CBR Hot C Position (%) aus dem verbose Output."""
    for line in verbose_output.splitlines():
        # z.B. "  Hot C Position: 65.4% (Median der Zwillinge)"
        if "Hot C Position:" in line:
            m = re.search(r"Hot C Position:\s*([\d.]+)\s*%", line)
            if m:
                return float(m.group(1)) / 100.0
    return None


def _parse_cbr_hot_a_pct(verbose_output: str) -> float | None:
    """Extrahiert die CBR Hot A Position (%) aus dem verbose Output."""
    for line in verbose_output.splitlines():
        if "Hot A Position:" in line:
            m = re.search(r"Hot A Position:\s*([\d.]+)\s*%", line)
            if m:
                return float(m.group(1)) / 100.0
    return None


def _fix_verdict(
    manual_ms: float,
    pred_ms: float | None,
    cbr_anchor_ms: float | None,
    beat_ms: float,
) -> str:
    """
    Beurteilt ob der vorgeschlagene Fix (CBR-Proximity-Sort) geholfen haette.

    Logik: Wenn der CBR-Anker naeher am manuellen Ziel liegt als die
    aktuelle Vorhersage, dann wuerde ein CBR-naher Kandidat gewaehlt
    werden → Fix hilft.

    Beachte: Dies ist eine Naeherung. Der echte Fix sortiert Phrase-
    Kandidaten nach Naehe zu CBR. Wir wissen nicht ob ein Phrase-
    Kandidat nah am manuellen Ziel existiert.
    """
    if pred_ms is None or cbr_anchor_ms is None:
        return "⬜ n/a"
    delta_pred   = abs(pred_ms   - manual_ms)
    delta_cbr    = abs(cbr_anchor_ms - manual_ms)
    delta_beats_pred = delta_pred / beat_ms
    delta_beats_cbr  = delta_cbr / beat_ms

    if delta_beats_pred <= 2:
        return "✅ schon korrekt"

    if delta_beats_cbr <= delta_beats_pred * 0.5:
        # CBR liegt deutlich naeher → Fix wuerde wahrscheinlich helfen
        return f"✅ Fix hilft (CBR bei {_fmt_time(cbr_anchor_ms)}, Δ={delta_beats_cbr:.0f}b)"
    elif delta_beats_cbr < delta_beats_pred:
        # CBR etwas naeher, aber nicht eindeutig
        return f"⚠️ Fix hilft teilweise (CBR Δ={delta_beats_cbr:.0f}b vs. Pred Δ={delta_beats_pred:.0f}b)"
    else:
        # CBR liegt gleich weit oder weiter → anderer Fehler
        return f"❌ Fix hilft nicht (CBR auch weit weg: Δ={delta_beats_cbr:.0f}b)"


# ---------------------------------------------------------------------------
# Haupt-Logik
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  AutoCue v27 — Optimierungsverifikation (5 neue Tracks)")
    print("=" * 65)

    # 1. DB oeffnen
    print("\n1. Rekordbox-DB oeffnen...")
    db = open_db()
    all_content = db.get_content().all()
    print(f"   {len(all_content)} Tracks in Bibliothek")

    # 2. Tracks filtern (manuell gesetzte Hot Cues, NICHT die ersten 5)
    print("2. Neue Tracks mit manuellen Hot Cues suchen...")
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
            (c for c in cues if c.Kind == 1
             and not _cfg.is_autocue_comment(c.Comment or "")), None
        )
        hot_c = next(
            (c for c in cues if c.Kind == 3
             and not _cfg.is_autocue_comment(c.Comment or "")), None
        )
        if hot_a and hot_c:
            bpm = (content.BPM or 0) / 100.0
            if bpm > 0:
                candidates.append((bpm, content))

    print(f"   {len(candidates)} Kandidaten gefunden (erste 5 ausgeschlossen)")
    candidates.sort(key=lambda x: x[0])
    selected = _pick_diverse(candidates, n=5, bpm_gap=22.0)

    print(f"   → {len(selected)} Tracks ausgewaehlt:")
    for c in selected:
        bpm = (c.BPM or 0) / 100.0
        artist = c.Artist.Name if c.Artist else "?"
        print(f"     • {c.Title} — {artist} ({bpm:.0f} BPM)")

    # 3. Analyse
    print("\n3. Analyse laeuft (verbose=True fuer CBR-Extraktion)...\n")

    from app.writer import process_track

    report_sections  = []
    summary_rows     = []

    for idx, content in enumerate(selected, 1):
        bpm_val = (content.BPM or 0) / 100.0
        artist  = content.Artist.Name if content.Artist else "?"
        title   = content.Title or "?"
        dur_s   = content.Length or 0
        dur_str = f"{int(dur_s//60)}:{dur_s%60:04.1f}"

        print(f"[{idx}/5] {title} — {artist} ({bpm_val:.0f} BPM)...")

        # Verbose-Output abfangen
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            result = process_track(content.FolderPath, dry_run=True, verbose=True)
        except Exception as exc:
            sys.stdout = old_stdout
            print(f"  ❌ Fehler: {exc}")
            report_sections.append(f"## {idx}. {title} — {artist}\n\n❌ Fehler: `{exc}`\n")
            continue
        finally:
            sys.stdout = old_stdout

        verbose_out = buf.getvalue()

        # CBR-Positionen parsen
        cbr_c_pct = _parse_cbr_hot_c_pct(verbose_out)
        cbr_a_pct = _parse_cbr_hot_a_pct(verbose_out)

        beat_ms   = 60_000.0 / max(result.grid.bpm, 1.0)
        dur_ms    = dur_s * 1000.0
        cues_db   = list(content.Cues)

        cbr_c_ms = cbr_c_pct * dur_ms if cbr_c_pct is not None else None
        cbr_a_ms = cbr_a_pct * dur_ms if cbr_a_pct is not None else None

        # Manuell gesetzte Hot Cues
        manual_a = next((c for c in cues_db if c.Kind == 1
                         and not _cfg.is_autocue_comment(c.Comment or "")), None)
        manual_b = next((c for c in cues_db if c.Kind == 2
                         and not _cfg.is_autocue_comment(c.Comment or "")), None)
        manual_c = next((c for c in cues_db if c.Kind == 3
                         and not _cfg.is_autocue_comment(c.Comment or "")), None)

        man_a_ms = manual_a.InMsec if manual_a else None
        man_b_ms = manual_b.InMsec if manual_b else None
        man_c_ms = manual_c.InMsec if manual_c else None

        # Vorhersagen
        dec_a = next((d for d in result.decisions if d.kind == 1), None)
        dec_b = next((d for d in result.decisions if d.kind == 2), None)
        dec_c = next((d for d in result.decisions if d.kind == 3), None)

        pred_a_ms = dec_a.time_sec * 1000 if (dec_a and dec_a.time_sec) else None
        pred_b_ms = dec_b.time_sec * 1000 if (dec_b and dec_b.time_sec) else None
        pred_c_ms = dec_c.time_sec * 1000 if (dec_c and dec_c.time_sec) else None

        # Deltas
        def _delta_b(pred, actual):
            if pred is None or actual is None:
                return None
            return round((pred - actual) / beat_ms, 1)

        da_b = _delta_b(pred_a_ms, man_a_ms)
        db_b = _delta_b(pred_b_ms, man_b_ms)
        dc_b = _delta_b(pred_c_ms, man_c_ms)

        skip_a = (dec_a.action == "skip") if dec_a else True
        skip_b = (dec_b.action == "skip") if dec_b else True
        skip_c = (dec_c.action == "skip") if dec_c else True

        # Fix-Verdikte
        fix_c = _fix_verdict(man_c_ms, pred_c_ms, cbr_c_ms, beat_ms) if man_c_ms else "⬜ n/a"
        fix_a = _fix_verdict(man_a_ms, pred_a_ms, cbr_a_ms, beat_ms) if man_a_ms else "⬜ n/a"

        # Quellen
        src_a = dec_a.sources_str if (dec_a and dec_a.sources) else ("skip" if skip_a else "—")
        src_c = dec_c.sources_str if (dec_c and dec_c.sources) else ("skip" if skip_c else "—")

        # Terminal-Ausgabe
        cbr_c_str = _fmt_time(cbr_c_ms) if cbr_c_ms else "?"
        cbr_a_str = _fmt_time(cbr_a_ms) if cbr_a_ms else "?"
        print(f"  CBR Hot A Anker: {cbr_a_str}  |  CBR Hot C Anker: {cbr_c_str}")
        for label, man, pred, db, src, fix in [
            ("Hot A", man_a_ms, pred_a_ms, da_b, src_a, fix_a),
            ("Hot C", man_c_ms, pred_c_ms, dc_b, src_c, fix_c),
        ]:
            stat = _status(db, pred is None)
            db_str = f"{db:+.1f}b" if db is not None else "—"
            print(f"  {label}: man={_fmt_time(man)} pred={_fmt_time(pred)} Δ={db_str} {stat} [{src}]  → {fix}")

        # Report-Sektion
        cbr_c_pct_str = f"{cbr_c_pct*100:.1f}%" if cbr_c_pct else "?"
        cbr_a_pct_str = f"{cbr_a_pct*100:.1f}%" if cbr_a_pct else "?"

        table_rows = []
        for label, man, pred, db, skp, src in [
            ("Hot A", man_a_ms, pred_a_ms, da_b, skip_a, src_a),
            ("Hot B", man_b_ms, pred_b_ms, db_b, skip_b, "DERIVED"),
            ("Hot C", man_c_ms, pred_c_ms, dc_b, skip_c, src_c),
        ]:
            stat   = _status(db, skp)
            db_str = f"{db:+.1f}" if db is not None else "—"
            ds_str = f"{db * beat_ms / 1000:+.2f}s" if db is not None else "—"
            table_rows.append(
                f"| {label:<6} | {_fmt_time(man):<8} | {_fmt_time(pred) if not skp else 'skip':<11} | {db_str:<7} | {ds_str:<7} | {stat:<12} | {src:<15} |"
            )

        header = "| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |"
        sep    = "|--------|----------|-------------|---------|---------|--------------|-----------------|"

        fix_block = "\n".join([
            f"**CBR-Anker:** Hot A = {cbr_a_str} ({cbr_a_pct_str} der Trackdauer) | Hot C = {cbr_c_str} ({cbr_c_pct_str} der Trackdauer)",
            "",
            f"**Verifikation CBR-Proximity-Fix:**",
            f"- Hot A: {fix_a}",
            f"- Hot C: {fix_c}",
        ])

        section = "\n".join([
            f"## {idx}. {title} — {artist}",
            f"BPM: {bpm_val:.0f} | Dauer: {dur_str} | Hot A: `{src_a}` | Hot C: `{src_c}`",
            "",
            header, sep,
            *table_rows,
            "",
            fix_block,
        ])
        report_sections.append(section)

        summary_rows.append({
            "title":   title,
            "bpm":     bpm_val,
            "da":      da_b,
            "dc":      dc_b,
            "skip_a":  skip_a,
            "skip_c":  skip_c,
            "fix_c":   fix_c,
            "fix_a":   fix_a,
            "src_a":   src_a,
            "src_c":   src_c,
        })

    # 4. Gesamt-Auswertung
    print("\n4. Auswertung...")

    hot_c_rows_valid = [r for r in summary_rows if r["dc"] is not None and not r["skip_c"]]
    hot_a_rows_valid = [r for r in summary_rows if r["da"] is not None and not r["skip_a"]]

    correct_c = sum(1 for r in hot_c_rows_valid if abs(r["dc"]) <= 2)
    near_c    = sum(1 for r in hot_c_rows_valid if 2 < abs(r["dc"]) <= 8)
    wrong_c   = sum(1 for r in hot_c_rows_valid if abs(r["dc"]) > 8)
    correct_a = sum(1 for r in hot_a_rows_valid if abs(r["da"]) <= 2)

    avg_abs_c = (
        sum(abs(r["dc"]) for r in hot_c_rows_valid) / len(hot_c_rows_valid)
    ) if hot_c_rows_valid else 0
    avg_abs_a = (
        sum(abs(r["da"]) for r in hot_a_rows_valid) / len(hot_a_rows_valid)
    ) if hot_a_rows_valid else 0

    fix_helps    = sum(1 for r in summary_rows if "Fix hilft" in r["fix_c"] and "nicht" not in r["fix_c"])
    fix_partial  = sum(1 for r in summary_rows if "teilweise" in r["fix_c"])
    fix_nothings = sum(1 for r in summary_rows if "hilft nicht" in r["fix_c"])
    fix_ok       = sum(1 for r in summary_rows if "schon korrekt" in r["fix_c"])

    stage1_count = sum(1 for r in summary_rows if "MIK" in r["src_a"] and "PHRASE" in r["src_a"])
    pwav_c_count = sum(1 for r in summary_rows if "PWAV" in r["src_c"])

    summary_md = "\n".join([
        "## Gesamt-Auswertung (5 neue Tracks)\n",
        "| Metrik                          | Wert     |",
        "|---------------------------------|----------|",
        f"| Tracks analysiert               | {len(summary_rows)}        |",
        f"| Hot A ✅ Treffer (±2 Beats)      | {correct_a}/{len(hot_a_rows_valid)}      |",
        f"| Hot A skip                      | {sum(1 for r in summary_rows if r['skip_a'])}/5      |",
        f"| Hot C ✅ Treffer (±2 Beats)      | {correct_c}/{len(hot_c_rows_valid)}      |",
        f"| Hot C ⚠️ nah (2-8 Beats)         | {near_c}/{len(hot_c_rows_valid)}      |",
        f"| Hot C ❌ falsch (>8 Beats)       | {wrong_c}/{len(hot_c_rows_valid)}      |",
        f"| Ø \\|Δ Hot A\\| (Beats)            | {avg_abs_a:.1f}    |",
        f"| Ø \\|Δ Hot C\\| (Beats)            | {avg_abs_c:.1f}    |",
        f"| Stage 1 (MIK×Phrase) aktiv      | {stage1_count}/5      |",
        f"| PWAV aktiv (Hot C)              | {pwav_c_count}/5      |",
        "",
        "## Fix-Verifikation: CBR-Proximity-Sort fuer Hot C\n",
        "| Verdikt                         | Anzahl   |",
        "|---------------------------------|----------|",
        f"| ✅ Fix hilft (CBR naeher am Ziel) | {fix_helps}/5      |",
        f"| ⚠️ Fix hilft teilweise           | {fix_partial}/5      |",
        f"| ❌ Fix hilft nicht (CBR auch weit) | {fix_nothings}/5     |",
        f"| ✅ Schon korrekt (kein Fix noetig) | {fix_ok}/5      |",
    ])

    if fix_helps >= 3:
        conclusion = (
            f"\n**Fazit: CBR-Proximity-Fix ist BESTAETIGT ({fix_helps}/5 Tracks).** "
            "Die Optimierung wuerde in der Mehrzahl der Faelle den richtigen Drop auswaehlen."
        )
    elif fix_helps + fix_partial >= 3:
        conclusion = (
            f"\n**Fazit: CBR-Proximity-Fix BEDINGT BESTAETIGT ({fix_helps} klar + {fix_partial} teilweise).** "
            "Die Optimierung hilft in den meisten Faellen, loest aber nicht alle Fehler."
        )
    else:
        conclusion = (
            f"\n**Fazit: CBR-Proximity-Fix NICHT BESTAETIGT ({fix_helps}/5).** "
            "Die Fehler haben andere Ursachen — naechste Optimierung pruefen."
        )
    summary_md += conclusion

    # 5. Report schreiben
    report_path = Path(__file__).parent / "verify_report.md"
    report = "\n\n---\n\n".join([
        f"# AutoCue v27 — Optimierungsverifikation: CBR-Proximity-Sort fuer Hot C\n"
        f"Datum: {date.today()} | 5 neue Tracks (andere BPM-Bereiche als Vergleich 1)\n",
        *report_sections,
        summary_md,
    ])
    report_path.write_text(report, encoding="utf-8")

    print(f"\n✅ Report: {report_path}")
    print(f"\n{'='*65}")
    print(f"  Hot A Treffer: {correct_a}/{len(hot_a_rows_valid)} | Ø |Δ|: {avg_abs_a:.1f} Beats")
    print(f"  Hot C Treffer: {correct_c}/{len(hot_c_rows_valid)} | Ø |Δ|: {avg_abs_c:.1f} Beats")
    print(f"  Fix-Verdikte: {fix_helps}x hilft, {fix_partial}x teilweise, {fix_nothings}x hilft nicht, {fix_ok}x schon ok")
    print(f"{'='*65}")
    print(conclusion)


if __name__ == "__main__":
    main()
