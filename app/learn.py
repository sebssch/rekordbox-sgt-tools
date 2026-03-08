"""
AutoCue Lern-Kommando — korrigierte Tracks analysieren und Patterns lernen.

Scannt alle Tracks fuer die AutoCue Vorhersagen gespeichert hat, vergleicht
die gesetzten Cues mit dem aktuellen Rekordbox-Stand und lernt aus Korrekturen
(verschobene/geloeschte Cues).

Nutzung:
    python -m app.learn                 # Review & Learn + DB-Zusammenfassung
    python -m app.learn --feedback      # Zusaetzlich: Detailanalyse pro Track
    python -m app.learn --summary-only  # Nur DB-Statistiken ohne neuen Lernlauf
"""

import argparse
import logging
import sys


# --- CLI ---

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.learn",
        description=(
            "AutoCue Lern-Kommando\n"
            "Erkennt Korrekturen in Rekordbox und verbessert zukuenftige Vorhersagen.\n\n"
            "Ablauf:\n"
            "  1. Alle AutoCue-Tracks werden mit aktuellem Rekordbox-Stand verglichen.\n"
            "  2. Verschobene/geloeschte Cues werden als Korrekturen erkannt.\n"
            "  3. Gelernte Offsets und Patterns werden in agent_learning.db gespeichert.\n"
            "  4. Zukuenftige Laeufe nutzen diese Patterns automatisch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m app.learn                 # Lernlauf + Zusammenfassung
  python -m app.learn --feedback      # Zusaetzlich Detailanalyse pro Track
  python -m app.learn --summary-only  # Nur Statistiken, kein neuer Lernlauf
        """,
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        default=False,
        help="Detaillierte Feedback-Analyse pro Track anzeigen",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        help="Nur DB-Statistiken anzeigen (kein neuer Review-Lauf)",
    )
    return parser


def run(feedback: bool = False, summary_only: bool = False) -> dict:
    """
    Hauptfunktion: Review & Learn + optionales Feedback.

    Args:
        feedback:     Detaillierte Feedback-Ausgabe pro Track.
        summary_only: Kein neuer Lernlauf, nur Statistiken.

    Returns:
        dict mit Ergebnissen (n_tracks, n_corrections, accuracy)
    """
    # Pyrekordbox-Warnungen unterdrücken
    logging.getLogger("pyrekordbox").setLevel(logging.ERROR)

    from app.learning_db import (
        get_db,
        review_and_learn,
        print_db_summary,
    )
    from app.feedback import collect_feedback, print_feedback_summary
    from app.beatgrid import open_db

    print()
    print("=" * 60)
    print("  AutoCue — Lernlauf")
    print("=" * 60)

    result = {"n_tracks": 0, "n_corrections": 0, "accuracy": 0.0}

    # --- Rekordbox-DB oeffnen ---
    try:
        rb_db = open_db()
    except Exception as e:
        print(f"\n  FEHLER: Rekordbox-DB konnte nicht geoeffnet werden:\n  {e}")
        sys.exit(1)

    # --- Review & Learn ---
    if not summary_only:
        print("\n  Schritt 1: Korrekturen erkennen und lernen ...")
        try:
            result = review_and_learn(rekordbox_db=rb_db)
        except Exception as e:
            print(f"\n  FEHLER beim Review & Learn:\n  {e}")
            logging.exception("review_and_learn fehlgeschlagen")
    else:
        print("\n  (Lernlauf uebersprungen — nur Statistiken)")

    # --- DB-Zusammenfassung ---
    print()
    try:
        print_db_summary()
    except Exception as e:
        print(f"  WARNUNG: Konnte DB-Zusammenfassung nicht laden: {e}")

    # --- Optionale Detailanalyse ---
    if feedback:
        print()
        print("=" * 60)
        print("  Detailanalyse pro Track")
        print("=" * 60)
        try:
            fb = collect_feedback(db=rb_db)
            print_feedback_summary(fb)
        except Exception as e:
            print(f"  FEHLER bei Feedback-Analyse: {e}")

    return result


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        feedback=args.feedback,
        summary_only=args.summary_only,
    )
