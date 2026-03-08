# Rekordbox AutoCue v26.1

> ML-basierte Cue-Point-Vorhersage für Rekordbox — lernt aus deinen ~5.000 Tracks.

AutoCue analysiert deine Tracks akustisch, nutzt Mixed In Key-Daten und lernt aus deinen manuellen Korrekturen — vollautomatisch, ohne Rekordbox-Cloud.

---

## Features

| Feature | Beschreibung |
|---------|-------------|
| **Beatgrid-Sync** | Cues snappen exakt auf Downbeats via ANLZ-Beatgrid |
| **Triple-Check (v26.1)** | Jeder Hot Cue unabhängig validiert: MIK × Phrase × Library |
| **PSSI-Phrasen (v26.1)** | Rekordbox-Phrase-Analyse aus ANLZ .EXT-Dateien (Intro/Drop/Break/Outro) |
| **Recursive Intelligence (v26.1)** | Gelernte Offsets auto-angewendet wenn Konfidenz >80% |
| **Akustische Segmentierung** | SSM + Novelty + Energy + Percussive Ratio → Intro / Drop / Break / Outro |
| **MIK-Integration** | Mixed In Key Cue-Positionen als primäre Hotspots (ID3-Tags + SQLite) |
| **CBR (5k-Brain)** | Case-Based Reasoning — findet Zwillings-Tracks aus deiner Sammlung |
| **Feedback-Loop** | Lernschleife: Rekordbox-Korrekturen → automatische Parameteranpassung |
| **Konfigurierbar** | Alle Parameter in `config.yaml`, kein Hardcoding |

---

## Cue-Hierarchie

```
Hot Cue A  →  "The Break"     — Break-Einstieg (Percussion fällt nach Intro)
Hot Cue B  →  "The Setup"     — Exakt N Beats vor Hot C (Default: 32 Beats)
Hot Cue C  →  "The Last Drop" — Letzter Drop (Segment + Visual Edge)

Memory Cues (max. 10):
  Prio 1 — Erster Downbeat
  Prio 2 — Intro-Struktur (alle 32 Beats)
  Prio 3 — First Drop, Second Break (Ankerpunkte)
  Prio 4 — Weitere Phrasen-Übergänge
  Prio 5 — Outro-Struktur (nie am letzten Schlag)
```

Constraints:
- Hot A ↔ Hot B: mind. 128 Beats (32 Bars) Abstand
- Memory Cues: mind. 32 Beats von jedem Hot Cue entfernt
- Memory Cues: mind. 32 Beats untereinander

---

## Voraussetzungen

- macOS 13+ (Apple Silicon oder Intel)
- Python 3.12+
- Rekordbox 6.x oder 7.x installiert und konfiguriert
- Mixed In Key 11 *(optional — für MIK Cue-Integration)*

---

## Installation

```bash
# 1. Repository klonen
git clone <repo-url> rekordbox-autocue-tool
cd rekordbox-autocue-tool

# 2. Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Optional: MIK ID3-Tag-Support
pip install mutagen
```

---

## Konfiguration

Alle Parameter werden in `config.yaml` gesteuert:

```yaml
# Cue-Prefix — erscheint in Rekordbox als Cue-Label
cue_prefix: "A:"               # z.B. "A: The Break", "A: The Last Drop"

# Hot Cue Constraints
min_hot_a_b_distance_beats: 128
hot_b_offset_beats: 32

# Memory Cues
max_memory_cues: 10
memory_min_hot_distance_beats: 32

# Playlist die analysiert werden soll
analyse_playlist: "--analyse-playlist"
```

Wichtig: Keine Änderungen am Code nötig — nur `config.yaml` anpassen.

---

## Workflow

### 1. Tracks vorbereiten

Erstelle in Rekordbox eine Playlist mit dem Namen `--analyse-tracks` (oder was in `config.yaml` konfiguriert ist) und füge die zu analysierenden Tracks hinzu.

### 2. Bibliothek vektorisieren (einmalig / nach großen Änderungen)

Berechnet Feature-Vektoren für alle Rekordbox-Tracks (für CBR / 5k-Brain).
Nur nötig wenn erstmalig eingerichtet oder viele neue Tracks hinzugekommen sind.

```bash
source .venv/bin/activate

# Schnell: nur Metadaten aus Rekordbox-DB (~30 Sekunden)
python -m app.vectorize

# Vollständig: mit echter Audio-Energie (~2-3 Stunden für 5.000 Tracks)
python -m app.vectorize --mode full
```

Ergebnis: `app/data/track_vectors.npz` + `app/data/track_meta.pkl`

### 3. Analyse starten

```bash
source .venv/bin/activate

# Dry-Run (empfohlen zuerst) — nichts wird geschrieben
python -m app.batch

# Cues wirklich in Rekordbox schreiben
python -m app.batch --live

# Mit automatischer Bereinigung alter Cues
python -m app.batch --live --clean

# Alle Tracks neu verarbeiten (auch bereits analysierte)
python -m app.batch --live --force

# Andere Playlist verwenden
python -m app.batch --live --playlist "Meine Playlist"
```

### 3. Terminal-Ausgabe

```
╭──────────────────────────────────────────────────╮
│  AutoCue v26.1                            LIVE   │
│  Playlist: --analyse-tracks                      │
╰──────────────────────────────────────────────────╯
✓ Sicherheitscheck: Rekordbox nicht aktiv
✓ Playlist: --analyse-tracks (ID=42)

  ⠸ Daft Punk – Get Lucky          ████████████████  5/10  0:01:23
  ✓ Daft Punk – Get Lucky          A:2:14.08  B:5:58.00  C:6:30.04  Mem:7
    ✓ Hot A   2:14.08  [MIK+PHRASE]    The Break
    ✓ Hot B   5:58.00  [DERIVED]       Setup (32b vor Drop)
    ✓ Hot C   6:30.04  [MIK+PHRASE]    The Last Drop
  ✓ Mauro Picotto – Komodo         A:1:28.00  B:4:32.08  C:5:04.00  Mem:6
    ✗ Hot A     —      [—]             No Consensus: MIK, Phrase und Library...
    ✓ Hot B   4:32.08  [DERIVED]       Setup (32b vor Drop)
    ✓ Hot C   5:04.00  [MIK]           The Last Drop
```

Triple-Check-Logik pro Cue:
- `[MIK+PHRASE]` — MIK und PSSI-Phrase übereinstimmend (stärkster Konsens)
- `[MIK+LIBRARY]` oder `[PHRASE+LIBRARY]` — 2 von 3 Quellen bestätigen
- `[MIK]` / `[PHRASE]` / `[LIBRARY]` — Lenient-Fallback (nur Hot C)
- `[DERIVED]` — Hot B: immer aus Hot C abgeleitet, kein Triple-Check
- `✗ [—]` — Kein Konsens, Cue wird übersprungen (Partial Success)

### 4. Aus Korrekturen lernen

Nach der Analyse: Öffne Rekordbox und passe Cues manuell an.
Dann den Lern-Befehl ausführen damit AutoCue deine Korrekturen versteht:

```bash
# Lernlauf: Korrekturen erkennen + Patterns aktualisieren
python -m app.learn

# Mit detaillierter Analyse pro Track
python -m app.learn --feedback

# Nur Statistiken anzeigen (kein neuer Lernlauf)
python -m app.learn --summary-only
```

Was passiert:
1. **Korrektur-Erkennung** — Verschobene/gelöschte Cues werden erkannt
2. **Pattern-Update** — Gelernte Offsets in `agent_learning.db` gespeichert
3. **Recursive Intelligence** — Nächste Analyse nutzt Patterns automatisch

*Hinweis: `--live` ruft Review & Learn auch automatisch vor jeder Analyse auf.*

### 5. Intelligente Playlisten nach Monat/Jahr generieren

Erstellt 12 Smart-Playlisten (Monats-Filter) für ein Zieljahr. **Rekordbox muss dazu geschlossen sein.**

```bash
python -m app.playlists
```

Zweistufiges Menü:

```
  AutoCue — Rekordbox Playlist Generator
  ──────────────────────────────────────────────────────
  Schritt 1 — Modus wählen

  1   Direkt in Rekordbox-Datenbank schreiben  (Smart Playlists)
  2   XML-Datei exportieren                    (manueller Import)
  0   Beenden

  Modus: 1

  ──────────────────────────────────────────────────────
  Schritt 2 — Funktion wählen  [Modus: DB]

  1   Jahres-Playlisten erstellen  (01-YYYY … 12-YYYY)
  0   Zurück

  Auswahl (oder Jahreszahl): 2025

  ── Jahres-Playlisten 2025 ──────────────────────────
  ✓  01-2025  erstellt
  ✓  02-2025  erstellt
  ✗  03-2025  existiert bereits — übersprungen
  …
  ✓  12-2025  erstellt
  ✓  11 von 12 Playlisten erstellt
```

**Modus 1 — DB direkt schreiben:**
- Legt Smart Playlists direkt in `master.db` an — kein Import nötig
- Ordnerstruktur: `Year → 2025 → 01-2025 … 12-2025`
- Bestehende Playlisten werden **nicht** überschrieben (Schutz vor Datenverlust)
- Filter-Basis: `StockDate` (= "Hinzugefügt am"), Schaltjahr-korrekt per `calendar.monthrange`

**Modus 2 — XML exportieren:**
- Exportiert nach `export/rekordbox_YYYY.xml`
- Import: **Rekordbox → Datei → Bibliothek → XML-Bibliothek importieren**
- Überschreiben bestehender Dateien wird vor Export abgefragt

**Tipp:** Im Funktions-Menü kann die Jahreszahl direkt eingegeben werden (z. B. `2027`) statt erst `1` zu wählen.

---

## Projektstruktur

```
rekordbox-autocue-tool/
├── config.yaml              ← Alle Parameter hier konfigurieren
├── requirements.txt
├── README.md
├── app.log                  ← Automatisch erstellt (Laufzeit-Log)
│
├── app/                      ← Aktive Pipeline (v26.1)
│   ├── config.py            ← Config-Loader (config.yaml → Python dict)
│   ├── batch.py             ← CLI-Einstiegspunkt, rich Terminal-UI
│   ├── writer.py            ← Rekordbox DB-Writer + ProcessResult
│   ├── cue_logic.py         ← Triple-Check Cue-Hierarchie (v26.1)
│   ├── validator.py         ← Triple-Check Validator (MIK × Phrase × Library)
│   ├── phrase_reader.py     ← PSSI-Phrasen aus ANLZ .EXT-Dateien
│   ├── segments.py          ← Akustische Segmentierung (SSM + Novelty)
│   ├── beatgrid.py          ← ANLZ-Beatgrid-Reader, Snap-to-Grid
│   ├── mik_scraper.py       ← Mixed In Key Daten (ID3 + SQLite)
│   ├── cbr.py               ← Case-Based Reasoning (5k-Brain)
│   ├── learner.py           ← Feedback-Loop + Parameter-Update
│   ├── learning_db.py       ← SQLite Lern-DB + Recursive Intelligence
│   ├── feedback.py          ← Feedback-Datenmodell
│   ├── vectorize.py         ← CLI: Bibliothek vektorisieren (`python -m app.vectorize`)
│   ├── learn.py             ← CLI: Korrekturen lernen (`python -m app.learn`)
│   ├── playlists.py         ← CLI: Monats-Playlisten generieren (`python -m app.playlists`)
│   └── data/                ← Laufzeitdaten (predictions, learned_params)
│
└── app.log                  ← Wird automatisch erstellt (rotierendes Log)
```

---

## Versionierung

AutoCue verwendet Jahr-basierte Versionsnummern:

| Version | Beschreibung |
|---------|-------------|
| `v26`   | 2026 — Produktionsreife Basis-Pipeline |
| `v26.1` | Triple-Check + PSSI-Phrasen + Recursive Intelligence |

---

## Fehlerbehebung

**`Playlist nicht gefunden`**
→ In Rekordbox eine Playlist mit genau dem Namen aus `config.yaml → analyse_playlist` erstellen.

**`FEHLER: Rekordbox läuft noch!`**
→ Rekordbox vollständig beenden, dann erneut starten.

**`FileNotFoundError: Track nicht in DB`**
→ Track zuerst in Rekordbox importieren (Analyse-Daten müssen vorhanden sein).

**`[src/libmpg123/id3.c:process_extra()...] error: No extra frame text`**
→ Harmlose C-Library-Meldung von libmpg123 bei bestimmten MP3-ID3-Tags. Wird automatisch unterdrückt (kein Handlungsbedarf).

**MIK-Cues werden nicht erkannt**
→ Prüfen ob Mixed In Key 11 installiert ist und Tracks analysiert wurden.
→ Pfad in `config.yaml → mik_db_path` ggf. manuell setzen.

**Logs**
→ Details in `app.log` (rotierend, max. 5 MB × 3 Dateien).
