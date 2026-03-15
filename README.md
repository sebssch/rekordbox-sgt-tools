# Rekordbox AutoCue v27

> ML-basierte Cue-Point-Vorhersage fuer Rekordbox — lernt aus deinen ~5.000 Tracks.

AutoCue analysiert deine Tracks akustisch, nutzt Mixed In Key-Daten und ein LightGBM-Modell trainiert auf deiner Bibliothek — vollautomatisch, ohne Rekordbox-Cloud.

---

## Features

| Feature | Beschreibung |
|---------|-------------|
| **Beatgrid-Sync** | Cues snappen exakt auf Downbeats via ANLZ-Beatgrid |
| **Quad-Check (v27)** | Jeder Hot Cue unabhaengig validiert: MIK x Phrase x Library x ML |
| **LightGBM ML (v27)** | Supervised Learning auf 4.430 Tracks — Hot C 100% Accuracy (5/5 Test) |
| **PSSI-Phrasen** | Rekordbox-Phrase-Analyse aus ANLZ .EXT-Dateien (Intro/Drop/Break/Outro) |
| **Akustische Segmentierung** | SSM + Novelty + Energy + Percussive Ratio |
| **MIK-Integration** | Mixed In Key Cue-Positionen als primaere Hotspots (ID3-Tags + SQLite) |
| **CBR (5k-Brain)** | Case-Based Reasoning — findet Zwillings-Tracks aus deiner Sammlung |
| **Konfigurierbar** | Alle Parameter in `config.yaml`, kein Hardcoding |

---

## Cue-Hierarchie

```
Hot Cue A  →  "The Break"     — Break-Einstieg (Percussion faellt nach Intro)
Hot Cue B  →  "The Setup"     — Exakt N Beats vor Hot C (Default: 32 Beats)
Hot Cue C  →  "The Last Drop" — Letzter Drop (Segment + Visual Edge)

Memory Cues (max. 10):
  Prio 1 — Erster Downbeat
  Prio 2 — Intro-Struktur (alle 32 Beats)
  Prio 3 — MIK-Anker + First Drop, Second Break
  Prio 4 — ML-Vorhersagen + weitere Phrasen-Uebergaenge
  Prio 5 — Outro-Struktur (nie am letzten Schlag, mind. 32 Beats vor Track-Ende)
```

Constraints:
- Hot A <-> Hot B: mind. 128 Beats (32 Bars) Abstand
- Memory Cues: mind. 32 Beats von jedem Hot Cue entfernt
- Memory Cues: mind. 32 Beats untereinander
- Memory Cues: mind. 32 Beats vor Track-Ende (kein letzter Schlag)

---

## Voraussetzungen

- macOS 13+ (Apple Silicon oder Intel)
- Python 3.12+
- Rekordbox 6.x oder 7.x installiert und konfiguriert
- Mixed In Key 11 *(optional — fuer MIK Cue-Integration)*

---

## Installation

```bash
# 1. Repository klonen
git clone <repo-url> rekordbox-autocue-tool
cd rekordbox-autocue-tool

# 2. Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# 3. Abhaengigkeiten installieren
pip install -r requirements.txt

# 4. libomp fuer LightGBM (macOS)
brew install libomp
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

# ML Engine
cue_engine: "auto"              # "auto" = ML wenn Modell vorhanden, sonst Triple-Check
use_learned_offsets: false       # Gelernte Offsets deaktiviert (ML-Predictor ersetzt dies)

# Playlist die analysiert werden soll
analyse_playlist: "--analyse-tracks"
```

---

## Workflow

### 1. Tracks vorbereiten

Erstelle in Rekordbox eine Playlist mit dem Namen `--analyse-tracks` (oder was in `config.yaml` konfiguriert ist) und fuege die zu analysierenden Tracks hinzu.

### 2. Bibliothek vektorisieren (einmalig / nach grossen Aenderungen)

Berechnet Feature-Vektoren fuer alle Rekordbox-Tracks (fuer CBR / 5k-Brain).

```bash
source .venv/bin/activate

# Schnell: nur Metadaten aus Rekordbox-DB (~30 Sekunden)
python -m app.vectorize

# Vollstaendig: mit echter Audio-Energie (~2-3 Stunden fuer 5.000 Tracks)
python -m app.vectorize --mode full
```

Ergebnis: `app/data/track_vectors.npz` + `app/data/track_meta.pkl`

### 3. ML-Modell trainieren (einmalig / nach grossen Aenderungen)

Trainiert LightGBM-Regressoren auf deinen manuell gesetzten Hot Cues.
Nutzt PWAV-Waveforms, MIK-Daten, PSSI-Phrasen und CBR-Twin-Positionen als Features.

```bash
source .venv/bin/activate

# Schritt 1: Trainingsdaten exportieren (~7 Minuten fuer 4.430 Tracks)
python tools/export_training_data.py --mode fast
# → data/ml/features_X.npy  (4430, 469)
# → data/ml/labels_Y.npy    (4430, 2)
# → data/ml/meta.pkl

# Schritt 2: Modelle trainieren + Cross-Validation
python tools/train_cue_model.py
# → models/ml_hot_a.lgb  (~3 MB)
# → models/ml_hot_c.lgb  (~3 MB)
# → 5-Fold CV mit Beat-Accuracy Metriken
```

Ohne trainiertes Modell: automatischer Fallback auf Triple-Check (kein Fehler, kein Abbruch).

### 4. Analyse starten

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

### 5. Vorhersagen vergleichen (Entwicklung)

Vergleicht die Vorhersagen gegen manuell gesetzte Hot Cues (Dry-Run, keine DB-Aenderungen):

```bash
python tools/compare_predictions.py
# → tools/compare_report.md
```

### 6. Terminal-Ausgabe

```
  AutoCue v27                              LIVE
  Playlist: --analyse-tracks

  ✓ Daft Punk – Get Lucky          A:2:14.08  B:5:58.00  C:6:30.04  Mem:7
    ✓ Hot A   2:14.08  [MIK+PWAV]     The Break
    ✓ Hot B   5:58.00  [DERIVED]      Setup (32b vor Drop)
    ✓ Hot C   6:30.04  [MIK+PHRASE]   The Last Drop
```

Quad-Check-Logik pro Cue:
- `[MIK+PHRASE]` — MIK und PSSI-Phrase uebereinstimmend (staerkster Konsens)
- `[MIK+PWAV]` / `[PHRASE+PWAV]` — ML bestaetigt MIK oder Phrase
- `[MIK+LIBRARY]` oder `[PHRASE+LIBRARY]` — CBR bestaetigt eine Quelle
- `[PWAV+LIBRARY]` — ML + CBR Konsens
- `[DERIVED]` — Hot B: immer aus Hot C abgeleitet, kein Quad-Check
- `✗ [—]` — Kein Konsens, Cue wird uebersprungen (Partial Success)

### 7. Intelligente Playlisten nach Monat/Jahr generieren

Erstellt 12 Smart-Playlisten (Monats-Filter) fuer ein Zieljahr. **Rekordbox muss dazu geschlossen sein.**

```bash
python -m app.playlists
```

---

## Projektstruktur

```
rekordbox-autocue-tool/
├── config.yaml              ← Alle Parameter hier konfigurieren
├── requirements.txt
├── README.md
├── app.log                  ← Automatisch erstellt (Laufzeit-Log)
│
├── app/                     ← Aktive Pipeline (v27)
│   ├── config.py            ← Config-Loader (config.yaml → Python dict)
│   ├── batch.py             ← CLI-Einstiegspunkt, rich Terminal-UI
│   ├── writer.py            ← Rekordbox DB-Writer + ProcessResult (v27-Pipeline)
│   ├── cue_logic.py         ← Quad-Check Cue-Hierarchie (v27)
│   ├── validator.py         ← Quad-Check Validator (MIK x Phrase x Library x ML)
│   ├── ml_predictor.py      ← LightGBM Inference (Hot A + Hot C Prediction)
│   ├── phrase_reader.py     ← PSSI-Phrasen aus ANLZ .EXT-Dateien
│   ├── segments.py          ← Akustische Segmentierung (SSM + Novelty)
│   ├── beatgrid.py          ← ANLZ-Beatgrid-Reader, Snap-to-Grid
│   ├── mik_scraper.py       ← Mixed In Key Daten (ID3 + SQLite)
│   ├── cbr.py               ← Case-Based Reasoning (5k-Brain)
│   ├── waveform.py          ← PWAV-Waveform lesen
│   ├── vectorize.py         ← CLI: Bibliothek vektorisieren
│   ├── learn.py             ← CLI: Korrekturen lernen
│   ├── playlists.py         ← CLI: Monats-Playlisten generieren
│   └── data/                ← Laufzeitdaten (predictions.jsonl, ...)
│
├── models/                  ← Trainierte LightGBM-Modelle
│   ├── ml_hot_a.lgb         ← Hot A Regressor (~3 MB)
│   └── ml_hot_c.lgb         ← Hot C Regressor (~3 MB)
│
├── data/ml/                 ← ML-Trainingsdaten
│   ├── features_X.npy       ← Feature-Matrix (4430, 469)
│   ├── labels_Y.npy         ← Labels (4430, 2)
│   └── meta.pkl             ← Track-Metadaten
│
└── tools/                   ← Entwicklungs-Tools
    ├── export_training_data.py  ← Feature-Export fuer ML-Training
    ├── train_cue_model.py       ← LightGBM Training + Cross-Validation
    └── compare_predictions.py   ← Vorhersage vs. Manuell Vergleich
```

---

## ML-Modell Details

Das LightGBM-Modell nutzt 469 Features pro Track:

| Slot | Feature | Dimension |
|------|---------|-----------|
| 0-399 | PWAV Waveform | 400 |
| 400-403 | BPM, Duration, Genre, Key | 4 |
| 404-412 | MIK Cue Rel-Positionen + Count | 9 |
| 413-453 | Phrase Kinds + Rel-Starts + Count | 41 |
| 454-466 | CBR Twin Positionen + Spacing | 13 |
| 467-468 | Beat Count, Bar Count | 2 |

Cross-Validation Ergebnisse (5-Fold, 4.430 Tracks):

| Cue | ML MAE | Baseline MAE | Verbesserung | ±8b Accuracy |
|-----|--------|--------------|--------------|--------------|
| Hot A | 34.6b | 37.7b | +8.3% | 28.8% |
| Hot C | 14.4b | 27.1b | +46.8% | 54.2% |

---

## Versionierung

| Version | Beschreibung |
|---------|-------------|
| `v26` | Produktionsreife Basis-Pipeline |
| `v26.1` | Triple-Check + PSSI-Phrasen |
| `v27` | Quad-Check + LightGBM ML trainiert auf 4.430 Tracks |

---

## Fehlerbehebung

**`Playlist nicht gefunden`**
→ In Rekordbox eine Playlist mit genau dem Namen aus `config.yaml` erstellen.

**`FEHLER: Rekordbox laeuft noch!`**
→ Rekordbox vollstaendig beenden, dann erneut starten.

**`FileNotFoundError: Track nicht in DB`**
→ Track zuerst in Rekordbox importieren (Analyse-Daten muessen vorhanden sein).

**MIK-Cues werden nicht erkannt**
→ Pruefen ob Mixed In Key 11 installiert ist und Tracks analysiert wurden.
→ Pfad in `config.yaml → mik_db_path` ggf. manuell setzen.

**ML-Modell nicht gefunden (kein Quad-Check)**
→ Modell trainieren: `python tools/export_training_data.py` → `python tools/train_cue_model.py`
→ Ohne Modell: automatischer Fallback auf Triple-Check (kein Fehler).

**`OSError: libomp.dylib not found`**
→ `brew install libomp` ausfuehren (LightGBM braucht OpenMP).

**Logs**
→ Details in `app.log` (rotierend, max. 5 MB x 3 Dateien).
