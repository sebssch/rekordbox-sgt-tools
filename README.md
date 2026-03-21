# Rekordbox AutoCue v28.1

> ML-basierte Cue-Point-Vorhersage fuer Rekordbox — lernt aus deinen Tracks.

AutoCue analysiert deine Tracks akustisch (Spektrogramm + Waveform), nutzt Mixed In Key-Daten und ein LightGBM-Modell trainiert auf deiner Bibliothek — vollautomatisch, ohne Rekordbox-Cloud.

<!-- TODO: Screenshot einfuegen -->
<!-- ![AutoCue Terminal](images/terminal.png) -->

---

## Features

| Feature | Beschreibung |
|---------|-------------|
| **Beatgrid-Sync** | Cues snappen exakt auf Downbeats via ANLZ-Beatgrid |
| **Quad-Check** | Jeder Hot Cue unabhaengig validiert: MIK x Phrase x Library x ML |
| **LightGBM ML** | Supervised Learning, 693-dim Feature-Vektor |
| **Spektral-Features** | Mel-Spektrogramm-Analyse: Band-Energien, Novelty, Onset, Contrast, Flatness |
| **Memory Cue Engine** | Phrasen-basierte Memory Cues: Intro/Outro-Gliederung + Strukturmitte |
| **PSSI-Phrasen** | Rekordbox-Phrase-Analyse aus ANLZ .EXT-Dateien (Intro/Drop/Break/Outro) |
| **Akustische Segmentierung** | SSM + Novelty + Energy + Percussive Ratio |
| **Datenbank-Backup** | Automatisches Backup der Rekordbox-DB vor jeder Aenderung (max. 5 Backups) |
| **Mixed In Key-Integration** | Mixed In Key Cue-Positionen als zusaetzliche Quelle (optional, nicht erforderlich) |
| **CBR (5k-Brain)** | Case-Based Reasoning — findet Zwillings-Tracks aus deiner Sammlung |
| **Konfigurierbar** | Alle Parameter in `config.yaml`, kein Hardcoding |

---

## Cue-Hierarchie

### Hot Cues

```
Hot Cue A  →  "The Break"     — Break-Einstieg (Percussion faellt nach Intro)
Hot Cue B  →  "The Setup"     — Exakt N Beats vor Hot C (Default: 32 Beats)
Hot Cue C  →  "The Last Drop" — Letzter Drop (Segment + Visual Edge)
```

<!-- TODO: Screenshot Rekordbox mit Hot Cues -->
<!-- ![Hot Cues in Rekordbox](images/hot_cues.png) -->

### Memory Cues

Memory Cues gliedern die Track-Struktur relativ zu den Hot Cues:

```
Prio 1 — Erster Downbeat (immer gesetzt)
Prio 2 — Intro-Gliederung (rueckwaerts von Hot A)
           → max 3 Cues, alle 32 Beats Richtung Hot A
           → bei Intro >= 192 Beats (6+ Slots): alle 64 Beats
Prio 4 — Struktur-Mitte: PSSI-Phrasen direkt uebernehmen
           → Phrasen >= 16 Beats werden beruecksichtigt
           → Positionen direkt nutzen (kein globales 32-Raster-Snapping)
           → Track-interner Versatz (z.B. mod32=16) wird respektiert
           → Fallback ohne PSSI: alle 32 Beats ab Hot A
Prio 5 — Outro-Gliederung (vorwaerts von Hot C)
           → Nur bei Kick-Outro (Energie > 0.4, sauberer Beat-Auslauf)
           → Kein Cue bei Fade-Outro (Vocals/Melodie auslaufend)
           → Abstands-Regeln relativ zu Hot C:
             1-4 Slots: alle 32 Beats
             5 Slots:   erster +64b, Rest +32b
             6+ Slots:  alle 64 Beats
           → Kein Cue am Track-Ende (mind. 32 Beats Abstand)

ML-Cues nur als Validierung: "(ML)" im Comment wenn ML uebereinstimmt.
MIK-Daten werden fuer Memory Cues NICHT verwendet.
```

<!-- TODO: Screenshot Memory Cues in Rekordbox -->
<!-- ![Memory Cues](images/memory_cues.png) -->

### Constraints

- Hot A ↔ Hot B: mind. 128 Beats (32 Bars) Abstand
- Memory Cues: mind. 32 Beats von jedem Hot Cue entfernt (0.5 Beat Toleranz)
- Memory Cues: mind. 32 Beats untereinander (0.5 Beat Toleranz)
- Memory Cues: mind. 32 Beats vor Track-Ende (kein letzter Schlag)
- Lieber keinen Memory Cue als einen falschen

---

## Voraussetzungen

- macOS 13+ (Apple Silicon oder Intel)
- Python 3.12+
- [Rekordbox](https://rekordbox.com/de/) 6.x oder 7.x installiert und konfiguriert
- [Mixed In Key](https://mixedinkey.com/) 11 *(optional — das Skript laeuft auch ohne MIK)*

---

## Rekordbox Datenbank

AutoCue liest und schreibt direkt in die Rekordbox `master.db`. Der Standard-Pfad unter macOS:

```
~/Library/Pioneer/rekordbox/master.db
```

Dieser Pfad kann in `config.yaml` ueberschrieben werden:

```yaml
rekordbox_db_path: "~/Library/Pioneer/rekordbox/master.db"
```

> **Sicherheit:** Vor jeder Live-Aenderung wird automatisch ein Backup erstellt (`master.db.backup_YYYYMMDD_HHMMSS`). Es werden maximal 5 Backups aufbewahrt — aeltere werden automatisch geloescht.

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

# Spektral-Features
spectral_mode: "custom"         # "custom" | "openl3" | "auto" | "off"
spectral_cache_dir: "data/spectral_cache"

# Rekordbox Datenbank (leer = macOS-Standard)
rekordbox_db_path: ""

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
Nutzt PWAV-Waveforms, PSSI-Phrasen, Spektral-Features, CBR-Twin-Positionen und optional MIK-Daten.

```bash
source .venv/bin/activate

# Schritt 1: Trainingsdaten exportieren (beim ersten Mal ~17 Minuten, danach Cache)
python tools/export_training_data.py --spectral custom

# Schritt 2: Modelle trainieren + Cross-Validation
python tools/train_cue_model.py
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
```

### 6. Terminal-Ausgabe

```
  AutoCue v28.1                            DRY-RUN
  Playlist: --analyse-tracks (12 Tracks)

  ✓ Daft Punk – Get Lucky
    Hot A  [2:14.08]  Bar  71  — The Break
    Hot B  [5:58.00]  Bar 179  — Setup (32b vor Drop)
    Hot C  [6:30.04]  Bar 195  — The Last Drop
    Mem    [0:00.12]  Bar   1  — Erster Downbeat        (orange)
    Mem    [1:42.00]  Bar  51  — Intro Beat 64          (orange)
    Mem    [4:26.08]  Bar 133  — Chorus Start           (orange)
    Mem    [7:02.04]  Bar 211  — Outro +32b             (orange)
```

<!-- TODO: Screenshot Terminal-Ausgabe -->
<!-- ![Terminal-Ausgabe](images/terminal_output.png) -->

Memory Cues werden in der Terminal-Ausgabe orange angezeigt.

Quad-Check-Logik pro Hot Cue:
- `[MIK+PHRASE]` — MIK und PSSI-Phrase uebereinstimmend (staerkster Konsens)
- `[MIK+PWAV]` / `[PHRASE+PWAV]` — ML bestaetigt MIK oder Phrase
- `[MIK+LIBRARY]` oder `[PHRASE+LIBRARY]` — CBR bestaetigt eine Quelle
- `[PWAV+LIBRARY]` — ML + CBR Konsens
- `[DERIVED]` — Hot B: immer aus Hot C abgeleitet, kein Quad-Check
- `✗ [—]` — Kein Konsens, Cue wird uebersprungen (Partial Success)

### 7. Cues einer Playlist zuruecksetzen

Loescht alle Cues (Hot + Memory) fuer Tracks in einer Playlist. Doppelte Sicherheitsabfrage.

```bash
python tools/reset_playlist_cues.py "--analyse-tracks"
```

### 8. Intelligente Playlisten nach Monat/Jahr generieren

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
├── app/                     ← Aktive Pipeline
│   ├── config.py            ← Config-Loader (config.yaml → Python dict)
│   ├── batch.py             ← CLI-Einstiegspunkt, rich Terminal-UI
│   ├── writer.py            ← Rekordbox DB-Writer + ProcessResult
│   ├── cue_logic.py         ← Quad-Check Cue-Hierarchie + Memory Cue Engine
│   ├── validator.py         ← Quad-Check Validator (MIK x Phrase x Library x ML)
│   ├── ml_predictor.py      ← LightGBM Inference (Hot A + Hot C + Memory Prediction)
│   ├── spectral.py          ← Spektral-Feature-Extraktion (custom/openl3/auto)
│   ├── phrase_reader.py     ← PSSI-Phrasen aus ANLZ .EXT-Dateien
│   ├── segments.py          ← Akustische Segmentierung (SSM + Novelty)
│   ├── beatgrid.py          ← ANLZ-Beatgrid-Reader, Snap-to-Grid
│   ├── mik_scraper.py       ← Mixed In Key Daten (ID3 + SQLite, optional)
│   ├── cbr.py               ← Case-Based Reasoning (5k-Brain)
│   ├── waveform.py          ← PWAV-Waveform lesen
│   ├── vectorize.py         ← CLI: Bibliothek vektorisieren
│   ├── learn.py             ← CLI: Korrekturen lernen
│   ├── playlists.py         ← CLI: Monats-Playlisten generieren
│   └── data/                ← Laufzeitdaten (predictions.jsonl, ...)
│
├── images/                  ← Screenshots und Bilder fuer README
│
├── models/                  ← Trainierte LightGBM-Modelle
│   ├── ml_hot_a.lgb
│   ├── ml_hot_c.lgb
│   └── ml_mem_2..6.lgb
│
├── data/ml/                 ← ML-Trainingsdaten (werden lokal generiert)
│
├── data/spectral_cache/     ← Gecachte Spektral-Features (pro Track)
│
└── tools/                   ← Entwicklungs-Tools
    ├── export_training_data.py  ← Feature-Export fuer ML-Training
    ├── train_cue_model.py       ← LightGBM Training + Cross-Validation
    ├── compare_predictions.py   ← Vorhersage vs. Manuell Vergleich
    └── reset_playlist_cues.py   ← Cue-Reset fuer Playlist (doppelte Bestaetigung)
```

---

## ML-Modell Details

Das LightGBM-Modell nutzt 693 Features pro Track (469 Base + 224 Spektral):

| Slot | Feature | Dimension |
|------|---------|-----------|
| 0-399 | PWAV Waveform | 400 |
| 400-403 | BPM, Duration, Genre, Key | 4 |
| 404-412 | MIK Cue Rel-Positionen + Count (optional) | 9 |
| 413-453 | Phrase Kinds + Rel-Starts + Count | 41 |
| 454-466 | CBR Twin Positionen + Spacing | 13 |
| 467-468 | Beat Count, Bar Count | 2 |
| 469-500 | Spektral: Low-Band Energie (20-300 Hz) | 32 |
| 501-532 | Spektral: Mid-Band Energie (300-4000 Hz) | 32 |
| 533-564 | Spektral: High-Band Energie (4000+ Hz) | 32 |
| 565-596 | Spektral: Spectral Novelty | 32 |
| 597-628 | Spektral: Onset Strength | 32 |
| 629-660 | Spektral: Spectral Contrast | 32 |
| 661-692 | Spektral: Spectral Flatness | 32 |

### Spektral-Modi

| Mode | Feature-Dim | Beschreibung |
|------|-------------|-------------|
| `off` | 469 | Nur PWAV + Metadaten (schnellster Modus) |
| `custom` | 693 | +224 eigene librosa-Features (7 x 32 Segmente) |
| `openl3` | 981 | +512 vortrainierte CNN-Embeddings via [torchopenl3](https://github.com/hugofloresgarcia/torchopenl3) |
| `auto` | 1205 | custom + openl3 kombiniert |

> **Hinweis zu OpenL3:** Der `openl3`-Modus nutzt [torchopenl3](https://github.com/hugofloresgarcia/torchopenl3), ein vortrainiertes CNN fuer Audio-Embeddings. Dieser Modus ist aktuell fuer Python 3.14 nicht verfuegbar, da die Abhaengigkeit `resampy` das entfernte `imp`-Modul verwendet. Sobald `resampy` aktualisiert wird, kann dieser Modus aktiviert werden.

Die Spektral-Features werden pro Track einmal berechnet (~1s) und im Cache gespeichert (`data/spectral_cache/`).

---

## Versionierung

| Version | Beschreibung |
|---------|-------------|
| `v26` | Produktionsreife Basis-Pipeline |
| `v26.1` | Triple-Check + PSSI-Phrasen |
| `v27` | Quad-Check + LightGBM ML |
| `v27.1` | Memory Cue ML-Modelle (Slot 2-6), Konfidenz-Scoring, Phrase×ML Source |
| `v27.2` | Spektral-Features: Mel-Spektrogramm-Analyse (+224 dims) |
| `v27.3` | Memory Cues: 32-Beat-Grid-Snapping, PSSI-Phrasen primaer, MIK entfernt |
| `v28` | Memory Cue Logik komplett neu: Intro/Outro relativ zu Hot Cues, Kick-Outro-Erkennung |
| `v28.1` | Phrasen direkt nutzen (kein 32-Raster-Snapping), 16-Beat-Phrasen, Spacing-Toleranz |

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
→ MIK ist optional — das Skript funktioniert auch ohne Mixed In Key.

**ML-Modell nicht gefunden (kein Quad-Check)**
→ Modell trainieren: `python tools/export_training_data.py` → `python tools/train_cue_model.py`
→ Ohne Modell: automatischer Fallback auf Triple-Check (kein Fehler).

**`OSError: libomp.dylib not found`**
→ `brew install libomp` ausfuehren (LightGBM braucht OpenMP).

**Logs**
→ Details in `app.log` (rotierend, max. 5 MB x 3 Dateien).
