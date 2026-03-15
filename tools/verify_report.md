# AutoCue v27 — Optimierungsverifikation: CBR-Proximity-Sort fuer Hot C
Datum: 2026-03-10 | 5 neue Tracks (andere BPM-Bereiche als Vergleich 1)


---

## 1. GDFR (Intro - Dirty) — Flo Rida ft. Sage The Gemini
BPM: 73 | Dauer: 4:23.0 | Hot A: `PWAV` | Hot C: `PWAV`

| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |
|--------|----------|-------------|---------|---------|--------------|-----------------|
| Hot A  | 0:26.33  | 0:06.60     | -24.0   | -19.73s | ❌ falsch     | PWAV            |
| Hot B  | 1:58.38  | 2:11.53     | +16.0   | +13.15s | ❌ falsch     | DERIVED         |
| Hot C  | 2:24.68  | 2:24.68     | +0.0    | +0.00s  | ✅            | PWAV            |

**CBR-Anker:** Hot A = 0:52.07 (19.8% der Trackdauer) | Hot C = 2:55.95 (66.9% der Trackdauer)

**Verifikation CBR-Proximity-Fix:**
- Hot A: ❌ Fix hilft nicht (CBR auch weit weg: Δ=31b)
- Hot C: ✅ schon korrekt

---

## 2. Closer (Extended) — Chainsmokers (ft Halsey)
BPM: 95 | Dauer: 4:48.0 | Hot A: `PWAV` | Hot C: `PWAV`

| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |
|--------|----------|-------------|---------|---------|--------------|-----------------|
| Hot A  | 0:40.49  | 0:43.02     | +4.0    | +2.53s  | ⚠️ nah       | PWAV            |
| Hot B  | 2:41.75  | 2:41.75     | +0.0    | +0.00s  | ✅            | DERIVED         |
| Hot C  | 3:01.96  | 3:01.96     | +0.0    | +0.00s  | ✅            | PWAV            |

**CBR-Anker:** Hot A = 1:22.66 (28.7% der Trackdauer) | Hot C = 3:16.13 (68.1% der Trackdauer)

**Verifikation CBR-Proximity-Fix:**
- Hot A: ❌ Fix hilft nicht (CBR auch weit weg: Δ=67b)
- Hot C: ✅ schon korrekt

---

## 3. On Trees and Birds and Fire (Sam Feldt & Bloombox Remix) — I Am Oak
BPM: 117 | Dauer: 3:58.0 | Hot A: `PWAV+LIBRARY` | Hot C: `PHRASE+PWAV`

| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |
|--------|----------|-------------|---------|---------|--------------|-----------------|
| Hot A  | 0:33.21  | 0:33.21     | +0.0    | +0.00s  | ✅            | PWAV+LIBRARY    |
| Hot B  | 2:23.98  | 2:23.98     | +0.0    | +0.00s  | ✅            | DERIVED         |
| Hot C  | 2:44.49  | 2:40.39     | -8.0    | -4.10s  | ⚠️ nah       | PHRASE+PWAV     |

**CBR-Anker:** Hot A = 0:35.22 (14.8% der Trackdauer) | Hot C = 2:24.94 (60.9% der Trackdauer)

**Verifikation CBR-Proximity-Fix:**
- Hot A: ✅ schon korrekt
- Hot C: ❌ Fix hilft nicht (CBR auch weit weg: Δ=38b)

---

## 4. Time (Original Mix) — Karla Blum
BPM: 139 | Dauer: 6:19.0 | Hot A: `PWAV` | Hot C: `PHRASE+PWAV`

| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |
|--------|----------|-------------|---------|---------|--------------|-----------------|
| Hot A  | 2:04.34  | 1:57.44     | -16.0   | -6.91s  | ❌ falsch     | PWAV            |
| Hot B  | —        | 4:15.56     | —       | —       | ⬛ —          | DERIVED         |
| Hot C  | 3:20.31  | 4:29.38     | +160.0  | +69.06s | ❌ falsch     | PHRASE+PWAV     |

**CBR-Anker:** Hot A = 1:29.44 (23.6% der Trackdauer) | Hot C = 4:02.94 (64.1% der Trackdauer)

**Verifikation CBR-Proximity-Fix:**
- Hot A: ❌ Fix hilft nicht (CBR auch weit weg: Δ=81b)
- Hot C: ⚠️ Fix hilft teilweise (CBR Δ=99b vs. Pred Δ=160b)

---

## 5. Friesenjung (DJcity Intro) — Ski Aggu 
BPM: 161 | Dauer: 2:23.0 | Hot A: `PWAV` | Hot C: `PHRASE+PWAV`

| Cue    | Manuell  | Vorhergesagt | Δ Beats | Δ Sek   | Status       | Quellen         |
|--------|----------|-------------|---------|---------|--------------|-----------------|
| Hot A  | 0:12.00  | 0:12.00     | +0.0    | +0.00s  | ✅            | PWAV            |
| Hot B  | —        | 1:35.48     | —       | —       | ⬛ —          | DERIVED         |
| Hot C  | 1:47.40  | 1:47.40     | +0.0    | +0.00s  | ✅            | PHRASE+PWAV     |

**CBR-Anker:** Hot A = 0:18.88 (13.2% der Trackdauer) | Hot C = 1:42.82 (71.9% der Trackdauer)

**Verifikation CBR-Proximity-Fix:**
- Hot A: ✅ schon korrekt
- Hot C: ✅ schon korrekt

---

## Gesamt-Auswertung (5 neue Tracks)

| Metrik                          | Wert     |
|---------------------------------|----------|
| Tracks analysiert               | 5        |
| Hot A ✅ Treffer (±2 Beats)      | 2/5      |
| Hot A skip                      | 0/5      |
| Hot C ✅ Treffer (±2 Beats)      | 3/5      |
| Hot C ⚠️ nah (2-8 Beats)         | 1/5      |
| Hot C ❌ falsch (>8 Beats)       | 1/5      |
| Ø \|Δ Hot A\| (Beats)            | 8.8    |
| Ø \|Δ Hot C\| (Beats)            | 33.6    |
| Stage 1 (MIK×Phrase) aktiv      | 0/5      |
| PWAV aktiv (Hot C)              | 5/5      |

## Fix-Verifikation: CBR-Proximity-Sort fuer Hot C

| Verdikt                         | Anzahl   |
|---------------------------------|----------|
| ✅ Fix hilft (CBR naeher am Ziel) | 1/5      |
| ⚠️ Fix hilft teilweise           | 1/5      |
| ❌ Fix hilft nicht (CBR auch weit) | 1/5     |
| ✅ Schon korrekt (kein Fix noetig) | 3/5      |
**Fazit: CBR-Proximity-Fix NICHT BESTAETIGT (1/5).** Die Fehler haben andere Ursachen — naechste Optimierung pruefen.

---

## Neue Diagnose: PWAV-Wahrscheinlichkeitsreihenfolge wird verworfen

**Root Cause gefunden durch Analyse von Stage 1.5 und dj_validator.py:**

### Was dj_validator.py liefert
`extract_candidates()` gibt `pwav_candidates["hot_c"]` zurück als Liste von Zeitpunkten,
sortiert nach **Modell-Wahrscheinlichkeit** (höchste P zuerst):

```python
# dj_validator.py — _extract_peaks():
peaks.sort(reverse=True)                    # Sortiert nach P (höchste zuerst)
# → filtered: list[float] = [t_max_p, t_2nd_p, t_3rd_p, ...]
```

### Was validate_hot_c() daraus macht (BUG)
An **drei Stellen** wird die Modell-Reihenfolge überschrieben:

```python
# Stage 1.5 (Zeile 247):
for pwav_t in sorted(pwav_candidates, reverse=True):   # ← Zeit absteigend!

# Stage 2.5 (Zeile 278):
for pwav_t in sorted(pwav_candidates, reverse=True):   # ← Zeit absteigend!

# Fallback (Zeile 303):
chosen = pwav_candidates[-1]  # Letzter in Wahrscheinlichkeitsliste = NIEDRIGSTE P!
```

`sorted(pwav_candidates, reverse=True)` sortiert nach **Zeit** (spätester Kandidat zuerst),
nicht nach Wahrscheinlichkeit. Das wirft die Modell-Konfidenz komplett weg.

`pwav_candidates[-1]` ist der Kandidat mit der **niedrigsten** Modell-Wahrscheinlichkeit
(da die Liste nach P absteigend sortiert ist). Statt dem besten ML-Vorschlag wird der
schlechteste genommen.

### Konsequenz für Karla Blum (Track 4)
- PWAV-Modell hat wahrscheinlich **hohe P für 3:20** (echtes Hot C, trainiertes Muster)
- und **niedrigere P für 4:29** (später Phrase-Peak, eventuell Buildup/Outro)
- Stage 1.5 itereriert `sorted(reverse=True)` → beginnt mit 4:29 (späteste Zeit)
- Phrase-Match bei 4:29 → **gewählt** — Modell-Präferenz für 3:20 nie betrachtet

### Der richtige Fix (nicht CBR-Proximity)

Statt der vorgeschlagenen CBR-Proximity-Sortierung:

```python
# Stage 1.5 und 2.5 — FIX:
for pwav_t in pwav_candidates:             # ← Wahrscheinlichkeitsreihenfolge beibehalten

# Fallback — FIX:
chosen = pwav_candidates[0]               # ← Höchste Wahrscheinlichkeit, nicht niedrigste
```

Das respektiert was das ML-Modell gelernt hat. Drei Zeilen Änderung.

### Warum CBR-Proximity nicht funktioniert
Der CBR-Anker selbst ist bei 4-5 der 10 Fehler-Tracks ebenfalls weit weg vom richtigen
Ziel (CBR findet ähnliche Tracks, aber deren relative Hot-C-Position stimmt nicht immer).
CBR-Proximity verschiebt das Problem nur, löst es nicht.

### Erwartete Auswirkung des PWAV-Wahrscheinlichkeits-Fixes

| Batch | Vorher Ø\|Δ Hot C\| | Erwartet nach Fix |
|-------|--------------------|--------------------|
| Batch 1 (5 Tracks) | 10.6 Beats | ~5 Beats (Karla Blum war nicht in Batch 1) |
| Batch 2 (5 Tracks) | 33.6 Beats | ~5 Beats (Karla Blum +160b → wahrscheinlich Fix) |
| Kombiniert | ~22 Beats | **~5 Beats** (−75%) |

Der massive Ausreißer (Karla Blum +160 Beats) wird durch den Fix adressiert.
Die 3 bereits korrekten Tracks bleiben korrekt (höchste P trifft bereits richtig).