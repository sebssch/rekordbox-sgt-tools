"""
ML Cue Model Training — LightGBM Regressoren fuer Hot A + Hot C

Trainiert zwei separate Modelle auf den exportierten Features.
5-Fold Cross-Validation mit Beat-Accuracy Metriken.

Nutzung:
    python tools/export_training_data.py   # Erst exportieren
    python tools/train_cue_model.py        # Dann trainieren
"""
from __future__ import annotations

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR  = "data/ml"
MODEL_DIR = "models"

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
N_ESTIMATORS = 500
EARLY_STOPPING = 50


N_MEM_MODELS = 5  # Memory-Slot 2 bis 6

def load_dataset(data_dir: str = DATA_DIR):
    """Laedt exportierte Features, Labels, Metadata."""
    X = np.load(os.path.join(data_dir, "features_X.npy"))
    Y = np.load(os.path.join(data_dir, "labels_Y.npy"))
    Y_mem_path = os.path.join(data_dir, "labels_Y_mem.npy")
    Y_mem = np.load(Y_mem_path) if os.path.exists(Y_mem_path) else None
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return X, Y, meta, Y_mem


def rel_to_beats(delta_rel: float, duration: float, bpm: float) -> float:
    """Konvertiert relative Position-Differenz in Beats."""
    if bpm <= 0 or duration <= 0:
        return 999.0
    delta_sec = abs(delta_rel) * duration
    return delta_sec * (bpm / 60.0)


def cross_validate(X: np.ndarray, Y: np.ndarray, meta: list,
                   n_folds: int = 5) -> dict:
    """
    5-Fold Cross-Validation mit Beat-Accuracy Metriken.
    Vergleicht ML-Modell gegen Baseline (globaler Median).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {"hot_a": [], "hot_c": [], "baseline_a": [], "baseline_c": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        meta_val = [meta[i] for i in val_idx]

        # Baseline: Median der Trainings-Labels
        baseline_a = float(np.median(y_train[:, 0]))
        baseline_c = float(np.median(y_train[:, 1]))

        # Hot A Modell
        ds_train_a = lgb.Dataset(X_train, label=y_train[:, 0])
        ds_val_a   = lgb.Dataset(X_val, label=y_val[:, 0], reference=ds_train_a)
        model_a = lgb.train(
            LGB_PARAMS, ds_train_a, N_ESTIMATORS,
            valid_sets=[ds_val_a],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)],
        )
        pred_a = model_a.predict(X_val)

        # Hot C Modell
        ds_train_c = lgb.Dataset(X_train, label=y_train[:, 1])
        ds_val_c   = lgb.Dataset(X_val, label=y_val[:, 1], reference=ds_train_c)
        model_c = lgb.train(
            LGB_PARAMS, ds_train_c, N_ESTIMATORS,
            valid_sets=[ds_val_c],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)],
        )
        pred_c = model_c.predict(X_val)

        # Metriken pro Track
        for i in range(len(X_val)):
            dur = meta_val[i]["duration"]
            bpm = meta_val[i]["bpm"]

            # ML
            delta_a = rel_to_beats(pred_a[i] - y_val[i, 0], dur, bpm)
            delta_c = rel_to_beats(pred_c[i] - y_val[i, 1], dur, bpm)
            results["hot_a"].append(delta_a)
            results["hot_c"].append(delta_c)

            # Baseline
            bl_a = rel_to_beats(baseline_a - y_val[i, 0], dur, bpm)
            bl_c = rel_to_beats(baseline_c - y_val[i, 1], dur, bpm)
            results["baseline_a"].append(bl_a)
            results["baseline_c"].append(bl_c)

    return results


def print_results(results: dict):
    """Druckt Cross-Validation Ergebnisse."""
    for cue_type, key, bl_key in [("Hot A", "hot_a", "baseline_a"),
                                    ("Hot C", "hot_c", "baseline_c")]:
        deltas = np.array(results[key])
        bl = np.array(results[bl_key])

        within_2  = np.mean(deltas <= 2.0) * 100
        within_8  = np.mean(deltas <= 8.0) * 100
        bl_within_2 = np.mean(bl <= 2.0) * 100
        bl_within_8 = np.mean(bl <= 8.0) * 100

        print(f"\n  {cue_type}:")
        print(f"    ML Model:  MAE={np.mean(deltas):.1f}b  "
              f"Median={np.median(deltas):.1f}b  "
              f"±2b={within_2:.1f}%  ±8b={within_8:.1f}%")
        print(f"    Baseline:  MAE={np.mean(bl):.1f}b  "
              f"Median={np.median(bl):.1f}b  "
              f"±2b={bl_within_2:.1f}%  ±8b={bl_within_8:.1f}%")
        improvement = (np.mean(bl) - np.mean(deltas)) / np.mean(bl) * 100
        print(f"    Verbesserung: {improvement:+.1f}% MAE")


def train_final_models(X: np.ndarray, Y: np.ndarray) -> tuple:
    """Trainiert finale Modelle auf dem gesamten Datensatz."""
    # Hot A
    ds_a = lgb.Dataset(X, label=Y[:, 0])
    model_a = lgb.train(LGB_PARAMS, ds_a, N_ESTIMATORS)

    # Hot C
    ds_c = lgb.Dataset(X, label=Y[:, 1])
    model_c = lgb.train(LGB_PARAMS, ds_c, N_ESTIMATORS)

    return model_a, model_c


def train_memory_models(X: np.ndarray, Y_mem: np.ndarray) -> list:
    """
    Trainiert Memory-Cue-Modelle fuer Slot 2-6 (5 Regressoren).
    Nur Tracks mit tatsaechlichen Memory Cues an dieser Position verwenden.
    """
    models = []
    for i in range(N_MEM_MODELS):
        # Nur Tracks mit gueltigen Labels (> 0) fuer diesen Slot
        mask = Y_mem[:, i] > 0.01  # > 1% Position = gueltig
        X_valid = X[mask]
        y_valid = Y_mem[:, i][mask]

        if len(X_valid) < 100:
            print(f"    Mem Slot {i+2}: zu wenig Daten ({len(X_valid)}), uebersprungen")
            models.append(None)
            continue

        ds = lgb.Dataset(X_valid, label=y_valid)
        model = lgb.train(LGB_PARAMS, ds, N_ESTIMATORS)
        models.append(model)
        print(f"    Mem Slot {i+2}: {len(X_valid)} Tracks, trainiert")

    return models


def cv_memory_models(X: np.ndarray, Y_mem: np.ndarray, meta: list,
                     n_folds: int = 5) -> dict:
    """Cross-Validation fuer Memory-Cue-Modelle."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for slot in range(N_MEM_MODELS):
        mask = Y_mem[:, slot] > 0.01
        X_valid = X[mask]
        y_valid = Y_mem[:, slot][mask]
        meta_valid = [meta[i] for i, m in enumerate(mask) if m]

        if len(X_valid) < 100:
            continue

        deltas = []
        bl_deltas = []

        for train_idx, val_idx in kf.split(X_valid):
            X_tr, X_vl = X_valid[train_idx], X_valid[val_idx]
            y_tr, y_vl = y_valid[train_idx], y_valid[val_idx]
            meta_vl = [meta_valid[i] for i in val_idx]

            baseline = float(np.median(y_tr))

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            ds_vl = lgb.Dataset(X_vl, label=y_vl, reference=ds_tr)
            model = lgb.train(
                LGB_PARAMS, ds_tr, N_ESTIMATORS,
                valid_sets=[ds_vl],
                callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)],
            )
            pred = model.predict(X_vl)

            for j in range(len(X_vl)):
                dur = meta_vl[j]["duration"]
                bpm = meta_vl[j]["bpm"]
                deltas.append(rel_to_beats(pred[j] - y_vl[j], dur, bpm))
                bl_deltas.append(rel_to_beats(baseline - y_vl[j], dur, bpm))

        results[f"mem_{slot+2}"] = {
            "ml": np.array(deltas),
            "baseline": np.array(bl_deltas),
            "n_tracks": len(X_valid),
        }

    return results


def print_feature_importances(model, name: str, top_n: int = 15):
    """Zeigt die wichtigsten Features."""
    importance = model.feature_importance(importance_type="gain")
    indices = np.argsort(importance)[::-1][:top_n]

    # Feature-Namen Mapping
    names = []
    for i in range(len(importance)):
        if i < 400:
            names.append(f"pwav_{i}")
        elif i == 400:
            names.append("bpm")
        elif i == 401:
            names.append("duration")
        elif i == 402:
            names.append("genre_hash")
        elif i == 403:
            names.append("key")
        elif 404 <= i <= 411:
            names.append(f"mik_pos_{i - 404}")
        elif i == 412:
            names.append("mik_count")
        elif 413 <= i <= 432:
            names.append(f"phrase_kind_{i - 413}")
        elif 433 <= i <= 452:
            names.append(f"phrase_start_{i - 433}")
        elif i == 453:
            names.append("phrase_count")
        elif 454 <= i <= 458:
            names.append(f"twin_hot_a_{i - 454}")
        elif 459 <= i <= 463:
            names.append(f"twin_hot_c_{i - 459}")
        elif i == 464:
            names.append("cbr_median_a")
        elif i == 465:
            names.append("cbr_median_c")
        elif i == 466:
            names.append("cbr_spacing")
        elif i == 467:
            names.append("beat_count")
        elif i == 468:
            names.append("bar_count")
        else:
            names.append(f"feat_{i}")

    print(f"\n  Top {top_n} Features ({name}):")
    for rank, idx in enumerate(indices, 1):
        n = names[idx] if idx < len(names) else f"feat_{idx}"
        print(f"    {rank:2d}. {n:<20s}  gain={importance[idx]:.1f}")


def main():
    print("=" * 60)
    print("  ML Cue Model Training")
    print("=" * 60)

    # 1. Daten laden
    print("\n1. Daten laden...")
    X, Y, meta, Y_mem = load_dataset()
    print(f"   {X.shape[0]} Tracks, {X.shape[1]} Features")
    print(f"   Hot A rel: {np.mean(Y[:, 0]):.3f} ± {np.std(Y[:, 0]):.3f}")
    print(f"   Hot C rel: {np.mean(Y[:, 1]):.3f} ± {np.std(Y[:, 1]):.3f}")
    if Y_mem is not None:
        print(f"   Memory Labels: {Y_mem.shape}")

    # 2. Cross-Validation (Hot Cues)
    print("\n2. 5-Fold Cross-Validation (Hot Cues)...")
    results = cross_validate(X, Y, meta)
    print_results(results)

    # 2b. Cross-Validation (Memory Cues)
    if Y_mem is not None:
        print("\n2b. 5-Fold Cross-Validation (Memory Cues)...")
        mem_results = cv_memory_models(X, Y_mem, meta)
        for slot_name, res in mem_results.items():
            ml_mae = np.mean(res["ml"])
            bl_mae = np.mean(res["baseline"])
            within_8 = np.mean(res["ml"] <= 8.0) * 100
            improvement = (bl_mae - ml_mae) / bl_mae * 100
            print(f"    {slot_name}: ML MAE={ml_mae:.1f}b  "
                  f"Baseline={bl_mae:.1f}b  "
                  f"±8b={within_8:.1f}%  "
                  f"Verbesserung={improvement:+.1f}%  "
                  f"(N={res['n_tracks']})")

    # 3. Finale Modelle trainieren
    print("\n3. Finale Modelle trainieren (gesamter Datensatz)...")
    model_a, model_c = train_final_models(X, Y)

    # 4. Feature Importances
    print_feature_importances(model_a, "Hot A")
    print_feature_importances(model_c, "Hot C")

    # 5. Speichern (Hot Cue Modelle)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_a.save_model(os.path.join(MODEL_DIR, "ml_hot_a.lgb"))
    model_c.save_model(os.path.join(MODEL_DIR, "ml_hot_c.lgb"))
    print(f"\n  Hot Cue Modelle gespeichert: {MODEL_DIR}/ml_hot_a.lgb, ml_hot_c.lgb")

    a_size = os.path.getsize(os.path.join(MODEL_DIR, "ml_hot_a.lgb"))
    c_size = os.path.getsize(os.path.join(MODEL_DIR, "ml_hot_c.lgb"))
    print(f"  Hot A: {a_size / 1024:.0f} KB  |  Hot C: {c_size / 1024:.0f} KB")

    # 5b. Memory Cue Modelle trainieren und speichern
    if Y_mem is not None:
        print("\n  Memory Cue Modelle trainieren...")
        mem_models = train_memory_models(X, Y_mem)
        for i, model in enumerate(mem_models):
            if model is not None:
                path = os.path.join(MODEL_DIR, f"ml_mem_{i+2}.lgb")
                model.save_model(path)
                size = os.path.getsize(path) / 1024
                print(f"    ml_mem_{i+2}.lgb: {size:.0f} KB")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
