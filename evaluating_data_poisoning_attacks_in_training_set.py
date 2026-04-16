# -*- coding: utf-8 -*-
"""
evaluating_data_poisoning_attacks_in_training_set.py
=====================================================
Loads pre-computed embeddings (*_with_labels.pkl) for every data-poisoning
attack variant, tunes classifier hyperparameters on the VALIDATION SET,
evaluates the best model on the TEST SET, and checks whether attacks managed
to trick the model on multiple-choice jailbreak prompts.

After each model is trained it is stored via `register_model` so that results
can be plotted later.

Run in Google Colab after the embedding PKL files have been generated:
    %run evaluating_data_poisoning_attacks_in_training_set.py

# ─── Expected PKL structure (*_with_labels.pkl) ───────────────────────────
# {
#   "<dataset_name>": {
#       "train":  {"bow": matrix, "w2v": ndarray, "glove_twitter": ndarray,
#                  "glove_wiki": ndarray, "fasttext": ndarray, "sbert": ndarray},
#       "val":    { ... same keys ... },
#       "test":   { ... same keys ... },
#       "y_train": array-like,   # labels for training split
#       "y_val":   array-like,   # labels for validation split
#       "y_test":  array-like,   # labels for test split
#       "vectorizers": {"bow": CountVectorizer, ...}   # optional
#   },
#   ...
# }
# Labels are either strings ("jailbreak"/"benign") or integers (1/0).
# ─────────────────────────────────────────────────────────────────────────
"""

# ============================================================
# CONFIGURATION  –  adjust paths / flags as needed
# ============================================================

DRIVE_BASE = "/content/drive/MyDrive/Evaluating_Data_Poisoning_Attacks"
REGISTRY_PATH = f"{DRIVE_BASE}/models_registry.pkl"

# Set USE_MERGED_FILE = True to process the single all_merged file,
# or False to iterate over each attack-specific PKL file.
USE_MERGED_FILE = False

MERGED_PKL_PATH = f"{DRIVE_BASE}/all_merged_embeddings_with_labels.pkl"

# One PKL file per attack type
ATTACK_PKL_FILES = {
    "testset_in_trainingset":  f"{DRIVE_BASE}/testset_in_trainingset_with_labels.pkl",
    "label_flipping":          f"{DRIVE_BASE}/label_flipping_embeddings_with_labels.pkl",
    "word_order_perturbation": f"{DRIVE_BASE}/word_order_perturbation_embeddings_with_labels.pkl",
    "synonym_attack":          f"{DRIVE_BASE}/synonym_attack_embeddings_with_labels.pkl",
    "backdoor":                f"{DRIVE_BASE}/original+backdoor_embeddings_with_labels.pkl",
}

RANDOM_SEED = 50

# Order in which embeddings are evaluated (set to None to use all found)
EMBEDDING_TYPES = ["bow", "w2v", "glove_twitter", "glove_wiki", "fasttext", "sbert"]

# Human-readable class names (index = binary label: 0=benign, 1=jailbreak)
LABEL_NAMES = ["benign", "jailbreak"]

# Key used for the clean multiple-choice dataset inside every PKL
MC_CLEAN_KEY = "multiple_choice_clean"

# ============================================================
# IMPORTS
# ============================================================

import os
import pickle
import warnings
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# HELPER: label encoding
# ============================================================

def to_binary_label(label):
    """
    Map a single label value to binary integer.
        "jailbreak" / 1  ->  1
        "benign"    / 0  ->  0
    """
    if isinstance(label, (int, np.integer)):
        return int(label)
    label = str(label).strip().lower()
    if label == "jailbreak":
        return 1
    if label == "benign":
        return 0
    return int(label)


def encode_labels(y):
    """Convert any label collection (Series, list, ndarray) to a binary int ndarray."""
    if hasattr(y, "values"):       # pandas Series
        y = y.values
    return np.array([to_binary_label(lbl) for lbl in y], dtype=int)


# ============================================================
# evaluate_classifier
# ============================================================

def evaluate_classifier(clf, X, y,
                         dataset_name="", embedding_name="", split_name="",
                         verbose=True):
    """
    Evaluate a trained classifier on (X, y) and return a metrics dictionary.

    Parameters
    ----------
    clf            : fitted sklearn estimator
    X              : feature matrix (dense ndarray or sparse matrix)
    y              : true labels – strings or 0/1 integers
    dataset_name   : identifier for display only
    embedding_name : identifier for display only
    split_name     : "val", "test", or "mc_test" – for display only
    verbose        : print accuracy, F1, and classification report

    Returns
    -------
    dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_jailbreak, recall_jailbreak, f1_jailbreak,
        confusion_matrix (list), report_str (str)
    """
    y_enc  = encode_labels(y)
    y_pred = clf.predict(X)

    metrics = {
        "accuracy":            float(accuracy_score(y_enc, y_pred)),
        "precision_macro":     float(precision_score(y_enc, y_pred, average="macro",
                                                      zero_division=0)),
        "recall_macro":        float(recall_score(y_enc, y_pred,    average="macro",
                                                   zero_division=0)),
        "f1_macro":            float(f1_score(y_enc, y_pred,        average="macro",
                                              zero_division=0)),
        "precision_jailbreak": float(precision_score(y_enc, y_pred, pos_label=1,
                                                      average="binary", zero_division=0)),
        "recall_jailbreak":    float(recall_score(y_enc, y_pred,    pos_label=1,
                                                   average="binary", zero_division=0)),
        "f1_jailbreak":        float(f1_score(y_enc, y_pred,        pos_label=1,
                                              average="binary", zero_division=0)),
        "confusion_matrix":    confusion_matrix(y_enc, y_pred).tolist(),
        "report_str":          classification_report(y_enc, y_pred,
                                                     target_names=LABEL_NAMES,
                                                     zero_division=0),
    }

    if verbose:
        tag = f"[{dataset_name}] [{embedding_name}] [{split_name}]"
        print(f"\n{tag}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1 macro  : {metrics['f1_macro']:.4f}")
        print(f"  F1 jailbr.: {metrics['f1_jailbreak']:.4f}")
        print(metrics["report_str"])

    return metrics


# ============================================================
# register_model
# ============================================================

def register_model(
    registry,
    clf,
    model_name,
    dataset_name,
    embedding_name,
    best_params,
    val_metrics,
    test_metrics,
    mc_test_metrics=None,
    attack_type="",
    registry_path=REGISTRY_PATH,
):
    """
    Append a trained model entry to the registry list and persist to disk.

    Parameters
    ----------
    registry        : list   – current registry (modified in-place)
    clf             : fitted sklearn estimator
    model_name      : str    – e.g. "LogisticRegression"
    dataset_name    : str    – key from the embeddings dict
    embedding_name  : str    – e.g. "sbert"
    best_params     : dict   – hyperparameters selected on the validation set
    val_metrics     : dict   – output of evaluate_classifier on validation set
    test_metrics    : dict   – output of evaluate_classifier on test set
    mc_test_metrics : dict   – evaluate_classifier on MC jailbreak prompts, or None
    attack_type     : str    – e.g. "testset_in_trainingset"
    registry_path   : str    – path to models_registry.pkl

    Returns
    -------
    entry : dict – the registered entry
    """
    entry = {
        "model_name":      model_name,
        "attack_type":     attack_type,
        "dataset_name":    dataset_name,
        "embedding_name":  embedding_name,
        "best_params":     best_params,
        "val_metrics":     val_metrics,
        "test_metrics":    test_metrics,
        "mc_test_metrics": mc_test_metrics,
        "clf":             clf,
    }
    registry.append(entry)

    try:
        with open(registry_path, "wb") as fh:
            pickle.dump(registry, fh)
    except Exception as exc:
        print(f"[WARNING] Could not save registry to {registry_path}: {exc}")

    return entry


# ============================================================
# LOAD HELPERS
# ============================================================

def load_embeddings_pkl(pkl_path):
    """
    Load a *_with_labels.pkl file and return the embeddings dict.
    Returns an empty dict if the file is missing or cannot be read.
    """
    if not os.path.exists(pkl_path):
        print(f"[ERROR] File not found: {pkl_path}")
        return {}
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        print(f"Loaded  '{os.path.basename(pkl_path)}'  –  {len(data)} datasets")
        for name in data:
            print(f"  • {name}")
        return data
    except Exception as exc:
        print(f"[ERROR] Failed to load '{pkl_path}': {exc}")
        return {}


def load_registry(registry_path):
    """Load an existing model registry, or return an empty list."""
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "rb") as fh:
                reg = pickle.load(fh)
            print(f"Loaded existing registry: {len(reg)} entries")
            return reg
        except Exception as exc:
            print(f"[WARNING] Could not load registry ({exc}) – starting fresh.")
    return []


def _get_labels(bundle, split):
    """
    Retrieve labels for a split from a dataset bundle.
    Looks in bundle["y_<split>"] first, then bundle[split]["y"].
    Returns None if labels are not found.
    """
    y = bundle.get(f"y_{split}")
    if y is None:
        y = bundle.get(split, {}).get("y")
    return y


# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================

def build_candidate_models(random_state=RANDOM_SEED):
    """
    Return a list of (model_name, clf_instance, params_dict) to sweep.

    Models and hyperparameter values tried:
      - LogisticRegression : C ∈ {0.01, 0.1, 1, 10, 100}
      - LinearSVC          : C ∈ {0.01, 0.1, 1, 10, 100}
      - RandomForest       : (n_estimators, max_depth) ∈ {50/None, 100/None,
                               200/None, 100/10, 100/20}
    """
    candidates = []

    # ── Logistic Regression ──────────────────────────────────
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        params = {"C": C, "solver": "lbfgs", "max_iter": 2000}
        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=random_state,
        )
        candidates.append(("LogisticRegression", clf, params))

    # ── LinearSVC ────────────────────────────────────────────
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        params = {"C": C}
        clf = LinearSVC(
            C=C, class_weight="balanced",
            max_iter=5000, random_state=random_state,
        )
        candidates.append(("LinearSVC", clf, params))

    # ── Random Forest ────────────────────────────────────────
    for n_est, max_d in [(50, None), (100, None), (200, None), (100, 10), (100, 20)]:
        params = {"n_estimators": n_est, "max_depth": max_d}
        clf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_d,
            class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        candidates.append(("RandomForest", clf, params))

    return candidates


# ============================================================
# VALIDATION-SET HYPERPARAMETER SEARCH
# ============================================================

def tune_on_val(candidates, X_train, y_train, X_val, y_val):
    """
    Fit every candidate on the training set and score it on the validation
    set using F1-macro.  Returns the best (clf, model_name, params, val_f1,
    val_metrics) tuple.
    """
    y_train_enc = encode_labels(y_train)
    y_val_enc   = encode_labels(y_val)

    best_f1, best_result = -1.0, None

    print("  Hyperparameter search (sorted by val F1-macro):")
    print(f"  {'Model':<20} {'Params':<40} {'Val F1':>7}")
    print("  " + "-" * 72)

    rows = []
    for model_name, clf, params in candidates:
        try:
            clf.fit(X_train, y_train_enc)
            metrics = evaluate_classifier(
                clf, X_val, y_val_enc, verbose=False
            )
            f1 = metrics["f1_macro"]
            rows.append((f1, model_name, clf, params, metrics))
        except Exception as exc:
            print(f"  [SKIP] {model_name} {params}: {exc}")

    # Print all results sorted by val F1 descending
    rows.sort(key=lambda r: r[0], reverse=True)
    for f1, model_name, clf, params, metrics in rows:
        marker = " ◄ BEST" if f1 == rows[0][0] else ""
        print(f"  {model_name:<20} {str(params):<40} {f1:>7.4f}{marker}")

    if not rows:
        return None, None, None, -1.0, None

    best_f1, best_name, best_clf, best_params, best_metrics = rows[0]
    return best_clf, best_name, best_params, best_f1, best_metrics


# ============================================================
# MAIN LOOP: one PKL file
# ============================================================

def run_attack_pkl(all_embeddings, attack_type, models_registry):
    """
    For every (dataset, embedding) combination inside one PKL file:
      1. Tune hyperparameters on the VALIDATION SET.
      2. Evaluate the best model on the TEST SET.
      3. Evaluate on the MC jailbreak cluster and measure attack success rate.
      4. Call register_model to persist results.
    """
    emb_order = EMBEDDING_TYPES

    # Pre-load MC test embeddings (same for every trained model in this PKL)
    mc_test_splits = {}
    y_mc_test_enc  = np.array([])
    if MC_CLEAN_KEY in all_embeddings:
        mc_bundle = all_embeddings[MC_CLEAN_KEY]
        mc_test_splits = mc_bundle.get("test", {})
        y_mc_raw = _get_labels(mc_bundle, "test")
        if y_mc_raw is not None:
            y_mc_test_enc = encode_labels(y_mc_raw)

    # ── Iterate over datasets ─────────────────────────────────────────────
    for dataset_name, bundle in all_embeddings.items():

        print("\n" + "=" * 72)
        print(f"  DATASET : {dataset_name}")
        print(f"  ATTACK  : {attack_type}")
        print("=" * 72)

        # Labels
        y_train_raw = _get_labels(bundle, "train")
        y_val_raw   = _get_labels(bundle, "val")
        y_test_raw  = _get_labels(bundle, "test")

        if y_train_raw is None or y_val_raw is None or y_test_raw is None:
            print(f"  [SKIP] Labels not found in bundle for '{dataset_name}'.")
            print("         Ensure the PKL was saved with the _with_labels suffix.")
            continue

        y_train_enc = encode_labels(y_train_raw)
        y_val_enc   = encode_labels(y_val_raw)
        y_test_enc  = encode_labels(y_test_raw)

        print(f"  Train size : {len(y_train_enc)}  "
              f"(jailbreak={y_train_enc.sum()}, benign={(y_train_enc==0).sum()})")
        print(f"  Val size   : {len(y_val_enc)}")
        print(f"  Test size  : {len(y_test_enc)}")

        # Determine which embeddings are available
        avail_embs = [e for e in emb_order if e in bundle.get("train", {})]
        if not avail_embs:
            print(f"  [SKIP] No embeddings found under bundle['train'].")
            continue

        # ── Iterate over embedding types ──────────────────────────────────
        for emb_name in avail_embs:

            print(f"\n{'─'*72}")
            print(f"  Embedding : {emb_name}")
            print(f"{'─'*72}")

            X_train = bundle["train"][emb_name]
            X_val   = bundle["val"][emb_name]
            X_test  = bundle["test"][emb_name]

            shape_str = (str(X_train.shape) if hasattr(X_train, "shape")
                         else f"({len(X_train)},?)")
            print(f"  Feature matrix shape (train): {shape_str}")

            candidates = build_candidate_models()

            # ── Step 1: Hyperparameter search on VALIDATION SET ───────────
            print(f"\n  [1/3] Hyperparameter search on validation set …")
            best_clf, best_name, best_params, best_val_f1, _ = tune_on_val(
                candidates, X_train, y_train_enc, X_val, y_val_enc
            )

            if best_clf is None:
                print(f"  [SKIP] All candidates failed – skipping {emb_name}.")
                continue

            print(f"\n  >> Best model : {best_name}  |  params : {best_params}"
                  f"  |  val F1-macro : {best_val_f1:.4f}")

            # Full report on validation set
            val_metrics = evaluate_classifier(
                best_clf, X_val, y_val_enc,
                dataset_name=dataset_name,
                embedding_name=emb_name,
                split_name="val",
                verbose=True,
            )

            # ── Step 2: Evaluate on TEST SET ──────────────────────────────
            print(f"\n  [2/3] Evaluating best model on TEST SET …")
            test_metrics = evaluate_classifier(
                best_clf, X_test, y_test_enc,
                dataset_name=dataset_name,
                embedding_name=emb_name,
                split_name="test",
                verbose=True,
            )

            # ── Step 3: MC jailbreak detection check ──────────────────────
            mc_metrics = None

            if emb_name in mc_test_splits and len(y_mc_test_enc) > 0 \
                    and dataset_name != MC_CLEAN_KEY:

                print(f"\n  [3/3] Evaluating on multiple-choice jailbreak cluster …")
                X_mc = mc_test_splits[emb_name]

                mc_metrics = evaluate_classifier(
                    best_clf, X_mc, y_mc_test_enc,
                    dataset_name=dataset_name,
                    embedding_name=emb_name,
                    split_name="mc_test",
                    verbose=True,
                )

                # Attack success = fraction of MC prompts classified as BENIGN
                mc_preds           = best_clf.predict(X_mc)
                attack_success_rate = float((mc_preds == 0).mean())
                mc_metrics["attack_success_rate"] = attack_success_rate

                fooled  = int((mc_preds == 0).sum())
                total   = len(mc_preds)
                print(f"\n  ┌─ MC jailbreak trick detection ───────────────────────┐")
                print(f"  │  Prompts classified as BENIGN  : {fooled:3d} / {total}            │")
                print(f"  │  Attack success rate           : {attack_success_rate:.1%}              │")
                if attack_success_rate > 0.5:
                    print(f"  │  ⚠  Attack SUCCEEDED  – model tricked on MC prompts  │")
                else:
                    print(f"  │  ✓  Attack FAILED    – model mostly correct on MC     │")
                print(f"  └──────────────────────────────────────────────────────┘")

            else:
                print(f"\n  [3/3] MC embeddings not available for '{emb_name}'"
                      f" – skipping jailbreak trick check.")

            # ── Step 4: Register model ────────────────────────────────────
            register_model(
                registry=models_registry,
                clf=best_clf,
                model_name=best_name,
                dataset_name=dataset_name,
                embedding_name=emb_name,
                best_params=best_params,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                mc_test_metrics=mc_metrics,
                attack_type=attack_type,
                registry_path=REGISTRY_PATH,
            )

            print(f"\n  ✔ Registered: {best_name} | {dataset_name} | {emb_name}")
            print(f"    val  F1={val_metrics['f1_macro']:.4f}  "
                  f"acc={val_metrics['accuracy']:.4f}")
            print(f"    test F1={test_metrics['f1_macro']:.4f}  "
                  f"acc={test_metrics['accuracy']:.4f}")


# ============================================================
# RESULTS SUMMARY TABLE
# ============================================================

def print_summary(models_registry):
    """Print a compact summary table of all registered models."""
    if not models_registry:
        print("Registry is empty – nothing to summarise.")
        return

    sep = "─" * 110
    print(f"\n{sep}")
    print("RESULTS SUMMARY")
    print(sep)
    print(f"{'Attack':<26} {'Dataset':<28} {'Emb':<14} {'Model':<20} "
          f"{'ValF1':>6} {'TstF1':>6} {'TstAcc':>7} {'MC_succ':>8}")
    print(sep)

    for e in models_registry:
        mc_str = ""
        mc = e.get("mc_test_metrics")
        if mc and "attack_success_rate" in mc:
            mc_str = f"{mc['attack_success_rate']:.1%}"

        vm = e.get("val_metrics",  {})
        tm = e.get("test_metrics", {})

        print(
            f"{e.get('attack_type',''):<26} "
            f"{e.get('dataset_name',''):<28} "
            f"{e.get('embedding_name',''):<14} "
            f"{e.get('model_name',''):<20} "
            f"{vm.get('f1_macro', float('nan')):>6.4f} "
            f"{tm.get('f1_macro', float('nan')):>6.4f} "
            f"{tm.get('accuracy',  float('nan')):>7.4f} "
            f"{mc_str:>8}"
        )

    print(sep)


# ============================================================
# ENTRY POINT
# ============================================================

models_registry = load_registry(REGISTRY_PATH)

if USE_MERGED_FILE:
    # ── Single merged PKL ─────────────────────────────────────────────────
    print(f"\n{'#'*72}")
    print(f"# Using merged PKL: {os.path.basename(MERGED_PKL_PATH)}")
    print(f"{'#'*72}")
    all_emb = load_embeddings_pkl(MERGED_PKL_PATH)
    if all_emb:
        run_attack_pkl(all_emb, attack_type="all_merged",
                       models_registry=models_registry)
    else:
        print("[SKIP] Could not load merged PKL.")

else:
    # ── One PKL per attack ────────────────────────────────────────────────
    for attack_type, pkl_path in ATTACK_PKL_FILES.items():
        print(f"\n{'#'*72}")
        print(f"# Attack  : {attack_type}")
        print(f"# PKL     : {os.path.basename(pkl_path)}")
        print(f"{'#'*72}")
        all_emb = load_embeddings_pkl(pkl_path)
        if all_emb:
            run_attack_pkl(all_emb, attack_type=attack_type,
                           models_registry=models_registry)
        else:
            print(f"[SKIP] No embeddings loaded for '{attack_type}'.")

# ── Final output ──────────────────────────────────────────────────────────
print_summary(models_registry)
print(f"\nTotal models registered : {len(models_registry)}")
print(f"Registry saved to       : {REGISTRY_PATH}")
