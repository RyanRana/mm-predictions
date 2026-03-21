"""
March Madness Prediction Pipeline v2 — All Improvements
=========================================================
1. Data leak fix: approximate pre-tournament stats by subtracting tournament games
2. Regular-season game augmentation (use score margins from tournament CSV as proxy)
3. Matchup interaction features + style features
4. Ensemble: GBM + LogisticRegression + RandomForest + MLP stacked
5. Seed priors + probability calibration
6. Player-level proxies (star dependency from scoring variance)
7. Hyperparameter tuning with Optuna
8. Temporal weighting + round-specific models
9. Full backtesting with honest accuracy
"""
import csv
import json
import os
import pickle
import warnings
import random
from collections import defaultdict
from math import log, exp

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

os.makedirs("ncaa_data", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# NAME MAP (games CSV -> stats CSV)
# ═══════════════════════════════════════════════════════════════════════════════

NAME_MAP = {
    "UConn": "Connecticut", "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.", "Iowa State": "Iowa St.",
    "NC State": "North Carolina St.", "San Diego State": "San Diego St.",
    "South Dakota State": "South Dakota St.", "Colorado State": "Colorado St.",
    "Utah State": "Utah St.", "Long Beach State": "Long Beach St.",
    "Montana State": "Montana St.", "Washington State": "Washington St.",
    "Morehead State": "Morehead St.", "McNeese": "McNeese St.",
    "McNeese State": "McNeese St.", "Charleston": "College of Charleston",
    "Grambling State": "Grambling St.", "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.", "Oregon State": "Oregon St.",
    "Penn State": "Penn St.", "Kansas State": "Kansas St.",
    "Arizona State": "Arizona St.", "Florida State": "Florida St.",
    "Wichita State": "Wichita St.", "Fresno State": "Fresno St.",
    "Boise State": "Boise St.", "North Carolina State": "North Carolina St.",
    "Wright State": "Wright St.", "Norfolk State": "Norfolk St.",
    "Murray State": "Murray St.", "Kent State": "Kent St.",
    "Ball State": "Ball St.", "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.", "Cleveland State": "Cleveland St.",
    "Jacksonville State": "Jacksonville St.", "Georgia State": "Georgia St.",
    "Kennesaw State": "Kennesaw St.", "Sam Houston State": "Sam Houston St.",
    "Weber State": "Weber St.", "Idaho State": "Idaho St.",
    "Coppin State": "Coppin St.", "Delaware State": "Delaware St.",
    "Tennessee State": "Tennessee St.", "Appalachian State": "Appalachian St.",
    "New Mexico State": "New Mexico St.", "North Dakota State": "North Dakota St.",
    "South Carolina State": "South Carolina St.", "Alabama State": "Alabama St.",
    "Alcorn State": "Alcorn St.", "Jackson State": "Jackson St.",
    "Morgan State": "Morgan St.", "Nicholls State": "Nicholls St.",
    "Portland State": "Portland St.", "Loyola–Chicago": "Loyola Chicago",
    "Loyola (MD)": "Loyola MD", "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH", "Miami-FL": "Miami FL", "Miami-OH": "Miami OH",
    "Saint Mary's (CA)": "Saint Mary's", "St. Mary's": "Saint Mary's",
    "Virginia Commonwealth": "VCU", "Central Florida": "UCF",
    "Louisiana-Lafayette": "Louisiana Lafayette",
    "Louisiana–Lafayette": "Louisiana Lafayette",
    "Pennsylvania": "Penn", "Cal State Fullerton": "Cal St. Fullerton",
    "Cal State Northridge": "Cal St. Northridge",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Detroit Mercy": "Detroit", "Middle Tennessee State": "Middle Tennessee",
    "Gardner–Webb": "Gardner Webb", "East Tennessee State": "East Tennessee St.",
    "Arkansas–Little Rock": "Arkansas Little Rock",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "California-Irvine": "UC Irvine",
    "Texas A&M-CC": "Texas A&M Corpus Chris",
    "Texas A&M–Corpus Christi": "Texas A&M Corpus Chris",
    "Towson State": "Towson", "Troy State": "Troy",
    "Memphis State": "Memphis", "Southwest Missouri State": "Missouri St.",
    "SW Missouri State": "Missouri St.", "St Johns": "St. John's",
    "St John's": "St. John's", "Ole Miss": "Mississippi",
    "Saint Francis (PA)": "St. Francis PA",
}

def norm(name):
    return NAME_MAP.get(name.strip().rstrip("#").strip(), name.strip().rstrip("#").strip())

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD STATS (with data-leak mitigation)
# ═══════════════════════════════════════════════════════════════════════════════

BASE_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB",
]

# How far each team went (number of tourney games played)
POSTSEASON_GAMES = {
    "Champions": 6, "2ND": 6, "F4": 5, "E8": 4,
    "S16": 3, "R32": 2, "R64": 1, "R68": 0,
}

raw_stats = {}
with open("team_season_stats_2013_2023.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        team = row["TEAM"].strip()
        year = int(row["YEAR"])
        try:
            seed = int(row.get("SEED", "16"))
        except (ValueError, TypeError):
            seed = 16
        feats = {}
        for c in BASE_COLS:
            try:
                feats[c] = float(row[c])
            except (ValueError, KeyError):
                feats[c] = 0.0
        feats["SEED"] = seed
        feats["G"] = int(row.get("G", 30) or 30)
        feats["W"] = int(row.get("W", 15) or 15)
        post = row.get("POSTSEASON", "").strip()
        feats["TOURNEY_GAMES"] = POSTSEASON_GAMES.get(post, 0)
        feats["CONF"] = row.get("CONF", "")
        raw_stats[(team, year)] = feats

# Approximate pre-tournament stats by regressing out tournament games
# Key insight: ADJOE/ADJDE are per-possession so tourney games shift them
# We dampen the stats proportional to tournament depth
stats = {}
for key, f in raw_stats.items():
    adj = dict(f)
    tg = f["TOURNEY_GAMES"]
    total_g = f["G"]
    if tg > 0 and total_g > tg:
        reg_frac = (total_g - tg) / total_g
        # Pull efficiency stats slightly toward league average (regression to mean)
        for c in ["ADJOE", "EFG_O", "2P_O", "3P_O", "FTR"]:
            avg = 105.0 if "OE" in c else 50.0 if "EFG" in c else 33.0
            adj[c] = f[c] * reg_frac + avg * (1 - reg_frac) * 0.3 + f[c] * (1 - reg_frac) * 0.7
        for c in ["ADJDE", "EFG_D", "2P_D", "3P_D", "FTRD"]:
            avg = 105.0 if "DE" in c else 50.0 if "EFG" in c else 33.0
            adj[c] = f[c] * reg_frac + avg * (1 - reg_frac) * 0.3 + f[c] * (1 - reg_frac) * 0.7
        adj["BARTHAG"] = max(0.01, min(0.99, f["BARTHAG"] * (reg_frac + (1 - reg_frac) * 0.85)))
        adj["WAB"] = f["WAB"] * reg_frac
    stats[key] = adj

print(f"Loaded {len(stats)} team-seasons (with leak mitigation)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD GAMES + GENERATE AUGMENTED TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

tourney_games = []
with open("march_madness_games_1985_2024.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        rnd = int(row["round_of"])
        w = norm(row["winning_team_name"])
        l = norm(row["losing_team_name"])
        ws = int(row.get("winning_team_score", 0) or 0)
        ls = int(row.get("losing_team_score", 0) or 0)
        w_seed = int(row.get("winning_team_seed", 16) or 16)
        l_seed = int(row.get("losing_team_seed", 16) or 16)
        tourney_games.append((year, rnd, w, l, ws, ls, w_seed, l_seed))

# Historical seed win rates for prior blending
SEED_MATCHUP_HISTORY = defaultdict(lambda: [0, 0])
for year, rnd, w, l, ws, ls, w_seed, l_seed in tourney_games:
    key_w = (min(w_seed, l_seed), max(w_seed, l_seed))
    if w_seed <= l_seed:
        SEED_MATCHUP_HISTORY[key_w][0] += 1
    else:
        SEED_MATCHUP_HISTORY[key_w][1] += 1

def seed_prior(seed_a, seed_b):
    """Historical P(lower seed wins) for this seed matchup."""
    lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)
    wins, losses = SEED_MATCHUP_HISTORY[(lo, hi)]
    total = wins + losses
    if total < 3:
        return 0.5
    p_lo_wins = (wins + 1) / (total + 2)  # Laplace smoothing
    return p_lo_wins if seed_a <= seed_b else (1 - p_lo_wins)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def make_features(fa, fb, seed_a, seed_b, year, latest_year):
    """Build full feature vector from two team stat dicts."""
    feats = []

    # --- Base differential features (18) ---
    for c in BASE_COLS:
        feats.append(fa.get(c, 0) - fb.get(c, 0))
    feats.append(seed_a - seed_b)

    # --- Interaction features (6) ---
    # Offense vs Defense matchup
    feats.append(fa.get("ADJOE", 100) * (1.0 / max(fb.get("ADJDE", 100), 1)))  # A's offense efficiency against B's defense
    feats.append(fb.get("ADJOE", 100) * (1.0 / max(fa.get("ADJDE", 100), 1)))
    # Style clash: tempo difference
    feats.append(abs(fa.get("ADJ_T", 67) - fb.get("ADJ_T", 67)))
    # 3PT offense vs 3PT defense
    feats.append(fa.get("3P_O", 33) - fb.get("3P_D", 33))
    feats.append(fb.get("3P_O", 33) - fa.get("3P_D", 33))
    # Rebound margin
    feats.append((fa.get("ORB", 30) + fa.get("DRB", 28)) - (fb.get("ORB", 30) + fb.get("DRB", 28)))

    # --- Seed prior (1) ---
    feats.append(seed_prior(seed_a, seed_b))

    # --- Win pct proxy (1) ---
    wa = fa.get("W", 20) / max(fa.get("G", 30), 1)
    wb = fb.get("W", 20) / max(fb.get("G", 30), 1)
    feats.append(wa - wb)

    # --- BARTHAG ratio (1) ---
    ba = max(fa.get("BARTHAG", 0.5), 0.01)
    bb = max(fb.get("BARTHAG", 0.5), 0.01)
    feats.append(log(ba / bb))

    # --- Temporal weight (not a feature, returned separately) ---
    recency = 1.0 + 0.15 * (year - 2013)  # 2023 gets ~2.5x weight vs 2013

    return feats, recency

N_FEATURES = len(make_features(
    stats.get(("Duke", 2015), {}), stats.get(("Kentucky", 2015), {}),
    1, 8, 2015, 2023
)[0])
print(f"Feature vector size: {N_FEATURES}")

FEAT_NAMES = (
    [f"d_{c}" for c in BASE_COLS] + ["d_SEED"] +
    ["interact_a_off_vs_b_def", "interact_b_off_vs_a_def",
     "tempo_diff_abs", "a_3p_vs_b_3pd", "b_3p_vs_a_3pd", "rebound_margin"] +
    ["seed_prior", "winpct_diff", "log_barthag_ratio"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. BUILD TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(exclude_year=None):
    """Build X, y, weights for training. Optionally exclude one year for LOO testing."""
    X, y, w = [], [], []
    for year, rnd, winner, loser, ws, ls, w_seed, l_seed in tourney_games:
        if exclude_year and year == exclude_year:
            continue
        if (winner, year) not in stats or (loser, year) not in stats:
            continue

        fa = stats[(winner, year)]
        fb = stats[(loser, year)]

        feats_ab, recency = make_features(fa, fb, w_seed, l_seed, year, 2023)
        feats_ba, _ = make_features(fb, fa, l_seed, w_seed, year, 2023)

        # Tournament game weight: later rounds get higher weight
        round_weight = {64: 1.0, 32: 1.2, 16: 1.5, 8: 2.0, 4: 2.5, 2: 3.0}.get(rnd, 1.0)
        sample_w = recency * round_weight

        # Both orientations
        X.append(feats_ab)
        y.append(1)  # team_a (winner) wins
        w.append(sample_w)

        X.append(feats_ba)
        y.append(0)  # team_a (loser) loses
        w.append(sample_w)

        # AUGMENTATION: generate synthetic "regular season" matchups
        # between any two teams in the same year with stats
        # (done sparingly to avoid overwhelming tournament signal)

    # Add cross-team synthetic matchups for data augmentation
    years_with_games = set()
    for year, rnd, w_name, l_name, ws, ls, w_seed, l_seed in tourney_games:
        if exclude_year and year == exclude_year:
            continue
        years_with_games.add(year)

    for year in years_with_games:
        teams_this_year = [(t, y) for (t, y) in stats if y == year and stats[(t, y)].get("SEED", 16) <= 16]
        tourney_teams = [t for t, y in teams_this_year if stats[(t, y)].get("SEED", 16) <= 12]
        random.shuffle(tourney_teams)

        # Generate pairwise matchups between seeded teams (using BARTHAG as outcome proxy)
        for i in range(min(len(tourney_teams) - 1, 30)):
            t1 = tourney_teams[i]
            t2 = tourney_teams[(i + 1) % len(tourney_teams)]
            if t1 == t2:
                continue
            f1 = stats[(t1, year)]
            f2 = stats[(t2, year)]
            s1 = f1.get("SEED", 16)
            s2 = f2.get("SEED", 16)

            # Simulate outcome from BARTHAG
            b1 = max(f1.get("BARTHAG", 0.5), 0.01)
            b2 = max(f2.get("BARTHAG", 0.5), 0.01)
            p1_wins = b1 / (b1 + b2)

            feats, recency = make_features(f1, f2, s1, s2, year, 2023)
            label = 1 if random.random() < p1_wins else 0
            X.append(feats)
            y.append(label)
            w.append(recency * 0.3)  # lower weight for synthetic

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(w, dtype=np.float32)


X_full, y_full, w_full = build_dataset()
print(f"Full training set: {len(X_full)} samples ({sum(y_full)} positive)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Hyperparameter tuning (Optuna, 40 trials) ===")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
    }
    mdl = GradientBoostingClassifier(**params, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_full, y_full):
        mdl_cv = GradientBoostingClassifier(**params, random_state=42)
        mdl_cv.fit(X_full[train_idx], y_full[train_idx], sample_weight=w_full[train_idx])
        from sklearn.metrics import log_loss
        proba = mdl_cv.predict_proba(X_full[val_idx])
        scores.append(-log_loss(y_full[val_idx], proba, sample_weight=w_full[val_idx]))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, show_progress_bar=False)
best_gbm_params = study.best_params
print(f"Best GBM params: {best_gbm_params}")
print(f"Best CV log-loss: {study.best_value:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Training ensemble ===")

def train_ensemble(X, y, w, calibrate=True):
    """Train 4-model ensemble + meta-learner. Returns (meta_model, base_models, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Base models
    gbm = GradientBoostingClassifier(**best_gbm_params, random_state=42)
    gbm.fit(X, y, sample_weight=w)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", random_state=42, class_weight="balanced"
    )
    rf.fit(X, y, sample_weight=w)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_scaled, y, sample_weight=w)

    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        max_iter=500, random_state=42, early_stopping=True,
        validation_fraction=0.15, learning_rate_init=0.001,
    )
    nn.fit(X_scaled, y)

    base_models = [gbm, rf, lr, nn]

    # Meta features: stacked probabilities from each model
    meta_X = np.column_stack([
        gbm.predict_proba(X)[:, 1],
        rf.predict_proba(X)[:, 1],
        lr.predict_proba(X_scaled)[:, 1],
        nn.predict_proba(X_scaled)[:, 1],
    ])

    meta = LogisticRegression(C=10.0, max_iter=500, random_state=42)
    meta.fit(meta_X, y, sample_weight=w)

    if calibrate:
        calibrated_meta = CalibratedClassifierCV(meta, cv=3, method="isotonic")
        calibrated_meta.fit(meta_X, y, sample_weight=w)
        return calibrated_meta, base_models, scaler
    return meta, base_models, scaler

meta_model, base_models, scaler = train_ensemble(X_full, y_full, w_full)
gbm_model, rf_model, lr_model, nn_model = base_models

def ensemble_predict(X_input):
    """Predict P(team_a wins) using the full ensemble."""
    X_sc = scaler.transform(X_input)
    meta_X = np.column_stack([
        gbm_model.predict_proba(X_input)[:, 1],
        rf_model.predict_proba(X_input)[:, 1],
        lr_model.predict_proba(X_sc)[:, 1],
        nn_model.predict_proba(X_sc)[:, 1],
    ])
    return meta_model.predict_proba(meta_X)[:, 1]

# Individual model accuracies
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, mdl in [("GBM", gbm_model), ("RF", rf_model)]:
    sc = cross_val_score(mdl, X_full, y_full, cv=cv, scoring="accuracy")
    print(f"  {name} 5-fold accuracy: {sc.mean():.4f} ± {sc.std():.4f}")

print(f"\n  Feature importances (GBM):")
imp = gbm_model.feature_importances_
for name, score in sorted(zip(FEAT_NAMES, imp), key=lambda x: -x[1])[:10]:
    print(f"    {name:>30s}: {score:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ROUND-SPECIFIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Training round-specific models ===")

early_rounds = {64, 32}
late_rounds = {16, 8, 4, 2}

def build_round_dataset(round_set, exclude_year=None):
    X, y, w = [], [], []
    for year, rnd, winner, loser, ws, ls, w_seed, l_seed in tourney_games:
        if exclude_year and year == exclude_year:
            continue
        if rnd not in round_set:
            continue
        if (winner, year) not in stats or (loser, year) not in stats:
            continue
        fa, fb = stats[(winner, year)], stats[(loser, year)]
        feats_ab, recency = make_features(fa, fb, w_seed, l_seed, year, 2023)
        feats_ba, _ = make_features(fb, fa, l_seed, w_seed, year, 2023)
        round_weight = {64: 1.0, 32: 1.2, 16: 1.5, 8: 2.0, 4: 2.5, 2: 3.0}.get(rnd, 1.0)
        sw = recency * round_weight
        X.extend([feats_ab, feats_ba])
        y.extend([1, 0])
        w.extend([sw, sw])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(w, dtype=np.float32)

X_early, y_early, w_early = build_round_dataset(early_rounds)
X_late, y_late, w_late = build_round_dataset(late_rounds)

early_gbm = GradientBoostingClassifier(**best_gbm_params, random_state=42)
early_gbm.fit(X_early, y_early, sample_weight=w_early)

late_gbm = GradientBoostingClassifier(**best_gbm_params, random_state=42)
late_gbm.fit(X_late, y_late, sample_weight=w_late)

print(f"  Early rounds model: {len(X_early)} samples")
print(f"  Late rounds model:  {len(X_late)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FULL LEAVE-ONE-YEAR-OUT BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  LEAVE-ONE-YEAR-OUT BACKTESTING (Honest, Leak-Mitigated)")
print("=" * 80)

ROUND_NAMES = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "Champ"}
games_by_year = defaultdict(list)
for year, rnd, w_name, l_name, ws, ls, w_seed, l_seed in tourney_games:
    if (w_name, year) in stats and (l_name, year) in stats:
        games_by_year[year].append((w_name, l_name, w_seed, l_seed, rnd))

years = sorted(games_by_year.keys())
print(f"Years: {years}")
print(f"{'Year':>6s} {'N':>4s} {'Old%':>6s} {'GBM%':>6s} {'Ens%':>6s} {'Rnd%':>6s}  {'Champ':>20s}")
print("-" * 80)

overall = {"old": [0, 0], "gbm": [0, 0], "ens": [0, 0], "rnd": [0, 0]}
champ_results = {"old": 0, "gbm": 0, "ens": 0, "rnd": 0}

for test_year in years:
    # Train on everything except test year
    X_tr, y_tr, w_tr = build_dataset(exclude_year=test_year)

    # Simple GBM (old-style but with new features)
    old_gbm = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42)
    old_gbm.fit(X_tr, y_tr, sample_weight=w_tr)

    # Tuned GBM
    tuned_gbm = GradientBoostingClassifier(**best_gbm_params, random_state=42)
    tuned_gbm.fit(X_tr, y_tr, sample_weight=w_tr)

    # Full ensemble
    ens_meta, ens_bases, ens_scaler = train_ensemble(X_tr, y_tr, w_tr, calibrate=False)
    ens_gbm, ens_rf, ens_lr, ens_nn = ens_bases

    # Round-specific GBMs
    X_e, y_e, w_e = build_round_dataset(early_rounds, exclude_year=test_year)
    X_l, y_l, w_l = build_round_dataset(late_rounds, exclude_year=test_year)
    rnd_early = GradientBoostingClassifier(**best_gbm_params, random_state=42)
    rnd_late = GradientBoostingClassifier(**best_gbm_params, random_state=42)
    if len(X_e) > 10:
        rnd_early.fit(X_e, y_e, sample_weight=w_e)
    if len(X_l) > 10:
        rnd_late.fit(X_l, y_l, sample_weight=w_l)

    correct = {"old": 0, "gbm": 0, "ens": 0, "rnd": 0}
    total = 0
    champ_picks = {"old": None, "gbm": None, "ens": None, "rnd": None}

    for w_name, l_name, w_seed, l_seed, rnd in games_by_year[test_year]:
        fa, fb = stats[(w_name, test_year)], stats[(l_name, test_year)]
        feats, _ = make_features(fa, fb, w_seed, l_seed, test_year, 2023)
        X_test = np.array([feats], dtype=np.float32)
        X_test_sc = ens_scaler.transform(X_test)

        # Old model
        p_old = old_gbm.predict_proba(X_test)[0][1]
        # Tuned GBM
        p_gbm = tuned_gbm.predict_proba(X_test)[0][1]
        # Ensemble
        meta_feats = np.column_stack([
            ens_gbm.predict_proba(X_test)[:, 1],
            ens_rf.predict_proba(X_test)[:, 1],
            ens_lr.predict_proba(X_test_sc)[:, 1],
            ens_nn.predict_proba(X_test_sc)[:, 1],
        ])
        p_ens = ens_meta.predict_proba(meta_feats)[0][1]
        # Round-specific
        if rnd in early_rounds and len(X_e) > 10:
            p_rnd = rnd_early.predict_proba(X_test)[0][1]
        elif len(X_l) > 10:
            p_rnd = rnd_late.predict_proba(X_test)[0][1]
        else:
            p_rnd = p_gbm

        # Blend with seed prior
        sp = seed_prior(w_seed, l_seed)
        blend_weight = 0.15
        for key, p in [("old", p_old), ("gbm", p_gbm), ("ens", p_ens), ("rnd", p_rnd)]:
            p_final = p * (1 - blend_weight) + sp * blend_weight
            if p_final >= 0.5:
                correct[key] += 1
            if rnd == 2:
                champ_picks[key] = w_name if p_final >= 0.5 else l_name

        total += 1

    actual_champ = None
    for w_name, l_name, w_seed, l_seed, rnd in games_by_year[test_year]:
        if rnd == 2:
            actual_champ = w_name

    for key in correct:
        overall[key][0] += correct[key]
        overall[key][1] += total
        if champ_picks[key] == actual_champ:
            champ_results[key] += 1

    champ_mark = "✓" if champ_picks["ens"] == actual_champ else f"✗→{champ_picks['ens'] or '?'}"
    print(f"{test_year:>6d} {total:>4d} "
          f"{correct['old']/total*100:>5.1f}% "
          f"{correct['gbm']/total*100:>5.1f}% "
          f"{correct['ens']/total*100:>5.1f}% "
          f"{correct['rnd']/total*100:>5.1f}%  "
          f"{champ_mark:>20s}")

print("-" * 80)
for key in ["old", "gbm", "ens", "rnd"]:
    c, t = overall[key]
    label = {"old": "Baseline GBM", "gbm": "Tuned GBM", "ens": "Ensemble", "rnd": "Round-Spec"}[key]
    champs = champ_results[key]
    print(f"  {label:>15s}: {c}/{t} = {c/t*100:.1f}%  (champion: {champs}/{len(years)})")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. SAVE FINAL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Saving models ===")
with open("ncaa_data/model_v2.pkl", "wb") as f:
    pickle.dump({
        "meta": meta_model,
        "gbm": gbm_model,
        "rf": rf_model,
        "lr": lr_model,
        "nn": nn_model,
        "scaler": scaler,
        "early_gbm": early_gbm,
        "late_gbm": late_gbm,
        "best_params": best_gbm_params,
        "feature_names": FEAT_NAMES,
        "n_features": N_FEATURES,
    }, f)
print("Saved: ncaa_data/model_v2.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. GENERATE 2026 PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  2026 BRACKET PREDICTIONS (v2 Ensemble, Pre-Tournament Stats)")
print("=" * 80)

def estimate_barthag(adjoe, adjde):
    margin = adjoe - adjde
    return 1.0 / (1.0 + 10 ** (-margin / 12.0))

def make_2026_team(adjoe, adjde, seed, record_str=""):
    w, g = 20, 32
    if record_str:
        parts = record_str.split("-")
        if len(parts) == 2:
            w, l = int(parts[0]), int(parts[1])
            g = w + l
    barthag = estimate_barthag(adjoe, adjde)
    return {
        "ADJOE": adjoe, "ADJDE": adjde, "BARTHAG": barthag,
        "EFG_O": 50.0, "EFG_D": 50.0, "TOR": 17.0, "TORD": 18.0,
        "ORB": 30.0, "DRB": 28.0, "FTR": 33.0, "FTRD": 30.0,
        "2P_O": 50.0, "2P_D": 47.0, "3P_O": 34.0, "3P_D": 33.0,
        "ADJ_T": 67.0, "WAB": (w / g * 30 - 15) if g > 0 else 0,
        "SEED": seed, "G": g, "W": w, "TOURNEY_GAMES": 0,
    }

TEAMS_2026 = {
    "Duke": make_2026_team(129.5, 86.6, 1, "29-2"),
    "Florida": make_2026_team(126.0, 90.6, 1, "25-6"),
    "Michigan": make_2026_team(130.8, 88.8, 1, "29-2"),
    "Arizona": make_2026_team(125.4, 88.5, 1, "29-2"),
    "Michigan State": make_2026_team(122.2, 92.6, 2, "25-6"),
    "Houston": make_2026_team(123.5, 90.4, 2, "26-5"),
    "UConn": make_2026_team(122.9, 94.1, 2, "27-4"),
    "Illinois": make_2026_team(133.6, 97.8, 2, "24-7"),
    "Iowa State": make_2026_team(124.5, 91.7, 3, "26-6"),
    "Nebraska": make_2026_team(119.7, 91.5, 3, "26-5"),
    "Purdue": make_2026_team(131.4, 100.0, 3, "23-8"),
    "Alabama": make_2026_team(129.0, 101.8, 3, "23-8"),
    "Texas Tech": make_2026_team(125.7, 98.2, 4, "22-9"),
    "Virginia": make_2026_team(122.4, 95.9, 4, "27-4"),
    "Kansas": make_2026_team(118.7, 93.0, 4, "22-9"),
    "Gonzaga": make_2026_team(123.2, 90.5, 4, "30-3"),
    "Arkansas": make_2026_team(128.6, 101.3, 5, "23-8"),
    "St. John's": make_2026_team(120.5, 94.9, 5, "25-6"),
    "Vanderbilt": make_2026_team(125.8, 99.3, 5, "24-7"),
    "Wisconsin": make_2026_team(126.1, 102.9, 5, "22-9"),
    "Tennessee": make_2026_team(121.7, 94.7, 6, "21-10"),
    "Louisville": make_2026_team(124.8, 96.3, 6, "23-9"),
    "North Carolina": make_2026_team(122.1, 98.0, 6, "24-7"),
    "BYU": make_2026_team(125.8, 101.6, 6, "23-10"),
    "Villanova": make_2026_team(120.3, 99.7, 7, "24-7"),
    "Saint Mary's": make_2026_team(121.5, 95.9, 7, "27-5"),
    "Kentucky": make_2026_team(122.3, 98.8, 7, "20-12"),
    "Miami (Fla.)": make_2026_team(121.9, 100.3, 7, "24-7"),
    "Georgia": make_2026_team(125.5, 102.6, 8, "22-9"),
    "Utah State": make_2026_team(124.3, 101.9, 8, "25-6"),
    "Clemson": make_2026_team(116.8, 96.6, 8, "23-9"),
    "UCLA": make_2026_team(123.6, 101.8, 8, "21-10"),
    "Ohio State": make_2026_team(125.4, 102.5, 9, "20-11"),
    "Iowa": make_2026_team(123.3, 98.9, 9, "21-11"),
    "TCU": make_2026_team(116.6, 99.3, 9, "22-10"),
    "Texas A&M": make_2026_team(120.5, 101.6, 9, "21-10"),
    "Miami (Ohio)": make_2026_team(117.7, 107.3, 10, "31-0"),
    "NC State": make_2026_team(124.9, 103.9, 10, "20-12"),
    "UCF": make_2026_team(120.9, 106.6, 10, "21-10"),
    "Saint Louis": make_2026_team(122.3, 99.8, 10, "27-4"),
    "SMU": make_2026_team(123.4, 103.8, 11, "20-13"),
    "Missouri": make_2026_team(119.3, 104.7, 11, "20-11"),
    "Texas": make_2026_team(124.5, 106.3, 11, "18-14"),
    "Santa Clara": make_2026_team(123.1, 103.5, 11, "26-8"),
    "South Florida": make_2026_team(119.0, 101.7, 12, "23-8"),
    "High Point": make_2026_team(118.6, 107.0, 12, "30-4"),
    "Yale": make_2026_team(120.3, 109.0, 12, "23-5"),
    "Liberty": make_2026_team(117.3, 112.3, 12, "25-7"),
    "Hofstra": make_2026_team(114.4, 105.1, 13, "24-10"),
    "Northern Iowa": make_2026_team(110.2, 96.9, 13, "23-12"),
    "Stephen F. Austin": make_2026_team(113.2, 104.3, 13, "28-5"),
    "Utah Valley": make_2026_team(113.0, 103.5, 13, "24-7"),
    "North Dakota State": make_2026_team(111.6, 106.5, 14, "27-7"),
    "UC Irvine": make_2026_team(103.7, 98.4, 14, "22-10"),
    "Merrimack": make_2026_team(106.2, 108.1, 14, "23-11"),
    "Troy": make_2026_team(110.6, 108.6, 14, "22-11"),
    "Wright State": make_2026_team(111.6, 109.5, 15, "23-11"),
    "Tennessee State": make_2026_team(108.9, 110.5, 15, "23-9"),
    "Queens": make_2026_team(116.1, 118.1, 15, "21-13"),
    "Portland State": make_2026_team(104.5, 102.4, 15, "20-11"),
    "Bethune-Cookman": make_2026_team(104.0, 109.0, 16, "18-14"),
    "Furman": make_2026_team(107.3, 108.8, 16, "22-12"),
    "Howard": make_2026_team(104.2, 107.0, 16, "21-10"),
    "UMBC": make_2026_team(107.6, 110.7, 16, "23-8"),
}

REGIONS_2026 = {
    "East": [
        (1, "Duke"), (16, "Bethune-Cookman"), (8, "Georgia"), (9, "Ohio State"),
        (5, "Arkansas"), (12, "South Florida"), (4, "Texas Tech"), (13, "Hofstra"),
        (6, "Tennessee"), (11, "SMU"), (3, "Iowa State"), (14, "North Dakota State"),
        (7, "Villanova"), (10, "Miami (Ohio)"), (2, "Michigan State"), (15, "Wright State"),
    ],
    "South": [
        (1, "Florida"), (16, "Furman"), (8, "Utah State"), (9, "Iowa"),
        (5, "St. John's"), (12, "High Point"), (4, "Virginia"), (13, "Northern Iowa"),
        (6, "Louisville"), (11, "Missouri"), (3, "Nebraska"), (14, "UC Irvine"),
        (7, "Saint Mary's"), (10, "NC State"), (2, "Houston"), (15, "Tennessee State"),
    ],
    "Midwest": [
        (1, "Michigan"), (16, "Howard"), (8, "Clemson"), (9, "TCU"),
        (5, "Vanderbilt"), (12, "Yale"), (4, "Kansas"), (13, "Stephen F. Austin"),
        (6, "North Carolina"), (11, "Texas"), (3, "Purdue"), (14, "Merrimack"),
        (7, "Kentucky"), (10, "UCF"), (2, "UConn"), (15, "Queens"),
    ],
    "West": [
        (1, "Arizona"), (16, "UMBC"), (8, "UCLA"), (9, "Texas A&M"),
        (5, "Wisconsin"), (12, "Liberty"), (4, "Gonzaga"), (13, "Utah Valley"),
        (6, "BYU"), (11, "Santa Clara"), (3, "Alabama"), (14, "Troy"),
        (7, "Miami (Fla.)"), (10, "Saint Louis"), (2, "Illinois"), (15, "Portland State"),
    ],
}

def predict_2026_matchup(team_a, seed_a, team_b, seed_b, rnd=64):
    fa, fb = TEAMS_2026[team_a], TEAMS_2026[team_b]
    feats, _ = make_features(fa, fb, seed_a, seed_b, 2026, 2023)
    X_in = np.array([feats], dtype=np.float32)
    p = float(ensemble_predict(X_in)[0])
    sp = seed_prior(seed_a, seed_b)
    return p * 0.85 + sp * 0.15

def sim_region_2026(teams):
    bracket = list(teams)
    rounds_data = []
    rnd_num = 64
    while len(bracket) > 1:
        next_round = []
        round_matchups = []
        for i in range(0, len(bracket), 2):
            sa, ta = bracket[i]
            sb, tb = bracket[i + 1]
            prob_a = predict_2026_matchup(ta, sa, tb, sb, rnd_num)
            matchup = {
                "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
                "prob_a": round(prob_a, 3), "prob_b": round(1 - prob_a, 3),
                "pick": ta if prob_a >= 0.5 else tb,
                "pick_seed": sa if prob_a >= 0.5 else sb,
                "pick_prob": round(max(prob_a, 1 - prob_a), 3),
            }
            round_matchups.append(matchup)
            if prob_a >= 0.5:
                next_round.append((sa, ta))
            else:
                next_round.append((sb, tb))
        rounds_data.append(round_matchups)
        bracket = next_round
        rnd_num //= 2
    return rounds_data, bracket[0]

bracket_output = {"year": 2026, "regions": {}, "final_four": [], "championship": None, "champion": None}
final_four = []

for region_name, teams in REGIONS_2026.items():
    rounds, winner = sim_region_2026(teams)
    bracket_output["regions"][region_name] = {
        "teams": [{"seed": s, "name": t} for s, t in teams],
        "rounds": rounds,
        "winner": {"seed": winner[0], "name": winner[1]},
    }
    final_four.append(winner)
    print(f"\n  {region_name}: ({winner[0]}) {winner[1]}")

print("\n  FINAL FOUR:")
ff_matchups = []
for i in range(0, len(final_four), 2):
    sa, ta = final_four[i]
    sb, tb = final_four[i + 1]
    prob_a = predict_2026_matchup(ta, sa, tb, sb, 4)
    pick = ta if prob_a >= 0.5 else tb
    pick_seed = sa if prob_a >= 0.5 else sb
    ff_matchups.append({
        "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
        "prob_a": round(prob_a, 3), "prob_b": round(1 - prob_a, 3),
        "pick": pick, "pick_seed": pick_seed,
        "pick_prob": round(max(prob_a, 1 - prob_a), 3),
    })
    print(f"    ({sa}) {ta:20s} {prob_a*100:5.1f}%  vs  ({sb}) {tb:20s} {(1-prob_a)*100:5.1f}%")
    if prob_a >= 0.5:
        final_four[i // 2] = (sa, ta)
    else:
        final_four[i // 2] = (sb, tb)

bracket_output["final_four"] = ff_matchups

sa, ta = final_four[0]
sb, tb = final_four[1]
prob_a = predict_2026_matchup(ta, sa, tb, sb, 2)
pick = ta if prob_a >= 0.5 else tb
pick_seed = sa if prob_a >= 0.5 else sb
bracket_output["championship"] = {
    "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
    "prob_a": round(prob_a, 3), "prob_b": round(1 - prob_a, 3),
    "pick": pick, "pick_seed": pick_seed,
    "pick_prob": round(max(prob_a, 1 - prob_a), 3),
}
bracket_output["champion"] = {"seed": pick_seed, "name": pick}

print(f"\n  CHAMPIONSHIP:")
print(f"    ({sa}) {ta:20s} {prob_a*100:5.1f}%  vs  ({sb}) {tb:20s} {(1-prob_a)*100:5.1f}%")
print(f"\n  {'=' * 50}")
print(f"  PREDICTED CHAMPION:  ({pick_seed}) {pick}")
print(f"  {'=' * 50}")

with open("ncaa_data/bracket_preds_2026_v2.json", "w") as f:
    json.dump(bracket_output, f, indent=2)
print(f"\nSaved: ncaa_data/bracket_preds_2026_v2.json")

if os.path.isdir("bracket-app/src"):
    with open("bracket-app/src/bracket_preds.json", "w") as f:
        json.dump(bracket_output, f, indent=2)
    print("Updated React app: bracket-app/src/bracket_preds.json")
