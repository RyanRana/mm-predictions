"""
Fully honest leave-one-year-out backtesting.
Fixes every leak from train_v2.py:
  1. NO postseason-based stat adjustment for the test year (uses raw end-of-season stats)
  2. Seed priors exclude test year
  3. No Optuna — fixed hyperparams to avoid HP-selection leak
  4. Stacking uses out-of-fold predictions (not in-sample)
Outputs: ncaa_data/backtest_results.json  (every game, every year)
"""
import csv, json, os, random, warnings
from collections import defaultdict
from math import log

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ─── NAME MAP ───────────────────────────────────────────────────────────────

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

# ─── LOAD RAW STATS (no postseason adjustment) ──────────────────────────────

BASE_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB",
]

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
        raw_stats[(team, year)] = feats

print(f"Loaded {len(raw_stats)} team-seasons (RAW, no postseason adjustment)")

# ─── LOAD TOURNEY GAMES ─────────────────────────────────────────────────────

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

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────

def build_seed_prior(exclude_year):
    """Build historical seed matchup win rates, EXCLUDING one year."""
    hist = defaultdict(lambda: [0, 0])
    for yr, rnd, w, l, ws, ls, w_seed, l_seed in tourney_games:
        if yr == exclude_year:
            continue
        key = (min(w_seed, l_seed), max(w_seed, l_seed))
        if w_seed <= l_seed:
            hist[key][0] += 1
        else:
            hist[key][1] += 1

    def _prior(seed_a, seed_b):
        lo, hi = min(seed_a, seed_b), max(seed_a, seed_b)
        wins, losses = hist[(lo, hi)]
        total = wins + losses
        if total < 3:
            return 0.5
        p_lo_wins = (wins + 1) / (total + 2)
        return p_lo_wins if seed_a <= seed_b else (1 - p_lo_wins)
    return _prior


def make_features(fa, fb, seed_a, seed_b, seed_prior_fn):
    """Build feature vector. No year-dependent logic — pure stats + seeds."""
    feats = []
    for c in BASE_COLS:
        feats.append(fa.get(c, 0) - fb.get(c, 0))
    feats.append(seed_a - seed_b)

    feats.append(fa.get("ADJOE", 100) / max(fb.get("ADJDE", 100), 1))
    feats.append(fb.get("ADJOE", 100) / max(fa.get("ADJDE", 100), 1))
    feats.append(abs(fa.get("ADJ_T", 67) - fb.get("ADJ_T", 67)))
    feats.append(fa.get("3P_O", 33) - fb.get("3P_D", 33))
    feats.append(fb.get("3P_O", 33) - fa.get("3P_D", 33))
    feats.append((fa.get("ORB", 30) + fa.get("DRB", 28)) - (fb.get("ORB", 30) + fb.get("DRB", 28)))

    feats.append(seed_prior_fn(seed_a, seed_b))

    wa = fa.get("W", 20) / max(fa.get("G", 30), 1)
    wb = fb.get("W", 20) / max(fb.get("G", 30), 1)
    feats.append(wa - wb)

    ba = max(fa.get("BARTHAG", 0.5), 0.01)
    bb = max(fb.get("BARTHAG", 0.5), 0.01)
    feats.append(log(ba / bb))

    return feats


FEAT_NAMES = (
    [f"d_{c}" for c in BASE_COLS] + ["d_SEED"] +
    ["off_vs_def_a", "off_vs_def_b", "tempo_gap", "a3p_vs_b3pd", "b3p_vs_a3pd", "reb_margin"] +
    ["seed_prior", "winpct_diff", "log_barthag_ratio"]
)
N_FEAT = len(FEAT_NAMES)

# ─── FIXED HYPERPARAMS (no Optuna leak) ─────────────────────────────────────

GBM_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "max_features": 0.8, "min_samples_leaf": 10,
    "random_state": 42,
}

# ─── BUILD DATASET ──────────────────────────────────────────────────────────

def build_train_data(exclude_year, seed_prior_fn):
    """Build training set from all years except exclude_year. Uses raw stats."""
    X, y, w = [], [], []
    for yr, rnd, winner, loser, ws, ls, w_seed, l_seed in tourney_games:
        if yr == exclude_year:
            continue
        if (winner, yr) not in raw_stats or (loser, yr) not in raw_stats:
            continue
        fa, fb = raw_stats[(winner, yr)], raw_stats[(loser, yr)]
        feats_ab = make_features(fa, fb, w_seed, l_seed, seed_prior_fn)
        feats_ba = make_features(fb, fa, l_seed, w_seed, seed_prior_fn)
        recency = 1.0 + 0.15 * (yr - 2013)
        rw = {64: 1.0, 32: 1.2, 16: 1.5, 8: 2.0, 4: 2.5, 2: 3.0}.get(rnd, 1.0)
        sw = recency * rw
        X.append(feats_ab); y.append(1); w.append(sw)
        X.append(feats_ba); y.append(0); w.append(sw)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(w, dtype=np.float32)


# ─── HONEST LOO BACKTESTING ─────────────────────────────────────────────────

ROUND_LABELS = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "Final"}

games_by_year = defaultdict(list)
for yr, rnd, w, l, ws, ls, w_seed, l_seed in tourney_games:
    if (w, yr) in raw_stats and (l, yr) in raw_stats:
        games_by_year[yr].append((rnd, w, l, w_seed, l_seed, ws, ls))

years = sorted(games_by_year.keys())
print(f"Backtest years: {years}")
print(f"{'Year':>6s} {'N':>4s} {'GBM%':>7s} {'Ens%':>7s} {'Seed%':>7s}  Champ")
print("-" * 70)

all_results = {}
overall_correct = {"gbm": 0, "ens": 0, "seed": 0}
overall_total = 0
round_correct = defaultdict(lambda: {"gbm": 0, "ens": 0, "seed": 0, "total": 0})

for test_year in years:
    # Build seed prior excluding test year
    sp_fn = build_seed_prior(test_year)

    # Build training data excluding test year
    X_tr, y_tr, w_tr = build_train_data(test_year, sp_fn)
    if len(X_tr) < 20:
        continue

    # --- Model 1: GBM ---
    gbm = GradientBoostingClassifier(**GBM_PARAMS)
    gbm.fit(X_tr, y_tr, sample_weight=w_tr)

    # --- Model 2: Ensemble (GBM + RF + LR) with out-of-fold stacking ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                max_features="sqrt", random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    gbm_full = GradientBoostingClassifier(**GBM_PARAMS)
    gbm_full.fit(X_tr, y_tr, sample_weight=w_tr)
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    lr.fit(X_tr_sc, y_tr, sample_weight=w_tr)

    # Out-of-fold predictions for stacking (honest meta-learner training)
    oof_gbm = np.zeros(len(X_tr))
    oof_rf = np.zeros(len(X_tr))
    oof_lr = np.zeros(len(X_tr))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_tr, fold_val in kf.split(X_tr, y_tr):
        g = GradientBoostingClassifier(**GBM_PARAMS)
        g.fit(X_tr[fold_tr], y_tr[fold_tr], sample_weight=w_tr[fold_tr])
        oof_gbm[fold_val] = g.predict_proba(X_tr[fold_val])[:, 1]
        r = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                   max_features="sqrt", random_state=42)
        r.fit(X_tr[fold_tr], y_tr[fold_tr], sample_weight=w_tr[fold_tr])
        oof_rf[fold_val] = r.predict_proba(X_tr[fold_val])[:, 1]
        sc2 = StandardScaler()
        X_fold_sc = sc2.fit_transform(X_tr[fold_tr])
        l = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        l.fit(X_fold_sc, y_tr[fold_tr], sample_weight=w_tr[fold_tr])
        oof_lr[fold_val] = l.predict_proba(sc2.transform(X_tr[fold_val]))[:, 1]

    meta_X_tr = np.column_stack([oof_gbm, oof_rf, oof_lr])
    meta = LogisticRegression(C=10.0, max_iter=500, random_state=42)
    meta.fit(meta_X_tr, y_tr, sample_weight=w_tr)

    # --- Evaluate on test year ---
    year_games = []
    correct_gbm, correct_ens, correct_seed = 0, 0, 0
    total = 0

    for rnd, w_name, l_name, w_seed, l_seed, w_score, l_score in games_by_year[test_year]:
        fa = raw_stats[(w_name, test_year)]
        fb = raw_stats[(l_name, test_year)]
        feats = make_features(fa, fb, w_seed, l_seed, sp_fn)
        X_test = np.array([feats], dtype=np.float32)
        X_test_sc = scaler.transform(X_test)

        # GBM prediction (p = P(team_a=winner wins))
        p_gbm = float(gbm.predict_proba(X_test)[0][1])

        # Ensemble prediction
        meta_X_test = np.column_stack([
            gbm_full.predict_proba(X_test)[:, 1],
            rf.predict_proba(X_test)[:, 1],
            lr.predict_proba(X_test_sc)[:, 1],
        ])
        p_ens = float(meta.predict_proba(meta_X_test)[0][1])

        # Seed-only baseline
        p_seed = sp_fn(w_seed, l_seed)

        # team_a = winner always, so correct prediction = prob >= 0.5
        gbm_right = p_gbm >= 0.5
        ens_right = p_ens >= 0.5
        seed_right = p_seed >= 0.5

        correct_gbm += int(gbm_right)
        correct_ens += int(ens_right)
        correct_seed += int(seed_right)
        total += 1

        rl = ROUND_LABELS.get(rnd, f"R{rnd}")
        round_correct[rl]["gbm"] += int(gbm_right)
        round_correct[rl]["ens"] += int(ens_right)
        round_correct[rl]["seed"] += int(seed_right)
        round_correct[rl]["total"] += 1

        year_games.append({
            "round": rl,
            "round_num": rnd,
            "winner": w_name,
            "loser": l_name,
            "w_seed": w_seed,
            "l_seed": l_seed,
            "w_score": w_score,
            "l_score": l_score,
            "p_gbm": round(p_gbm, 3),
            "p_ens": round(p_ens, 3),
            "p_seed": round(p_seed, 3),
            "gbm_correct": gbm_right,
            "ens_correct": ens_right,
            "seed_correct": seed_right,
            "upset": w_seed > l_seed,
        })

    overall_correct["gbm"] += correct_gbm
    overall_correct["ens"] += correct_ens
    overall_correct["seed"] += correct_seed
    overall_total += total

    actual_champ = None
    champ_pick_gbm = None
    champ_pick_ens = None
    for g in year_games:
        if g["round"] == "Final":
            actual_champ = g["winner"]
            champ_pick_gbm = g["winner"] if g["gbm_correct"] else g["loser"]
            champ_pick_ens = g["winner"] if g["ens_correct"] else g["loser"]

    all_results[str(test_year)] = {
        "games": year_games,
        "n": total,
        "gbm_correct": correct_gbm,
        "ens_correct": correct_ens,
        "seed_correct": correct_seed,
        "gbm_pct": round(correct_gbm / total * 100, 1),
        "ens_pct": round(correct_ens / total * 100, 1),
        "seed_pct": round(correct_seed / total * 100, 1),
        "champion": actual_champ,
        "champ_pick_gbm": champ_pick_gbm,
        "champ_pick_ens": champ_pick_ens,
    }

    champ_mark = "OK" if champ_pick_ens == actual_champ else f"MISS->{champ_pick_ens}"
    print(f"{test_year:>6d} {total:>4d} "
          f"{correct_gbm/total*100:>6.1f}% "
          f"{correct_ens/total*100:>6.1f}% "
          f"{correct_seed/total*100:>6.1f}%  "
          f"{actual_champ} ({champ_mark})")

print("-" * 70)
for key in ["gbm", "ens", "seed"]:
    c = overall_correct[key]
    label = {"gbm": "GBM", "ens": "Ensemble", "seed": "Seed-only"}[key]
    print(f"  {label:>12s}: {c}/{overall_total} = {c/overall_total*100:.1f}%")

print("\nBy round:")
for rl in ["R64", "R32", "S16", "E8", "F4", "Final"]:
    rc = round_correct[rl]
    if rc["total"] == 0:
        continue
    print(f"  {rl:>6s}: GBM {rc['gbm']}/{rc['total']}={rc['gbm']/rc['total']*100:.0f}%  "
          f"Ens {rc['ens']}/{rc['total']}={rc['ens']/rc['total']*100:.0f}%  "
          f"Seed {rc['seed']}/{rc['total']}={rc['seed']/rc['total']*100:.0f}%")

# ─── SAVE FULL RESULTS ──────────────────────────────────────────────────────

output = {
    "description": "Honest LOO backtest — no postseason leak, no seed-prior leak, no HP leak, OOF stacking",
    "years": [str(y) for y in years],
    "by_year": all_results,
    "overall": {
        "total": overall_total,
        "gbm_correct": overall_correct["gbm"],
        "ens_correct": overall_correct["ens"],
        "seed_correct": overall_correct["seed"],
        "gbm_pct": round(overall_correct["gbm"] / overall_total * 100, 1),
        "ens_pct": round(overall_correct["ens"] / overall_total * 100, 1),
        "seed_pct": round(overall_correct["seed"] / overall_total * 100, 1),
    },
    "by_round": {},
}
for rl in ["R64", "R32", "S16", "E8", "F4", "Final"]:
    rc = round_correct[rl]
    if rc["total"] > 0:
        output["by_round"][rl] = {
            "total": rc["total"],
            "gbm_correct": rc["gbm"], "ens_correct": rc["ens"], "seed_correct": rc["seed"],
            "gbm_pct": round(rc["gbm"] / rc["total"] * 100, 1),
            "ens_pct": round(rc["ens"] / rc["total"] * 100, 1),
            "seed_pct": round(rc["seed"] / rc["total"] * 100, 1),
        }

os.makedirs("ncaa_data", exist_ok=True)
with open("ncaa_data/backtest_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: ncaa_data/backtest_results.json")
