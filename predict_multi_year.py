"""
Generate bracket predictions for 2022, 2025, and 2026.
- 2022: uses actual bracket + stats from our CSV (train on all years except 2022)
- 2025: hardcoded bracket + KenPom pre-tournament ratings (train on all years)
- 2026: hardcoded bracket + estimated pre-tournament stats (train on all years)
Outputs: ncaa_data/bracket_preds_{year}.json for each year
"""
import csv, json, os, random, warnings, pickle
from collections import defaultdict
from math import log

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)
os.makedirs("ncaa_data", exist_ok=True)

# ─── NAME MAP ────────────────────────────────────────────────────────────────

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

# ─── LOAD DATA ───────────────────────────────────────────────────────────────

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

tourney_games = []
with open("march_madness_games_1985_2024.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        rnd = int(row["round_of"])
        w = norm(row["winning_team_name"])
        l = norm(row["losing_team_name"])
        w_seed = int(row.get("winning_team_seed", 16) or 16)
        l_seed = int(row.get("losing_team_seed", 16) or 16)
        tourney_games.append((year, rnd, w, l, w_seed, l_seed))

print(f"Stats: {len(raw_stats)} team-seasons | Games: {len(tourney_games)}")

# ─── SEED PRIORS ─────────────────────────────────────────────────────────────

def build_seed_prior(exclude_year=None):
    hist = defaultdict(lambda: [0, 0])
    for yr, rnd, w, l, w_seed, l_seed in tourney_games:
        if yr == exclude_year:
            continue
        key = (min(w_seed, l_seed), max(w_seed, l_seed))
        if w_seed <= l_seed:
            hist[key][0] += 1
        else:
            hist[key][1] += 1
    def _prior(sa, sb):
        lo, hi = min(sa, sb), max(sa, sb)
        wins, losses = hist[(lo, hi)]
        t = wins + losses
        if t < 3:
            return 0.5
        p = (wins + 1) / (t + 2)
        return p if sa <= sb else (1 - p)
    return _prior

# ─── FEATURES ────────────────────────────────────────────────────────────────

def make_features(fa, fb, sa, sb, sp_fn):
    f = []
    for c in BASE_COLS:
        f.append(fa.get(c, 0) - fb.get(c, 0))
    f.append(sa - sb)
    f.append(fa.get("ADJOE", 100) / max(fb.get("ADJDE", 100), 1))
    f.append(fb.get("ADJOE", 100) / max(fa.get("ADJDE", 100), 1))
    f.append(abs(fa.get("ADJ_T", 67) - fb.get("ADJ_T", 67)))
    f.append(fa.get("3P_O", 33) - fb.get("3P_D", 33))
    f.append(fb.get("3P_O", 33) - fa.get("3P_D", 33))
    f.append((fa.get("ORB", 30) + fa.get("DRB", 28)) - (fb.get("ORB", 30) + fb.get("DRB", 28)))
    f.append(sp_fn(sa, sb))
    wa = fa.get("W", 20) / max(fa.get("G", 30), 1)
    wb = fb.get("W", 20) / max(fb.get("G", 30), 1)
    f.append(wa - wb)
    ba = max(fa.get("BARTHAG", 0.5), 0.01)
    bb = max(fb.get("BARTHAG", 0.5), 0.01)
    f.append(log(ba / bb))
    return f

GBM_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "max_features": 0.8, "min_samples_leaf": 10,
    "random_state": 42,
}

def build_train_data(exclude_year, sp_fn):
    X, y, w = [], [], []
    for yr, rnd, winner, loser, w_seed, l_seed in tourney_games:
        if yr == exclude_year:
            continue
        if (winner, yr) not in raw_stats or (loser, yr) not in raw_stats:
            continue
        fa, fb = raw_stats[(winner, yr)], raw_stats[(loser, yr)]
        feats_ab = make_features(fa, fb, w_seed, l_seed, sp_fn)
        feats_ba = make_features(fb, fa, l_seed, w_seed, sp_fn)
        recency = 1.0 + 0.15 * (yr - 2013)
        rw = {64: 1.0, 32: 1.2, 16: 1.5, 8: 2.0, 4: 2.5, 2: 3.0}.get(rnd, 1.0)
        sw = recency * rw
        X.append(feats_ab); y.append(1); w.append(sw)
        X.append(feats_ba); y.append(0); w.append(sw)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(w, dtype=np.float32)

def train_ensemble(X, y, w):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    gbm = GradientBoostingClassifier(**GBM_PARAMS); gbm.fit(X, y, sample_weight=w)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                max_features="sqrt", random_state=42)
    rf.fit(X, y, sample_weight=w)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_sc, y, sample_weight=w)
    # OOF stacking
    oof_g, oof_r, oof_l = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in kf.split(X, y):
        g2 = GradientBoostingClassifier(**GBM_PARAMS); g2.fit(X[tr], y[tr], sample_weight=w[tr])
        oof_g[va] = g2.predict_proba(X[va])[:, 1]
        r2 = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                    max_features="sqrt", random_state=42)
        r2.fit(X[tr], y[tr], sample_weight=w[tr]); oof_r[va] = r2.predict_proba(X[va])[:, 1]
        s2 = StandardScaler(); Xf = s2.fit_transform(X[tr])
        l2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        l2.fit(Xf, y[tr], sample_weight=w[tr]); oof_l[va] = l2.predict_proba(s2.transform(X[va]))[:, 1]
    meta_X = np.column_stack([oof_g, oof_r, oof_l])
    meta = LogisticRegression(C=10.0, max_iter=500, random_state=42)
    meta.fit(meta_X, y, sample_weight=w)
    return {"gbm": gbm, "rf": rf, "lr": lr, "scaler": scaler, "meta": meta}

def ensemble_predict(mdl, X_in, sp_fn, sa, sb):
    X_sc = mdl["scaler"].transform(X_in)
    meta_X = np.column_stack([
        mdl["gbm"].predict_proba(X_in)[:, 1],
        mdl["rf"].predict_proba(X_in)[:, 1],
        mdl["lr"].predict_proba(X_sc)[:, 1],
    ])
    p = float(mdl["meta"].predict_proba(meta_X)[0][1])
    sp = sp_fn(sa, sb)
    return p * 0.85 + sp * 0.15

# ─── SIMULATE BRACKET ────────────────────────────────────────────────────────

def sim_bracket(regions, team_feats, mdl, sp_fn):
    output = {"regions": {}, "final_four": [], "championship": None, "champion": None}
    ff = []
    for rname, teams in regions.items():
        bracket = list(teams)
        rounds_data = []
        while len(bracket) > 1:
            nxt = []
            rnd_matchups = []
            for i in range(0, len(bracket), 2):
                sa, ta = bracket[i]
                sb, tb = bracket[i + 1]
                fa, fb = team_feats[ta], team_feats[tb]
                feats = make_features(fa, fb, sa, sb, sp_fn)
                X_in = np.array([feats], dtype=np.float32)
                pa = ensemble_predict(mdl, X_in, sp_fn, sa, sb)
                m = {
                    "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
                    "prob_a": round(pa, 3), "prob_b": round(1 - pa, 3),
                    "pick": ta if pa >= 0.5 else tb,
                    "pick_seed": sa if pa >= 0.5 else sb,
                    "pick_prob": round(max(pa, 1 - pa), 3),
                }
                rnd_matchups.append(m)
                nxt.append((sa, ta) if pa >= 0.5 else (sb, tb))
            rounds_data.append(rnd_matchups)
            bracket = nxt
        output["regions"][rname] = {
            "teams": [{"seed": s, "name": t} for s, t in teams],
            "rounds": rounds_data,
            "winner": {"seed": bracket[0][0], "name": bracket[0][1]},
        }
        ff.append(bracket[0])
        print(f"    {rname:>10s}: ({bracket[0][0]}) {bracket[0][1]}")

    ff_matchups = []
    for i in range(0, len(ff), 2):
        sa, ta = ff[i]; sb, tb = ff[i + 1]
        fa, fb = team_feats[ta], team_feats[tb]
        feats = make_features(fa, fb, sa, sb, sp_fn)
        X_in = np.array([feats], dtype=np.float32)
        pa = ensemble_predict(mdl, X_in, sp_fn, sa, sb)
        pick = ta if pa >= 0.5 else tb
        pick_seed = sa if pa >= 0.5 else sb
        ff_matchups.append({
            "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
            "prob_a": round(pa, 3), "prob_b": round(1 - pa, 3),
            "pick": pick, "pick_seed": pick_seed,
            "pick_prob": round(max(pa, 1 - pa), 3),
        })
        print(f"    FF: ({sa}){ta} {pa*100:.0f}% vs ({sb}){tb} {(1-pa)*100:.0f}%")
        if pa >= 0.5:
            ff[i // 2] = (sa, ta)
        else:
            ff[i // 2] = (sb, tb)
    output["final_four"] = ff_matchups

    sa, ta = ff[0]; sb, tb = ff[1]
    fa, fb = team_feats[ta], team_feats[tb]
    feats = make_features(fa, fb, sa, sb, sp_fn)
    X_in = np.array([feats], dtype=np.float32)
    pa = ensemble_predict(mdl, X_in, sp_fn, sa, sb)
    pick = ta if pa >= 0.5 else tb
    pick_seed = sa if pa >= 0.5 else sb
    output["championship"] = {
        "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
        "prob_a": round(pa, 3), "prob_b": round(1 - pa, 3),
        "pick": pick, "pick_seed": pick_seed,
        "pick_prob": round(max(pa, 1 - pa), 3),
    }
    output["champion"] = {"seed": pick_seed, "name": pick}
    print(f"    CHAMP: ({sa}){ta} {pa*100:.0f}% vs ({sb}){tb} {(1-pa)*100:.0f}%")
    print(f"    >>> ({pick_seed}) {pick} <<<")
    return output


def est_barthag(adjoe, adjde):
    margin = adjoe - adjde
    return 1.0 / (1.0 + 10 ** (-margin / 12.0))

def make_team(adjoe, adjde, seed, record="30-5"):
    parts = record.split("-")
    w, l = int(parts[0]), int(parts[1])
    g = w + l
    return {
        "ADJOE": adjoe, "ADJDE": adjde, "BARTHAG": est_barthag(adjoe, adjde),
        "EFG_O": 50.0, "EFG_D": 50.0, "TOR": 17.0, "TORD": 18.0,
        "ORB": 30.0, "DRB": 28.0, "FTR": 33.0, "FTRD": 30.0,
        "2P_O": 50.0, "2P_D": 47.0, "3P_O": 34.0, "3P_D": 33.0,
        "ADJ_T": 67.0, "WAB": (w / g * 30 - 15) if g > 0 else 0,
        "SEED": seed, "G": g, "W": w,
    }

# KenPom net rating -> approximate ADJOE/ADJDE split
def kenpom_to_stats(net_rating, seed, record="25-8"):
    adjoe = 105 + net_rating * 0.42
    adjde = 105 - net_rating * 0.58
    return make_team(adjoe, adjde, seed, record)

# ═══════════════════════════════════════════════════════════════════════════════
# 2022 PREDICTION (honest: train on everything except 2022)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  2022 BRACKET PREDICTION (trained without 2022 data)")
print("=" * 70)

sp_2022 = build_seed_prior(exclude_year=2022)
X_tr, y_tr, w_tr = build_train_data(exclude_year=2022, sp_fn=sp_2022)
mdl_2022 = train_ensemble(X_tr, y_tr, w_tr)
print(f"  Training samples: {len(X_tr)}")

feats_2022 = {}
for (team, yr), f in raw_stats.items():
    if yr == 2022:
        feats_2022[team] = f
for t in ["UConn", "Miami (FL)", "Saint Mary's", "Saint Peter's", "Cal State Fullerton",
          "South Dakota State", "Iowa State", "Michigan State", "Colorado State",
          "Georgia State", "Jacksonville State", "New Mexico State", "Montana State",
          "Boise State", "Ohio State", "Murray State", "Virginia Tech", "Norfolk State",
          "Wright State", "Texas Southern", "San Diego State", "Loyola Chicago"]:
    n = norm(t)
    if n not in feats_2022 and (n, 2022) in raw_stats:
        feats_2022[t] = raw_stats[(n, 2022)]
    elif t not in feats_2022 and (t, 2022) in raw_stats:
        feats_2022[t] = raw_stats[(t, 2022)]

REGIONS_2022 = {
    "West": [
        (1, "Gonzaga"), (16, "Georgia State"), (8, "Boise State"), (9, "Memphis"),
        (5, "UConn"), (12, "New Mexico State"), (4, "Arkansas"), (13, "Vermont"),
        (6, "Alabama"), (11, "Notre Dame"), (3, "Texas Tech"), (14, "Montana State"),
        (7, "Michigan State"), (10, "Davidson"), (2, "Duke"), (15, "Cal State Fullerton"),
    ],
    "East": [
        (1, "Baylor"), (16, "Norfolk State"), (8, "North Carolina"), (9, "Marquette"),
        (5, "Saint Mary's"), (12, "Indiana"), (4, "UCLA"), (13, "Akron"),
        (6, "Texas"), (11, "Virginia Tech"), (3, "Purdue"), (14, "Yale"),
        (7, "Murray State"), (10, "San Francisco"), (2, "Kentucky"), (15, "Saint Peter's"),
    ],
    "South": [
        (1, "Arizona"), (16, "Wright State"), (8, "Seton Hall"), (9, "TCU"),
        (5, "Houston"), (12, "UAB"), (4, "Illinois"), (13, "Chattanooga"),
        (6, "Colorado State"), (11, "Michigan"), (3, "Tennessee"), (14, "Longwood"),
        (7, "Ohio State"), (10, "Loyola Chicago"), (2, "Villanova"), (15, "Delaware"),
    ],
    "Midwest": [
        (1, "Kansas"), (16, "Texas Southern"), (8, "San Diego State"), (9, "Creighton"),
        (5, "Iowa"), (12, "Richmond"), (4, "Providence"), (13, "South Dakota State"),
        (6, "LSU"), (11, "Iowa State"), (3, "Wisconsin"), (14, "Colgate"),
        (7, "USC"), (10, "Miami (FL)"), (2, "Auburn"), (15, "Jacksonville State"),
    ],
}

# Ensure all teams have features
for rname, teams in REGIONS_2022.items():
    for seed, tname in teams:
        n = norm(tname)
        if tname not in feats_2022:
            if n in feats_2022:
                feats_2022[tname] = feats_2022[n]
            elif (n, 2022) in raw_stats:
                feats_2022[tname] = raw_stats[(n, 2022)]
            elif (tname, 2022) in raw_stats:
                feats_2022[tname] = raw_stats[(tname, 2022)]
            else:
                for yr in [2021, 2023]:
                    if (n, yr) in raw_stats:
                        feats_2022[tname] = raw_stats[(n, yr)]; break
                    if (tname, yr) in raw_stats:
                        feats_2022[tname] = raw_stats[(tname, yr)]; break
                else:
                    print(f"  WARNING: no stats for {tname} (norm: {n}), using seed baseline")
                    feats_2022[tname] = make_team(115 - seed * 1.5, 88 + seed * 1.5, seed)

result_2022 = sim_bracket(REGIONS_2022, feats_2022, mdl_2022, sp_2022)
result_2022["year"] = 2022
with open("ncaa_data/bracket_preds_2022.json", "w") as f:
    json.dump(result_2022, f, indent=2)
print("  Saved: ncaa_data/bracket_preds_2022.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 2025 PREDICTION (train on all years, use KenPom pre-tournament stats)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  2025 BRACKET PREDICTION (KenPom pre-tournament stats)")
print("=" * 70)

sp_all = build_seed_prior()
X_all, y_all, w_all = build_train_data(exclude_year=None, sp_fn=sp_all)
mdl_all = train_ensemble(X_all, y_all, w_all)
print(f"  Training samples: {len(X_all)}")

# KenPom net ratings for all 68 teams (pre-tournament March 2025)
TEAMS_2025 = {
    "Auburn": kenpom_to_stats(35.16, 1, "28-5"),
    "Duke": kenpom_to_stats(38.27, 1, "31-3"),
    "Houston": kenpom_to_stats(35.42, 1, "30-4"),
    "Florida": kenpom_to_stats(36.24, 1, "30-4"),
    "Tennessee": kenpom_to_stats(31.20, 2, "27-7"),
    "Alabama": kenpom_to_stats(30.05, 2, "25-8"),
    "Michigan State": kenpom_to_stats(27.64, 2, "27-6"),
    "St. John's": kenpom_to_stats(26.32, 2, "30-4"),
    "Iowa State": kenpom_to_stats(27.00, 3, "24-9"),
    "Wisconsin": kenpom_to_stats(25.99, 3, "26-9"),
    "Kentucky": kenpom_to_stats(24.32, 3, "22-11"),
    "Texas Tech": kenpom_to_stats(28.05, 3, "25-8"),
    "Texas A&M": kenpom_to_stats(24.13, 4, "22-10"),
    "Arizona": kenpom_to_stats(25.74, 4, "22-12"),
    "Purdue": kenpom_to_stats(23.87, 4, "22-11"),
    "Maryland": kenpom_to_stats(26.35, 4, "25-8"),
    "Michigan": kenpom_to_stats(22.31, 5, "25-9"),
    "Oregon": kenpom_to_stats(20.12, 5, "24-9"),
    "Clemson": kenpom_to_stats(23.94, 5, "27-6"),
    "Memphis": kenpom_to_stats(15.39, 5, "29-5"),
    "Ole Miss": kenpom_to_stats(22.14, 6, "22-11"),
    "BYU": kenpom_to_stats(22.36, 6, "24-9"),
    "Illinois": kenpom_to_stats(23.59, 6, "21-12"),
    "Missouri": kenpom_to_stats(24.49, 6, "22-11"),
    "Marquette": kenpom_to_stats(21.53, 7, "23-10"),
    "Saint Mary's": kenpom_to_stats(23.13, 7, "28-5"),
    "UCLA": kenpom_to_stats(21.78, 7, "22-10"),
    "Kansas": kenpom_to_stats(23.30, 7, "21-12"),
    "Louisville": kenpom_to_stats(22.72, 8, "27-7"),
    "Mississippi State": kenpom_to_stats(20.16, 8, "21-12"),
    "Gonzaga": kenpom_to_stats(27.28, 8, "25-8"),
    "UConn": kenpom_to_stats(19.29, 8, "23-10"),
    "Creighton": kenpom_to_stats(18.63, 9, "24-10"),
    "Oklahoma": kenpom_to_stats(18.39, 9, "20-13"),
    "Georgia": kenpom_to_stats(19.40, 9, "20-12"),
    "North Carolina": kenpom_to_stats(20.65, 11, "23-13"),
    "VCU": kenpom_to_stats(20.22, 11, "28-6"),
    "Drake": kenpom_to_stats(14.17, 11, "30-3"),
    "Xavier": kenpom_to_stats(17.17, 11, "21-11"),
    "Texas": kenpom_to_stats(17.19, 11, "19-15"),
    "New Mexico": kenpom_to_stats(17.01, 10, "26-7"),
    "Vanderbilt": kenpom_to_stats(16.20, 10, "20-12"),
    "Utah State": kenpom_to_stats(14.87, 10, "26-7"),
    "Arkansas": kenpom_to_stats(17.77, 10, "20-13"),
    "UC San Diego": kenpom_to_stats(18.40, 12, "30-4"),
    "Liberty": kenpom_to_stats(13.78, 12, "28-6"),
    "McNeese": kenpom_to_stats(13.79, 12, "27-6"),
    "Colorado State": kenpom_to_stats(16.95, 12, "25-9"),
    "Yale": kenpom_to_stats(11.03, 13, "22-7"),
    "Akron": kenpom_to_stats(6.92, 13, "28-6"),
    "High Point": kenpom_to_stats(9.35, 13, "29-5"),
    "Grand Canyon": kenpom_to_stats(7.19, 13, "26-7"),
    "Lipscomb": kenpom_to_stats(9.54, 14, "25-9"),
    "Montana": kenpom_to_stats(0.27, 14, "25-9"),
    "Troy": kenpom_to_stats(7.17, 14, "23-10"),
    "UNC Wilmington": kenpom_to_stats(5.18, 14, "27-7"),
    "Bryant": kenpom_to_stats(1.17, 15, "23-11"),
    "Robert Morris": kenpom_to_stats(2.15, 15, "26-8"),
    "Wofford": kenpom_to_stats(4.23, 15, "19-15"),
    "Omaha": kenpom_to_stats(0.47, 15, "22-12"),
    "Alabama State": kenpom_to_stats(-9.25, 16, "20-15"),
    "Mount St. Mary's": kenpom_to_stats(-6.72, 16, "22-12"),
    "SIU-Edwardsville": kenpom_to_stats(-4.25, 16, "22-11"),
    "Norfolk State": kenpom_to_stats(-1.50, 16, "24-10"),
    "San Diego State": kenpom_to_stats(14.91, 11, "21-10"),
}

REGIONS_2025 = {
    "South": [
        (1, "Auburn"), (16, "Alabama State"), (8, "Louisville"), (9, "Creighton"),
        (5, "Michigan"), (12, "UC San Diego"), (4, "Texas A&M"), (13, "Yale"),
        (6, "Ole Miss"), (11, "North Carolina"), (3, "Iowa State"), (14, "Lipscomb"),
        (7, "Marquette"), (10, "New Mexico"), (2, "Michigan State"), (15, "Bryant"),
    ],
    "East": [
        (1, "Duke"), (16, "Mount St. Mary's"), (8, "Mississippi State"), (9, "Oklahoma"),
        (5, "Oregon"), (12, "Liberty"), (4, "Arizona"), (13, "Akron"),
        (6, "BYU"), (11, "VCU"), (3, "Wisconsin"), (14, "Montana"),
        (7, "Saint Mary's"), (10, "Vanderbilt"), (2, "Alabama"), (15, "Robert Morris"),
    ],
    "Midwest": [
        (1, "Houston"), (16, "SIU-Edwardsville"), (8, "Gonzaga"), (9, "Georgia"),
        (5, "Clemson"), (12, "McNeese"), (4, "Purdue"), (13, "High Point"),
        (6, "Illinois"), (11, "Xavier"), (3, "Kentucky"), (14, "Troy"),
        (7, "UCLA"), (10, "Utah State"), (2, "Tennessee"), (15, "Wofford"),
    ],
    "West": [
        (1, "Florida"), (16, "Norfolk State"), (8, "UConn"), (9, "San Diego State"),
        (5, "Memphis"), (12, "Colorado State"), (4, "Maryland"), (13, "Grand Canyon"),
        (6, "Missouri"), (11, "Drake"), (3, "Texas Tech"), (14, "UNC Wilmington"),
        (7, "Kansas"), (10, "Arkansas"), (2, "St. John's"), (15, "Omaha"),
    ],
}

result_2025 = sim_bracket(REGIONS_2025, TEAMS_2025, mdl_all, sp_all)
result_2025["year"] = 2025
with open("ncaa_data/bracket_preds_2025.json", "w") as f:
    json.dump(result_2025, f, indent=2)
print("  Saved: ncaa_data/bracket_preds_2025.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 2026 PREDICTION (train on all years, use estimated pre-tournament stats)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  2026 BRACKET PREDICTION (estimated pre-tournament stats)")
print("=" * 70)

TEAMS_2026 = {
    "Duke": make_team(129.5, 86.6, 1, "29-2"),
    "Florida": make_team(126.0, 90.6, 1, "25-6"),
    "Michigan": make_team(130.8, 88.8, 1, "29-2"),
    "Arizona": make_team(125.4, 88.5, 1, "29-2"),
    "Michigan State": make_team(122.2, 92.6, 2, "25-6"),
    "Houston": make_team(123.5, 90.4, 2, "26-5"),
    "UConn": make_team(122.9, 94.1, 2, "27-4"),
    "Illinois": make_team(133.6, 97.8, 2, "24-7"),
    "Iowa State": make_team(124.5, 91.7, 3, "26-6"),
    "Nebraska": make_team(119.7, 91.5, 3, "26-5"),
    "Purdue": make_team(131.4, 100.0, 3, "23-8"),
    "Alabama": make_team(129.0, 101.8, 3, "23-8"),
    "Texas Tech": make_team(125.7, 98.2, 4, "22-9"),
    "Virginia": make_team(122.4, 95.9, 4, "27-4"),
    "Kansas": make_team(118.7, 93.0, 4, "22-9"),
    "Gonzaga": make_team(123.2, 90.5, 4, "30-3"),
    "Arkansas": make_team(128.6, 101.3, 5, "23-8"),
    "St. John's": make_team(120.5, 94.9, 5, "25-6"),
    "Vanderbilt": make_team(125.8, 99.3, 5, "24-7"),
    "Wisconsin": make_team(126.1, 102.9, 5, "22-9"),
    "Tennessee": make_team(121.7, 94.7, 6, "21-10"),
    "Louisville": make_team(124.8, 96.3, 6, "23-9"),
    "North Carolina": make_team(122.1, 98.0, 6, "24-7"),
    "BYU": make_team(125.8, 101.6, 6, "23-10"),
    "Villanova": make_team(120.3, 99.7, 7, "24-7"),
    "Saint Mary's": make_team(121.5, 95.9, 7, "27-5"),
    "Kentucky": make_team(122.3, 98.8, 7, "20-12"),
    "Miami (Fla.)": make_team(121.9, 100.3, 7, "24-7"),
    "Georgia": make_team(125.5, 102.6, 8, "22-9"),
    "Utah State": make_team(124.3, 101.9, 8, "25-6"),
    "Clemson": make_team(116.8, 96.6, 8, "23-9"),
    "UCLA": make_team(123.6, 101.8, 8, "21-10"),
    "Ohio State": make_team(125.4, 102.5, 9, "20-11"),
    "Iowa": make_team(123.3, 98.9, 9, "21-11"),
    "TCU": make_team(116.6, 99.3, 9, "22-10"),
    "Texas A&M": make_team(120.5, 101.6, 9, "21-10"),
    "Miami (Ohio)": make_team(117.7, 107.3, 10, "31-0"),
    "NC State": make_team(124.9, 103.9, 10, "20-12"),
    "UCF": make_team(120.9, 106.6, 10, "21-10"),
    "Saint Louis": make_team(122.3, 99.8, 10, "27-4"),
    "SMU": make_team(123.4, 103.8, 11, "20-13"),
    "Missouri": make_team(119.3, 104.7, 11, "20-11"),
    "Texas": make_team(124.5, 106.3, 11, "18-14"),
    "Santa Clara": make_team(123.1, 103.5, 11, "26-8"),
    "South Florida": make_team(119.0, 101.7, 12, "23-8"),
    "High Point": make_team(118.6, 107.0, 12, "30-4"),
    "Yale": make_team(120.3, 109.0, 12, "23-5"),
    "Liberty": make_team(117.3, 112.3, 12, "25-7"),
    "Hofstra": make_team(114.4, 105.1, 13, "24-10"),
    "Northern Iowa": make_team(110.2, 96.9, 13, "23-12"),
    "Stephen F. Austin": make_team(113.2, 104.3, 13, "28-5"),
    "Utah Valley": make_team(113.0, 103.5, 13, "24-7"),
    "North Dakota State": make_team(111.6, 106.5, 14, "27-7"),
    "UC Irvine": make_team(103.7, 98.4, 14, "22-10"),
    "Merrimack": make_team(106.2, 108.1, 14, "23-11"),
    "Troy": make_team(110.6, 108.6, 14, "22-11"),
    "Wright State": make_team(111.6, 109.5, 15, "23-11"),
    "Tennessee State": make_team(108.9, 110.5, 15, "23-9"),
    "Queens": make_team(116.1, 118.1, 15, "21-13"),
    "Portland State": make_team(104.5, 102.4, 15, "20-11"),
    "Bethune-Cookman": make_team(104.0, 109.0, 16, "18-14"),
    "Furman": make_team(107.3, 108.8, 16, "22-12"),
    "Howard": make_team(104.2, 107.0, 16, "21-10"),
    "UMBC": make_team(107.6, 110.7, 16, "23-8"),
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

result_2026 = sim_bracket(REGIONS_2026, TEAMS_2026, mdl_all, sp_all)
result_2026["year"] = 2026
with open("ncaa_data/bracket_preds_2026.json", "w") as f:
    json.dump(result_2026, f, indent=2)
print("  Saved: ncaa_data/bracket_preds_2026.json")

# Also update React app with 2026
if os.path.isdir("bracket-app/src"):
    with open("bracket-app/src/bracket_preds.json", "w") as f:
        json.dump(result_2026, f, indent=2)
    print("  Updated bracket-app/src/bracket_preds.json (2026)")

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
for yr, res in [(2022, result_2022), (2025, result_2025), (2026, result_2026)]:
    c = res["champion"]
    ff = [res["regions"][r]["winner"]["name"] for r in res["regions"]]
    print(f"  {yr}: Champion = ({c['seed']}) {c['name']}  |  FF = {', '.join(ff)}")
