"""
Train gradient-boosted model to predict March Madness game winners from team season stats.
Outputs:
  - ncaa_data/model_weights.json  (tree structure + params, portable)
  - ncaa_data/bracket_preds.json  (win probabilities for every possible matchup in 2024)
"""
import csv
import json
import random
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# ── 1. Load team season stats ────────────────────────────────────────────────

FEATURE_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB",
]

# Normalize team names: games CSV uses full names, stats CSV uses abbreviations
NAME_MAP = {
    "UConn": "Connecticut",
    "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.",
    "Iowa State": "Iowa St.",
    "NC State": "North Carolina St.",
    "San Diego State": "San Diego St.",
    "South Dakota State": "South Dakota St.",
    "Colorado State": "Colorado St.",
    "Utah State": "Utah St.",
    "Long Beach State": "Long Beach St.",
    "Montana State": "Montana St.",
    "Washington State": "Washington St.",
    "Morehead State": "Morehead St.",
    "McNeese": "McNeese St.",
    "McNeese State": "McNeese St.",
    "Charleston": "College of Charleston",
    "Grambling State": "Grambling St.",
    "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.",
    "Oregon State": "Oregon St.",
    "Penn State": "Penn St.",
    "Kansas State": "Kansas St.",
    "Arizona State": "Arizona St.",
    "Florida State": "Florida St.",
    "Wichita State": "Wichita St.",
    "Fresno State": "Fresno St.",
    "Boise State": "Boise St.",
    "North Carolina State": "North Carolina St.",
    "Wright State": "Wright St.",
    "Norfolk State": "Norfolk St.",
    "Murray State": "Murray St.",
    "Kent State": "Kent St.",
    "Ball State": "Ball St.",
    "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.",
    "Cleveland State": "Cleveland St.",
    "Jacksonville State": "Jacksonville St.",
    "Georgia State": "Georgia St.",
    "Kennesaw State": "Kennesaw St.",
    "Sam Houston State": "Sam Houston St.",
    "Weber State": "Weber St.",
    "Idaho State": "Idaho St.",
    "Coppin State": "Coppin St.",
    "Delaware State": "Delaware St.",
    "Tennessee State": "Tennessee St.",
    "Appalachian State": "Appalachian St.",
    "New Mexico State": "New Mexico St.",
    "North Dakota State": "North Dakota St.",
    "South Carolina State": "South Carolina St.",
    "Alabama State": "Alabama St.",
    "Alcorn State": "Alcorn St.",
    "Jackson State": "Jackson St.",
    "Morgan State": "Morgan St.",
    "Nicholls State": "Nicholls St.",
    "Portland State": "Portland St.",
    "Loyola–Chicago": "Loyola Chicago",
    "Loyola (MD)": "Loyola MD",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Miami-FL": "Miami FL",
    "Miami-OH": "Miami OH",
    "Saint Mary's (CA)": "Saint Mary's",
    "St. Mary's": "Saint Mary's",
    "Virginia Commonwealth": "VCU",
    "Central Florida": "UCF",
    "Louisiana-Lafayette": "Louisiana Lafayette",
    "Louisiana–Lafayette": "Louisiana Lafayette",
    "Pennsylvania": "Penn",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Cal State Northridge": "Cal St. Northridge",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Detroit Mercy": "Detroit",
    "Middle Tennessee State": "Middle Tennessee",
    "Gardner–Webb": "Gardner Webb",
    "East Tennessee State": "East Tennessee St.",
    "Arkansas–Little Rock": "Arkansas Little Rock",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "California-Irvine": "UC Irvine",
    "Texas A&M-CC": "Texas A&M Corpus Chris",
    "Texas A&M–Corpus Christi": "Texas A&M Corpus Chris",
    "Towson State": "Towson",
    "Troy State": "Troy",
    "Memphis State": "Memphis",
    "Southwest Missouri State": "Missouri St.",
    "SW Missouri State": "Missouri St.",
    "St Johns": "St. John's",
    "St John's": "St. John's",
    "Ole Miss": "Mississippi",
    "Saint Francis (PA)": "St. Francis PA",
}

def normalize_name(name):
    name = name.strip().rstrip("#").strip()
    return NAME_MAP.get(name, name)

stats = {}  # (team, year) -> {col: val}
with open("team_season_stats_2013_2023.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        team = row["TEAM"].strip()
        year = int(row["YEAR"])
        seed_raw = row.get("SEED", "")
        try:
            seed = int(seed_raw)
        except (ValueError, TypeError):
            seed = 16
        feats = {}
        for c in FEATURE_COLS:
            try:
                feats[c] = float(row[c])
            except (ValueError, KeyError):
                feats[c] = 0.0
        feats["SEED"] = seed
        stats[(team, year)] = feats

print(f"Loaded stats: {len(stats)} team-seasons")

# ── 2. Load tournament games & build training rows ──────────────────────────

def make_row(w_feats, l_feats):
    """Feature vector = winner_stats - loser_stats (differential), label = 1."""
    return [w_feats.get(c, 0) - l_feats.get(c, 0) for c in FEATURE_COLS + ["SEED"]]

X, y = [], []
skipped = 0
with open("march_madness_games_1985_2024.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        w_team = normalize_name(row["winning_team_name"])
        l_team = normalize_name(row["losing_team_name"])
        w_seed = int(row.get("winning_team_seed", 16) or 16)
        l_seed = int(row.get("losing_team_seed", 16) or 16)

        w_key = (w_team, year)
        l_key = (l_team, year)
        if w_key not in stats or l_key not in stats:
            skipped += 1
            continue

        w_feats = stats[w_key]
        l_feats = stats[l_key]

        # Randomly swap so the model sees both orientations (label 1 = team_a wins)
        if random.random() < 0.5:
            X.append(make_row(w_feats, l_feats))
            y.append(1)
        else:
            X.append(make_row(l_feats, w_feats))
            y.append(0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"Training samples: {len(X)}  (skipped {skipped} games w/o stats)")

# ── 3. Train gradient boosted classifier ────────────────────────────────────

model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    max_features=0.8,
    random_state=42,
)

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

model.fit(X, y)

import os
os.makedirs("ncaa_data", exist_ok=True)

with open("ncaa_data/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved: ncaa_data/model.pkl")

# ── 4. Feature importance ───────────────────────────────────────────────────

feat_names = FEATURE_COLS + ["SEED"]
imp = model.feature_importances_
ranked = sorted(zip(feat_names, imp), key=lambda x: -x[1])
print("\nTop features:")
for name, score in ranked[:10]:
    print(f"  {name:>8s}: {score:.3f}")

# ── 5. Generate bracket predictions for 2024 ───────────────────────────────

BRACKET_YEAR = 2024

# Use 2023 stats for 2024 tournament teams (since 2024 stats not in dataset)
STATS_YEAR = 2023

# 2024 tournament field (64 teams, seeds 1-16 x 4 regions)
REGIONS = {
    "East": [
        (1, "UConn"), (16, "Stetson"), (8, "Florida Atlantic"), (9, "Northwestern"),
        (5, "San Diego State"), (12, "UAB"), (4, "Auburn"), (13, "Yale"),
        (6, "BYU"), (11, "Duquesne"), (3, "Illinois"), (14, "Morehead State"),
        (7, "Washington State"), (10, "Drake"), (2, "Iowa State"), (15, "South Dakota State"),
    ],
    "West": [
        (1, "North Carolina"), (16, "Wagner"), (8, "Mississippi State"), (9, "Michigan State"),
        (5, "Saint Mary's"), (12, "Grand Canyon"), (4, "Alabama"), (13, "Charleston"),
        (6, "Clemson"), (11, "New Mexico"), (3, "Baylor"), (14, "Colgate"),
        (7, "Dayton"), (10, "Nevada"), (2, "Arizona"), (15, "Long Beach State"),
    ],
    "South": [
        (1, "Houston"), (16, "Longwood"), (8, "Nebraska"), (9, "Texas A&M"),
        (5, "Wisconsin"), (12, "James Madison"), (4, "Duke"), (13, "Vermont"),
        (6, "Texas Tech"), (11, "NC State"), (3, "Kentucky"), (14, "Oakland"),
        (7, "Florida"), (10, "Colorado"), (2, "Marquette"), (15, "Western Kentucky"),
    ],
    "Midwest": [
        (1, "Purdue"), (16, "Montana State"), (8, "Utah State"), (9, "TCU"),
        (5, "Gonzaga"), (12, "McNeese State"), (4, "Kansas"), (13, "Samford"),
        (6, "South Carolina"), (11, "Oregon"), (3, "Creighton"), (14, "Akron"),
        (7, "Texas"), (10, "Colorado State"), (2, "Tennessee"), (15, "Saint Peter's"),
    ],
}

def get_team_feats(team_name, seed):
    """Get features for a team. Try normalized name in STATS_YEAR, then fallback years."""
    norm = normalize_name(team_name)
    for yr in [STATS_YEAR, 2022, 2021]:
        key = (norm, yr)
        if key in stats:
            return stats[key]
    # Also try the original name
    for yr in [STATS_YEAR, 2022, 2021]:
        key = (team_name, yr)
        if key in stats:
            return stats[key]
    print(f"  WARNING: no stats for {team_name} (normalized: {norm}), using seed-based baseline")
    baseline_adjoe = max(115 - (seed - 1) * 2, 85)
    baseline_adjde = min(88 + (seed - 1) * 2, 110)
    baseline_barthag = max(0.95 - (seed - 1) * 0.06, 0.1)
    return {c: 100.0 for c in FEATURE_COLS} | {
        "SEED": seed, "ADJOE": baseline_adjoe, "ADJDE": baseline_adjde, "BARTHAG": baseline_barthag
    }

def predict_matchup(team_a, seed_a, team_b, seed_b):
    """Return P(team_a wins)."""
    fa = get_team_feats(team_a, seed_a)
    fb = get_team_feats(team_b, seed_b)
    diff = np.array([[fa.get(c, 0) - fb.get(c, 0) for c in FEATURE_COLS + ["SEED"]]], dtype=np.float32)
    prob = model.predict_proba(diff)[0][1]
    return float(prob)

def sim_region(teams):
    """Simulate a region bracket. teams = [(seed, name), ...] in standard bracket order."""
    bracket = list(teams)
    rounds_data = []
    while len(bracket) > 1:
        next_round = []
        round_matchups = []
        for i in range(0, len(bracket), 2):
            sa, ta = bracket[i]
            sb, tb = bracket[i + 1]
            prob_a = predict_matchup(ta, sa, tb, sb)
            round_matchups.append({
                "team_a": ta, "seed_a": sa,
                "team_b": tb, "seed_b": sb,
                "prob_a": round(prob_a, 3),
                "prob_b": round(1 - prob_a, 3),
                "pick": ta if prob_a >= 0.5 else tb,
                "pick_seed": sa if prob_a >= 0.5 else sb,
                "pick_prob": round(max(prob_a, 1 - prob_a), 3),
            })
            if prob_a >= 0.5:
                next_round.append((sa, ta))
            else:
                next_round.append((sb, tb))
        rounds_data.append(round_matchups)
        bracket = next_round
    return rounds_data, bracket[0]

print(f"\n=== Simulating 2024 bracket ===")
bracket_output = {"year": BRACKET_YEAR, "regions": {}, "final_four": [], "championship": None, "champion": None}

final_four = []
for region_name, teams in REGIONS.items():
    rounds, winner = sim_region(teams)
    bracket_output["regions"][region_name] = {
        "teams": [{"seed": s, "name": t} for s, t in teams],
        "rounds": rounds,
        "winner": {"seed": winner[0], "name": winner[1]},
    }
    final_four.append(winner)
    print(f"  {region_name}: {winner[1]} ({winner[0]} seed)")

# Final Four
print("\nFinal Four:")
ff_matchups = []
for i in range(0, len(final_four), 2):
    sa, ta = final_four[i]
    sb, tb = final_four[i + 1]
    prob_a = predict_matchup(ta, sa, tb, sb)
    pick = ta if prob_a >= 0.5 else tb
    pick_seed = sa if prob_a >= 0.5 else sb
    ff_matchups.append({
        "team_a": ta, "seed_a": sa,
        "team_b": tb, "seed_b": sb,
        "prob_a": round(prob_a, 3),
        "prob_b": round(1 - prob_a, 3),
        "pick": pick, "pick_seed": pick_seed,
        "pick_prob": round(max(prob_a, 1 - prob_a), 3),
    })
    print(f"  {ta}({sa}) vs {tb}({sb}) -> {pick} ({max(prob_a, 1-prob_a):.1%})")
    if prob_a >= 0.5:
        final_four[i // 2] = (sa, ta)
    else:
        final_four[i // 2] = (sb, tb)

bracket_output["final_four"] = ff_matchups
finalists = final_four[:2]

# Championship
sa, ta = finalists[0]
sb, tb = finalists[1]
prob_a = predict_matchup(ta, sa, tb, sb)
pick = ta if prob_a >= 0.5 else tb
pick_seed = sa if prob_a >= 0.5 else sb
bracket_output["championship"] = {
    "team_a": ta, "seed_a": sa,
    "team_b": tb, "seed_b": sb,
    "prob_a": round(prob_a, 3),
    "prob_b": round(1 - prob_a, 3),
    "pick": pick, "pick_seed": pick_seed,
    "pick_prob": round(max(prob_a, 1 - prob_a), 3),
}
bracket_output["champion"] = {"seed": pick_seed, "name": pick}
print(f"\nChampion: {pick} ({pick_seed} seed)")

with open("ncaa_data/bracket_preds.json", "w") as f:
    json.dump(bracket_output, f, indent=2)
print("\nPredictions saved: ncaa_data/bracket_preds.json")
