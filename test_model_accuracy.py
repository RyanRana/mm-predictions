"""
Leave-one-year-out backtesting: for each tournament year, train on ALL other years,
then predict every game in the held-out year. Reports per-year and per-round accuracy.
"""
import csv
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from collections import defaultdict

# ── Reuse config from train_bracket_model ────────────────────────────────────

FEATURE_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB",
]

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

def normalize_name(name):
    return NAME_MAP.get(name.strip().rstrip("#").strip(), name.strip().rstrip("#").strip())

# ── Load data ────────────────────────────────────────────────────────────────

stats = {}
with open("team_season_stats_2013_2023.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        team = row["TEAM"].strip()
        year = int(row["YEAR"])
        try:
            seed = int(row.get("SEED", "16"))
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

ROUND_NAMES = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "Champ"}

games_by_year = defaultdict(list)
with open("march_madness_games_1985_2024.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        year = int(row["year"])
        rnd = int(row["round_of"])
        w = normalize_name(row["winning_team_name"])
        l = normalize_name(row["losing_team_name"])
        w_seed = int(row.get("winning_team_seed", 16) or 16)
        l_seed = int(row.get("losing_team_seed", 16) or 16)
        if (w, year) in stats and (l, year) in stats:
            games_by_year[year].append((w, l, w_seed, l_seed, rnd))

ALL_FEATS = FEATURE_COLS + ["SEED"]

def make_diff(fa, fb):
    return [fa.get(c, 0) - fb.get(c, 0) for c in ALL_FEATS]

# ── Leave-one-year-out evaluation ────────────────────────────────────────────

years = sorted(games_by_year.keys())
print(f"Years with matchable games: {years}")
print(f"{'Year':>6s} {'Games':>5s} {'Correct':>7s} {'Acc%':>6s}  {'Champ':>7s}  By round")
print("-" * 90)

overall_correct = 0
overall_total = 0
round_correct = defaultdict(int)
round_total = defaultdict(int)
year_results = []

for test_year in years:
    # Build training set from all OTHER years
    X_train, y_train = [], []
    for yr in years:
        if yr == test_year:
            continue
        for w, l, ws, ls, rnd in games_by_year[yr]:
            wf = stats[(w, yr)]
            lf = stats[(l, yr)]
            X_train.append(make_diff(wf, lf))
            y_train.append(1)
            X_train.append(make_diff(lf, wf))
            y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    # Test on held-out year
    correct = 0
    total = 0
    round_acc = defaultdict(lambda: [0, 0])  # [correct, total]
    champion_pick = None

    for w, l, ws, ls, rnd in games_by_year[test_year]:
        wf = stats[(w, test_year)]
        lf = stats[(l, test_year)]
        diff = np.array([make_diff(wf, lf)], dtype=np.float32)
        prob_w = model.predict_proba(diff)[0][1]

        predicted_winner = w if prob_w >= 0.5 else l
        actual_winner = w
        is_correct = predicted_winner == actual_winner
        if is_correct:
            correct += 1
        total += 1
        rname = ROUND_NAMES.get(rnd, f"R{rnd}")
        round_acc[rname][0] += int(is_correct)
        round_acc[rname][1] += 1
        round_correct[rname] += int(is_correct)
        round_total[rname] += 1

        if rnd == 2:
            champion_pick = predicted_winner

    acc = correct / total * 100 if total else 0
    overall_correct += correct
    overall_total += total

    # Find actual champion
    for w, l, ws, ls, rnd in games_by_year[test_year]:
        if rnd == 2:
            actual_champ = w
            break
    else:
        actual_champ = "?"

    champ_str = "✓" if champion_pick == actual_champ else f"✗ ({champion_pick or '?'})"

    round_strs = []
    for rname in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
        if rname in round_acc:
            c, t = round_acc[rname]
            round_strs.append(f"{rname}:{c}/{t}")

    print(f"{test_year:>6d} {total:>5d} {correct:>7d} {acc:>5.1f}%  {champ_str:>7s}  {' '.join(round_strs)}")
    year_results.append((test_year, total, correct, acc, actual_champ, champion_pick))

print("-" * 90)
overall_acc = overall_correct / overall_total * 100 if overall_total else 0
print(f"{'TOTAL':>6s} {overall_total:>5d} {overall_correct:>7d} {overall_acc:>5.1f}%")

print(f"\nPer-round accuracy across all years:")
for rname in ["R64", "R32", "S16", "E8", "F4", "Champ"]:
    if rname in round_total:
        c = round_correct[rname]
        t = round_total[rname]
        print(f"  {rname:>5s}: {c:>4d}/{t:<4d} = {c/t*100:5.1f}%")

champ_correct = sum(1 for _, _, _, _, actual, pred in year_results if actual == pred)
print(f"\nChampion correct: {champ_correct}/{len(year_results)} = {champ_correct/len(year_results)*100:.1f}%")
