"""
2026 March Madness predictions using pre-tournament stats (no data leak).
Stats source: KenPom/DeepMetric adjusted efficiencies as of March 11, 2026.
Model: GradientBoosting trained on 2013-2023 tournament games.
"""
import pickle
import numpy as np
import json
import os

# ── Load trained model ───────────────────────────────────────────────────────

with open("ncaa_data/model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURE_COLS = [
    "ADJOE", "ADJDE", "BARTHAG", "EFG_O", "EFG_D",
    "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD",
    "2P_O", "2P_D", "3P_O", "3P_D", "ADJ_T", "WAB",
]

# ── 2026 pre-tournament stats (from DeepMetricAnalytics, March 11 2026) ─────
# Format: team -> {ADJOE, ADJDE, Net, W, G, SEED}
# BARTHAG estimated from Net rating: sigmoid(net/15) scaled roughly
# EFG/TOR/ORB/etc. not available from this source -> use averages (100) so
# only ADJOE, ADJDE, BARTHAG, SEED drive the prediction (the top-4 features).

def estimate_barthag(adjoe, adjde):
    """Estimate BARTHAG from efficiency margin using log5."""
    margin = adjoe - adjde
    return 1.0 / (1.0 + 10 ** (-margin / 12.0))

def make_team(adjoe, adjde, seed, record_str=""):
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
        "SEED": seed,
    }

# All 64 teams from Andy Katz's projected bracket (NCAA.com, March 10 2026)
# Stats from DeepMetricAnalytics.com adjusted efficiencies (March 11 2026)
TEAMS_2026 = {
    # 1 seeds
    "Duke":           make_team(129.5, 86.6, 1, "29-2"),
    "Florida":        make_team(126.0, 90.6, 1, "25-6"),
    "Michigan":       make_team(130.8, 88.8, 1, "29-2"),
    "Arizona":        make_team(125.4, 88.5, 1, "29-2"),
    # 2 seeds
    "Michigan State":  make_team(122.2, 92.6, 2, "25-6"),
    "Houston":         make_team(123.5, 90.4, 2, "26-5"),
    "UConn":           make_team(122.9, 94.1, 2, "27-4"),
    "Illinois":        make_team(133.6, 97.8, 2, "24-7"),
    # 3 seeds
    "Iowa State":      make_team(124.5, 91.7, 3, "26-6"),
    "Nebraska":        make_team(119.7, 91.5, 3, "26-5"),
    "Purdue":          make_team(131.4, 100.0, 3, "23-8"),
    "Alabama":         make_team(129.0, 101.8, 3, "23-8"),
    # 4 seeds
    "Texas Tech":      make_team(125.7, 98.2, 4, "22-9"),
    "Virginia":        make_team(122.4, 95.9, 4, "27-4"),
    "Kansas":          make_team(118.7, 93.0, 4, "22-9"),
    "Gonzaga":         make_team(123.2, 90.5, 4, "30-3"),
    # 5 seeds
    "Arkansas":        make_team(128.6, 101.3, 5, "23-8"),
    "St. John's":      make_team(120.5, 94.9, 5, "25-6"),
    "Vanderbilt":      make_team(125.8, 99.3, 5, "24-7"),
    "Wisconsin":       make_team(126.1, 102.9, 5, "22-9"),
    # 6 seeds
    "Tennessee":       make_team(121.7, 94.7, 6, "21-10"),
    "Louisville":      make_team(124.8, 96.3, 6, "23-9"),
    "North Carolina":  make_team(122.1, 98.0, 6, "24-7"),
    "BYU":             make_team(125.8, 101.6, 6, "23-10"),
    # 7 seeds
    "Villanova":       make_team(120.3, 99.7, 7, "24-7"),
    "Saint Mary's":    make_team(121.5, 95.9, 7, "27-5"),
    "Kentucky":        make_team(122.3, 98.8, 7, "20-12"),
    "Miami (Fla.)":    make_team(121.9, 100.3, 7, "24-7"),
    # 8 seeds
    "Georgia":         make_team(125.5, 102.6, 8, "22-9"),
    "Utah State":      make_team(124.3, 101.9, 8, "25-6"),
    "Clemson":         make_team(116.8, 96.6, 8, "23-9"),
    "UCLA":            make_team(123.6, 101.8, 8, "21-10"),
    # 9 seeds
    "Ohio State":      make_team(125.4, 102.5, 9, "20-11"),
    "Iowa":            make_team(123.3, 98.9, 9, "21-11"),
    "TCU":             make_team(116.6, 99.3, 9, "22-10"),
    "Texas A&M":       make_team(120.5, 101.6, 9, "21-10"),
    # 10 seeds
    "Miami (Ohio)":    make_team(117.7, 107.3, 10, "31-0"),
    "NC State":        make_team(124.9, 103.9, 10, "20-12"),
    "UCF":             make_team(120.9, 106.6, 10, "21-10"),
    "Saint Louis":     make_team(122.3, 99.8, 10, "27-4"),
    # 11 seeds (taking first listed for play-in spots)
    "SMU":             make_team(123.4, 103.8, 11, "20-13"),
    "Missouri":        make_team(119.3, 104.7, 11, "20-11"),
    "Texas":           make_team(124.5, 106.3, 11, "18-14"),
    "Santa Clara":     make_team(123.1, 103.5, 11, "26-8"),
    # 12 seeds
    "South Florida":   make_team(119.0, 101.7, 12, "23-8"),
    "High Point":      make_team(118.6, 107.0, 12, "30-4"),
    "Yale":            make_team(120.3, 109.0, 12, "23-5"),
    "Liberty":         make_team(117.3, 112.3, 12, "25-7"),
    # 13 seeds
    "Hofstra":         make_team(114.4, 105.1, 13, "24-10"),
    "Northern Iowa":   make_team(110.2, 96.9, 13, "23-12"),
    "Stephen F. Austin": make_team(113.2, 104.3, 13, "28-5"),
    "Utah Valley":     make_team(113.0, 103.5, 13, "24-7"),
    # 14 seeds
    "North Dakota State": make_team(111.6, 106.5, 14, "27-7"),
    "UC Irvine":       make_team(103.7, 98.4, 14, "22-10"),
    "Merrimack":       make_team(106.2, 108.1, 14, "23-11"),
    "Troy":            make_team(110.6, 108.6, 14, "22-11"),
    # 15 seeds
    "Wright State":    make_team(111.6, 109.5, 15, "23-11"),
    "Tennessee State": make_team(108.9, 110.5, 15, "23-9"),
    "Queens":          make_team(116.1, 118.1, 15, "21-13"),
    "Portland State":  make_team(104.5, 102.4, 15, "20-11"),
    # 16 seeds
    "Bethune-Cookman":  make_team(104.0, 109.0, 16, "18-14"),
    "Furman":           make_team(107.3, 108.8, 16, "22-12"),
    "Howard":           make_team(104.2, 107.0, 16, "21-10"),
    "UMBC":             make_team(107.6, 110.7, 16, "23-8"),
}

# ── Bracket structure (Katz projection, NCAA.com March 10) ───────────────────

REGIONS = {
    "East": [
        (1, "Duke"), (16, "Bethune-Cookman"),
        (8, "Georgia"), (9, "Ohio State"),
        (5, "Arkansas"), (12, "South Florida"),
        (4, "Texas Tech"), (13, "Hofstra"),
        (6, "Tennessee"), (11, "SMU"),
        (3, "Iowa State"), (14, "North Dakota State"),
        (7, "Villanova"), (10, "Miami (Ohio)"),
        (2, "Michigan State"), (15, "Wright State"),
    ],
    "South": [
        (1, "Florida"), (16, "Furman"),
        (8, "Utah State"), (9, "Iowa"),
        (5, "St. John's"), (12, "High Point"),
        (4, "Virginia"), (13, "Northern Iowa"),
        (6, "Louisville"), (11, "Missouri"),
        (3, "Nebraska"), (14, "UC Irvine"),
        (7, "Saint Mary's"), (10, "NC State"),
        (2, "Houston"), (15, "Tennessee State"),
    ],
    "Midwest": [
        (1, "Michigan"), (16, "Howard"),
        (8, "Clemson"), (9, "TCU"),
        (5, "Vanderbilt"), (12, "Yale"),
        (4, "Kansas"), (13, "Stephen F. Austin"),
        (6, "North Carolina"), (11, "Texas"),
        (3, "Purdue"), (14, "Merrimack"),
        (7, "Kentucky"), (10, "UCF"),
        (2, "UConn"), (15, "Queens"),
    ],
    "West": [
        (1, "Arizona"), (16, "UMBC"),
        (8, "UCLA"), (9, "Texas A&M"),
        (5, "Wisconsin"), (12, "Liberty"),
        (4, "Gonzaga"), (13, "Utah Valley"),
        (6, "BYU"), (11, "Santa Clara"),
        (3, "Alabama"), (14, "Troy"),
        (7, "Miami (Fla.)"), (10, "Saint Louis"),
        (2, "Illinois"), (15, "Portland State"),
    ],
}

# ── Predict ──────────────────────────────────────────────────────────────────

def predict_matchup(team_a, seed_a, team_b, seed_b):
    fa = TEAMS_2026[team_a]
    fb = TEAMS_2026[team_b]
    diff = np.array([[fa.get(c, 0) - fb.get(c, 0) for c in FEATURE_COLS + ["SEED"]]], dtype=np.float32)
    return float(model.predict_proba(diff)[0][1])

def sim_region(teams):
    bracket = list(teams)
    rounds_data = []
    while len(bracket) > 1:
        next_round = []
        round_matchups = []
        for i in range(0, len(bracket), 2):
            sa, ta = bracket[i]
            sb, tb = bracket[i + 1]
            prob_a = predict_matchup(ta, sa, tb, sb)
            matchup = {
                "team_a": ta, "seed_a": sa,
                "team_b": tb, "seed_b": sb,
                "prob_a": round(prob_a, 3),
                "prob_b": round(1 - prob_a, 3),
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
    return rounds_data, bracket[0]

print("=" * 70)
print("  2026 MARCH MADNESS PREDICTIONS (Pre-Tournament Stats, No Leak)")
print("  Model: GradientBoosting trained on 2013-2023 tournament data")
print("  Stats: DeepMetricAnalytics adjusted efficiencies, March 11 2026")
print("=" * 70)

bracket_output = {"year": 2026, "regions": {}, "final_four": [], "championship": None, "champion": None}

final_four = []
for region_name, teams in REGIONS.items():
    rounds, winner = sim_region(teams)
    bracket_output["regions"][region_name] = {
        "teams": [{"seed": s, "name": t} for s, t in teams],
        "rounds": rounds,
        "winner": {"seed": winner[0], "name": winner[1]},
    }
    final_four.append(winner)
    print(f"\n  {region_name} Region Winner: ({winner[0]}) {winner[1]}")
    for ri, rd in enumerate(rounds):
        rname = ["R64", "R32", "S16", "E8"][ri] if ri < 4 else f"R{ri}"
        picks = [f"({m['pick_seed']}){m['pick']}" for m in rd]
        print(f"    {rname}: {', '.join(picks)}")

print("\n" + "-" * 70)
print("  FINAL FOUR")
print("-" * 70)
ff_matchups = []
for i in range(0, len(final_four), 2):
    sa, ta = final_four[i]
    sb, tb = final_four[i + 1]
    prob_a = predict_matchup(ta, sa, tb, sb)
    pick = ta if prob_a >= 0.5 else tb
    pick_seed = sa if prob_a >= 0.5 else sb
    ff_matchups.append({
        "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
        "prob_a": round(prob_a, 3), "prob_b": round(1 - prob_a, 3),
        "pick": pick, "pick_seed": pick_seed,
        "pick_prob": round(max(prob_a, 1 - prob_a), 3),
    })
    print(f"  ({sa}) {ta:20s} {prob_a*100:5.1f}%  vs  ({sb}) {tb:20s} {(1-prob_a)*100:5.1f}%  -> {pick}")
    if prob_a >= 0.5:
        final_four[i // 2] = (sa, ta)
    else:
        final_four[i // 2] = (sb, tb)

bracket_output["final_four"] = ff_matchups
finalists = final_four[:2]

print("\n" + "-" * 70)
print("  CHAMPIONSHIP")
print("-" * 70)
sa, ta = finalists[0]
sb, tb = finalists[1]
prob_a = predict_matchup(ta, sa, tb, sb)
pick = ta if prob_a >= 0.5 else tb
pick_seed = sa if prob_a >= 0.5 else sb
bracket_output["championship"] = {
    "team_a": ta, "seed_a": sa, "team_b": tb, "seed_b": sb,
    "prob_a": round(prob_a, 3), "prob_b": round(1 - prob_a, 3),
    "pick": pick, "pick_seed": pick_seed,
    "pick_prob": round(max(prob_a, 1 - prob_a), 3),
}
bracket_output["champion"] = {"seed": pick_seed, "name": pick}
print(f"  ({sa}) {ta:20s} {prob_a*100:5.1f}%  vs  ({sb}) {tb:20s} {(1-prob_a)*100:5.1f}%")

print("\n" + "=" * 70)
print(f"  PREDICTED CHAMPION:  ({pick_seed}) {pick}")
print("=" * 70)

os.makedirs("ncaa_data", exist_ok=True)
with open("ncaa_data/bracket_preds_2026.json", "w") as f:
    json.dump(bracket_output, f, indent=2)
print(f"\nSaved: ncaa_data/bracket_preds_2026.json")

# Also copy to React app
react_path = "bracket-app/src/bracket_preds.json"
if os.path.isdir("bracket-app/src"):
    with open(react_path, "w") as f:
        json.dump(bracket_output, f, indent=2)
    print(f"Updated React app: {react_path}")
