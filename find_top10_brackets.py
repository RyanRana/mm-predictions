"""
Find the top-25 most probable complete 2026 March Madness brackets via beam search.
Joint probability = product of all 63 game-winner probabilities along a path.
Uses model_v2.pkl (GBM ensemble) + TEAMS_2026 / REGIONS_2026 from predict_multi_year.py.
"""
import csv, json, os, pickle, warnings
from collections import defaultdict
from math import log, exp
import numpy as np

warnings.filterwarnings("ignore")

os.makedirs("ncaa_data", exist_ok=True)

# ─── Load model ───────────────────────────────────────────────────────────────
with open("ncaa_data/model_v2.pkl", "rb") as f:
    bundle = pickle.load(f)

gbm    = bundle["gbm"]
rf     = bundle["rf"]
lr     = bundle["lr"]
nn     = bundle["nn"]
scaler = bundle["scaler"]
meta   = bundle["meta"]

# ─── Name map & constants ─────────────────────────────────────────────────────
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
    return NAME_MAP.get(name.strip(), name.strip())

BASE_COLS = [
    "ADJOE","ADJDE","BARTHAG","EFG_O","EFG_D",
    "TOR","TORD","ORB","DRB","FTR","FTRD",
    "2P_O","2P_D","3P_O","3P_D","ADJ_T","WAB",
]

# ─── Seed priors from historical data ─────────────────────────────────────────
tourney_games = []
with open("march_madness_games_1985_2024.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        yr = int(row["year"])
        rnd = int(row["round_of"])
        w = norm(row["winning_team_name"])
        l = norm(row["losing_team_name"])
        ws = int(row.get("winning_team_seed", 16) or 16)
        ls = int(row.get("losing_team_seed", 16) or 16)
        tourney_games.append((yr, rnd, w, l, ws, ls))

hist = defaultdict(lambda: [0, 0])
for yr, rnd, w, l, ws, ls in tourney_games:
    key = (min(ws, ls), max(ws, ls))
    if ws <= ls:
        hist[key][0] += 1
    else:
        hist[key][1] += 1

def seed_prior(sa, sb):
    lo, hi = min(sa, sb), max(sa, sb)
    wins, losses = hist[(lo, hi)]
    t = wins + losses
    if t < 3:
        return 0.5
    p = (wins + 1) / (t + 2)
    return p if sa <= sb else (1 - p)

# ─── Features ─────────────────────────────────────────────────────────────────
def make_features(fa, fb, sa, sb):
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
    f.append(seed_prior(sa, sb))
    wa = fa.get("W", 20) / max(fa.get("G", 30), 1)
    wb = fb.get("W", 20) / max(fb.get("G", 30), 1)
    f.append(wa - wb)
    ba = max(fa.get("BARTHAG", 0.5), 0.01)
    bb = max(fb.get("BARTHAG", 0.5), 0.01)
    f.append(log(ba / bb))
    return f

# Cache matchup probabilities to avoid recomputing
_prob_cache = {}

def matchup_prob(ta, sa, tb, sb):
    key = (ta, sa, tb, sb)
    if key in _prob_cache:
        return _prob_cache[key]
    fa, fb = TEAMS_2026[ta], TEAMS_2026[tb]
    feats = make_features(fa, fb, sa, sb)
    X = np.array([feats], dtype=np.float32)
    X_sc = scaler.transform(X)
    meta_X = np.column_stack([
        gbm.predict_proba(X)[:, 1],
        rf.predict_proba(X)[:, 1],
        lr.predict_proba(X_sc)[:, 1],
        nn.predict_proba(X_sc)[:, 1],
    ])
    p_ens = float(meta.predict_proba(meta_X)[0][1])
    sp = seed_prior(sa, sb)
    p = p_ens * 0.85 + sp * 0.15
    p = max(0.001, min(0.999, p))
    _prob_cache[key] = p
    return p

# ─── Team stats ───────────────────────────────────────────────────────────────
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
        (1,"Duke"),(16,"Bethune-Cookman"),(8,"Georgia"),(9,"Ohio State"),
        (5,"Arkansas"),(12,"South Florida"),(4,"Texas Tech"),(13,"Hofstra"),
        (6,"Tennessee"),(11,"SMU"),(3,"Iowa State"),(14,"North Dakota State"),
        (7,"Villanova"),(10,"Miami (Ohio)"),(2,"Michigan State"),(15,"Wright State"),
    ],
    "South": [
        (1,"Florida"),(16,"Furman"),(8,"Utah State"),(9,"Iowa"),
        (5,"St. John's"),(12,"High Point"),(4,"Virginia"),(13,"Northern Iowa"),
        (6,"Louisville"),(11,"Missouri"),(3,"Nebraska"),(14,"UC Irvine"),
        (7,"Saint Mary's"),(10,"NC State"),(2,"Houston"),(15,"Tennessee State"),
    ],
    "Midwest": [
        (1,"Michigan"),(16,"Howard"),(8,"Clemson"),(9,"TCU"),
        (5,"Vanderbilt"),(12,"Yale"),(4,"Kansas"),(13,"Stephen F. Austin"),
        (6,"North Carolina"),(11,"Texas"),(3,"Purdue"),(14,"Merrimack"),
        (7,"Kentucky"),(10,"UCF"),(2,"UConn"),(15,"Queens"),
    ],
    "West": [
        (1,"Arizona"),(16,"UMBC"),(8,"UCLA"),(9,"Texas A&M"),
        (5,"Wisconsin"),(12,"Liberty"),(4,"Gonzaga"),(13,"Utah Valley"),
        (6,"BYU"),(11,"Santa Clara"),(3,"Alabama"),(14,"Troy"),
        (7,"Miami (Fla.)"),(10,"Saint Louis"),(2,"Illinois"),(15,"Portland State"),
    ],
}

# FF pairing: East vs South, Midwest vs West
FF_PAIRS = [("East", "South"), ("Midwest", "West")]


# ─── Enumerate top-K bracket paths for a region ───────────────────────────────
def enum_region_top_k(rname, teams, K=50):
    """
    Returns list of K dicts:
      { log_prob, winner: (seed, name), rounds: [round0_matchups, round1_matchups, ...] }
    """
    # State: list of (log_prob, alive_teams [(seed,name),...], rounds_so_far)
    # alive_teams starts as the seeded bracket order
    init_alive = list(teams)
    states = [(0.0, init_alive, [])]

    while states and len(states[0][2]) < 4:  # 4 rounds per region
        next_states = []
        for lp, alive, rounds in states:
            n = len(alive)
            # Play one round: pair teams in order
            # Generate all 2^(n/2) outcomes for this round
            round_outcomes = [{"lp": 0.0, "new_alive": [], "matchups": []}]
            for i in range(0, n, 2):
                sa, ta = alive[i]
                sb, tb = alive[i + 1]
                pa = matchup_prob(ta, sa, tb, sb)
                pb = 1.0 - pa
                new_outcomes = []
                for outcome in round_outcomes:
                    m_base = {
                        "team_a": ta, "seed_a": sa,
                        "team_b": tb, "seed_b": sb,
                        "prob_a": round(pa, 3), "prob_b": round(pb, 3),
                    }
                    # Branch A wins
                    new_outcomes.append({
                        "lp": outcome["lp"] + log(pa),
                        "new_alive": outcome["new_alive"] + [(sa, ta)],
                        "matchups": outcome["matchups"] + [{
                            **m_base, "pick": ta, "pick_seed": sa,
                            "pick_prob": round(pa, 3),
                        }],
                    })
                    # Branch B wins
                    new_outcomes.append({
                        "lp": outcome["lp"] + log(pb),
                        "new_alive": outcome["new_alive"] + [(sb, tb)],
                        "matchups": outcome["matchups"] + [{
                            **m_base, "pick": tb, "pick_seed": sb,
                            "pick_prob": round(pb, 3),
                        }],
                    })
                round_outcomes = new_outcomes

            for oc in round_outcomes:
                next_states.append((lp + oc["lp"], oc["new_alive"], rounds + [oc["matchups"]]))

        # Prune: keep top-K by log prob
        next_states.sort(key=lambda x: -x[0])
        states = next_states[:K]

    results = []
    for lp, alive, rounds in states:
        winner = alive[0]
        results.append({
            "log_prob": lp,
            "winner": {"seed": winner[0], "name": winner[1]},
            "rounds": rounds,
            "teams": [{"seed": s, "name": t} for s, t in teams],
        })
    return results


# ─── Main beam search ─────────────────────────────────────────────────────────
K_REGION = 60  # top paths to keep per region

print("Enumerating top regional bracket paths...")
region_paths = {}
for rname, teams in REGIONS_2026.items():
    paths = enum_region_top_k(rname, teams, K=K_REGION)
    region_paths[rname] = paths
    print(f"  {rname}: {len(paths)} paths, best lp={paths[0]['log_prob']:.3f}  "
          f"winner={paths[0]['winner']['name']}")

# ─── Combine for Final Four ────────────────────────────────────────────────────
print("\nCombining Final Four pairs...")

def combine_ff_pair(r1_paths, r2_paths, K=60):
    """Combine two regional path lists, play FF game, return top-K."""
    candidates = []
    for p1 in r1_paths:
        for p2 in r2_paths:
            w1 = p1["winner"]
            w2 = p2["winner"]
            pa = matchup_prob(w1["name"], w1["seed"], w2["name"], w2["seed"])
            pb = 1.0 - pa
            m = {
                "team_a": w1["name"], "seed_a": w1["seed"],
                "team_b": w2["name"], "seed_b": w2["seed"],
                "prob_a": round(pa, 3), "prob_b": round(pb, 3),
            }
            for winner, lp_game in [(w1, log(pa)), (w2, log(pb))]:
                pick_prob = pa if winner == w1 else pb
                candidates.append({
                    "log_prob": p1["log_prob"] + p2["log_prob"] + lp_game,
                    "ff_game": {**m, "pick": winner["name"], "pick_seed": winner["seed"],
                                "pick_prob": round(pick_prob, 3)},
                    "winner": winner,
                    "paths": (p1, p2),
                })
    candidates.sort(key=lambda x: -x["log_prob"])
    return candidates[:K]

pair_A = combine_ff_pair(region_paths["East"],    region_paths["South"],   K=K_REGION)
pair_B = combine_ff_pair(region_paths["Midwest"],  region_paths["West"],    K=K_REGION)
print(f"  East/South pair: {len(pair_A)} candidates")
print(f"  Midwest/West pair: {len(pair_B)} candidates")

# ─── Championship ─────────────────────────────────────────────────────────────
print("Computing championship matchups...")
finals = []
for cA in pair_A:
    for cB in pair_B:
        wA = cA["winner"]
        wB = cB["winner"]
        pa = matchup_prob(wA["name"], wA["seed"], wB["name"], wB["seed"])
        pb = 1.0 - pa
        m = {
            "team_a": wA["name"], "seed_a": wA["seed"],
            "team_b": wB["name"], "seed_b": wB["seed"],
            "prob_a": round(pa, 3), "prob_b": round(pb, 3),
        }
        for winner, lp_game in [(wA, log(pa)), (wB, log(pb))]:
            pick_prob = pa if winner == wA else pb
            finals.append({
                "log_prob": cA["log_prob"] + cB["log_prob"] + lp_game,
                "championship": {**m, "pick": winner["name"], "pick_seed": winner["seed"],
                                  "pick_prob": round(pick_prob, 3)},
                "champion": winner,
                "pair_A": cA,
                "pair_B": cB,
            })

finals.sort(key=lambda x: -x["log_prob"])
top25 = finals[:25]
print(f"  Total final candidates: {len(finals)}")
print(f"  Top-25 selected.")

# ─── Assemble output ──────────────────────────────────────────────────────────
def assemble_bracket(entry, rank):
    cA = entry["pair_A"]
    cB = entry["pair_B"]
    pE, pS = cA["paths"]
    pM, pW = cB["paths"]

    region_map = {
        "East": pE, "South": pS, "Midwest": pM, "West": pW,
    }

    regions_out = {}
    for rname, rp in region_map.items():
        regions_out[rname] = {
            "teams": rp["teams"],
            "rounds": rp["rounds"],
            "winner": rp["winner"],
        }

    # joint probability = exp(log_prob)
    joint_prob = exp(entry["log_prob"])
    # log_prob is the sum over all 63 games
    # average per-game confidence
    avg_conf = exp(entry["log_prob"] / 63)

    return {
        "rank": rank,
        "log_prob": round(entry["log_prob"], 4),
        "joint_prob_pct": round(joint_prob * 100, 8),  # tiny %, but useful for ranking
        "avg_game_confidence": round(avg_conf * 100, 1),
        "champion": entry["champion"],
        "championship": entry["championship"],
        "final_four": [cA["ff_game"], cB["ff_game"]],
        "regions": regions_out,
        "year": 2026,
    }

brackets = [assemble_bracket(entry, i + 1) for i, entry in enumerate(top25)]

# ─── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TOP 25 MOST PROBABLE 2026 BRACKETS")
print("=" * 70)
for b in brackets:
    champ = b["champion"]
    ff_winners = [b["regions"][r]["winner"]["name"] for r in ["East", "South", "Midwest", "West"]]
    print(f"  #{b['rank']}  lp={b['log_prob']:.3f}  avg_conf={b['avg_game_confidence']}%  "
          f"Champ=({champ['seed']}){champ['name']}"
          f"  FF={', '.join(ff_winners)}")

out_path = "ncaa_data/top25_brackets_2026.json"
with open(out_path, "w") as f:
    json.dump(brackets, f, indent=2)
print(f"\nSaved: {out_path}  ({os.path.getsize(out_path)//1024} KB)")
