"""
Lossy-compressed NCAA data: IDs for teams, minimal columns, rounded stats.
Use when full CSVs aren't needed. Load from embedded payload or from
ncaa_data_download.R / ncaa_torvik_download.R output.

Schema:
  teams: list of [id, short_name]  (id = 1..n)
  games: list of [year, round, w_id, l_id, w_pts, l_pts]  (round = 64|32|16|8|4|2|1)
  team_stats: list of [id, year, g, w, adjoe, adjde, barthag]  (1 decimal)

  When built for years 21–25 (TARGET_YEARS), also writes ncaa_data/embedded_21_25.json with:
  years: [2021,2022,2023,2024,2025]
  by_year: {"2021": {"games": [...], "team_stats": [...]}, "2022": {...}, ...}
  Same rounding: adjoe/adjde 1 decimal, barthag 3 decimals.
"""
import json
import os
import glob

# Seasons to build when using organized-by-year output (21, 22, 23, 24, 25)
TARGET_YEARS = [2021, 2022, 2023, 2024, 2025]

# --- Fallback minimal payload when ncaa_data/embedded.json missing ---
# Schema: teams [id, short_name], games [y,r,w_id,l_id,ws,ls], team_stats [id,y,g,w,adjoe,adjde,barthag]
EMBEDDED_JSON = r'{"v":1,"teams":[[1,"UConn"],[2,"Stetson"]],"games":[[2024,64,1,2,91,52]],"team_stats":[[1,2024,40,37,118.2,88.1,0.98]]}'

def _build_from_csvs():
    """Build compressed payload from repo CSVs. Run once to regenerate EMBEDDED_JSON."""
    import csv
    teams_d = {}
    teams_list = []
    def tid(name):
        name = (name or "").strip()
        if name not in teams_d:
            teams_d[name] = len(teams_list) + 1
            teams_list.append([teams_d[name], name[:20]])
        return teams_d[name]

    games = []
    path_mm = "march_madness_games_1985_2024.csv"
    if os.path.isfile(path_mm):
        with open(path_mm, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                y = int(row.get("year", 0))
                rnd = int(row.get("round_of", 0))
                w_id = tid(row.get("winning_team_name", ""))
                l_id = tid(row.get("losing_team_name", ""))
                ws = int(row.get("winning_team_score", 0))
                ls = int(row.get("losing_team_score", 0))
                games.append([y, rnd, w_id, l_id, ws, ls])

    team_stats = []
    path_ts = "team_season_stats_2013_2023.csv"
    if os.path.isfile(path_ts):
        with open(path_ts, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                id_ = tid(row.get("TEAM", ""))
                y = int(row.get("YEAR", 0))
                g = int(row.get("G", 0))
                w = int(row.get("W", 0))
                adjoe = round(float(row.get("ADJOE", 0)), 1)
                adjde = round(float(row.get("ADJDE", 0)), 1)
                barthag = round(float(row.get("BARTHAG", 0)), 3)
                team_stats.append([id_, y, g, w, adjoe, adjde, barthag])

    return {"v": 1, "teams": teams_list, "games": games, "team_stats": team_stats}


def _organize_by_year(data, years):
    """Filter to given years and add by_year: { "2021": { games, team_stats }, ... }."""
    years_set = set(years)
    data["games"] = [g for g in data["games"] if g[0] in years_set]
    data["team_stats"] = [s for s in data["team_stats"] if s[1] in years_set]
    by_year = {}
    for y in sorted(years_set):
        by_year[str(y)] = {
            "games": [g for g in data["games"] if g[0] == y],
            "team_stats": [s for s in data["team_stats"] if s[1] == y],
        }
    data["by_year"] = by_year
    data["years"] = sorted(years_set)
    return data


def build_embedded(from_ncaa_data_dir=False, years=None, out_path=None):
    """Write compressed JSON to ncaa_data/embedded.json (or out_path).
    If years is set (e.g. TARGET_YEARS), filter to those years and add by_year for organized access.
    """
    if from_ncaa_data_dir:
        data = _build_from_ncaa_data()
    else:
        data = _build_from_csvs()
    if years:
        data = _organize_by_year(data, years)
    os.makedirs("ncaa_data", exist_ok=True)
    path = out_path or "ncaa_data/embedded.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    return data


def _build_from_ncaa_data():
    """Build from ncaa_data/ ALL_schedules, ALL_team_boxscores, torvik team_factors (lossy: IDs, 1 decimal)."""
    import csv
    teams_d = {}
    teams_list = []

    def tid(name):
        name = (str(name or "").strip())[:20]
        if name not in teams_d:
            teams_d[name] = len(teams_list) + 1
            teams_list.append([teams_d[name], name])
        return teams_d[name]

    games = []
    for path in ["ncaa_data/ALL_schedules.csv", "ncaa_data/schedules/schedule_2024.csv"]:
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                y = int(row.get("season", row.get("year", 0)) or 0)
                if y < 2000:
                    continue
                # hoopR schedule: home/away or team_*; infer winner/loser from score
                home = row.get("home_team_name", row.get("home_team", ""))
                away = row.get("away_team_name", row.get("away_team", ""))
                hs = int(row.get("home_score", row.get("home_team_score", 0)) or 0)
                as_ = int(row.get("away_score", row.get("away_team_score", 0)) or 0)
                if not home and not away:
                    continue
                hid, aid = tid(home), tid(away)
                if hs >= as_:
                    games.append([y, 0, hid, aid, hs, as_])
                else:
                    games.append([y, 0, aid, hid, as_, hs])
        break

    team_stats = []
    for path in glob.glob("ncaa_data/torvik/team_factors_*.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                id_ = tid(row.get("team", row.get("team_name", "")))
                y = int(row.get("year", row.get("season", 0)) or 0)
                g = int(row.get("g", row.get("G", 0)) or 0)
                w = int(row.get("w", row.get("W", 0)) or 0)
                adjoe = round(float(row.get("adjoe", row.get("ADJOE", 0)) or 0), 1)
                adjde = round(float(row.get("adjde", row.get("ADJDE", 0)) or 0), 1)
                barthag = round(float(row.get("barthag", row.get("BARTHAG", 0)) or 0), 3)
                team_stats.append([id_, y, g, w, adjoe, adjde, barthag])

    if not games and os.path.isfile("ncaa_data/espn_teams.csv"):
        with open("ncaa_data/espn_teams.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid(row.get("display_name", row.get("team", "")))
    return {"v": 1, "teams": teams_list, "games": games, "team_stats": team_stats}


def load_embedded(json_str=None, path=None):
    """Load from path, or ncaa_data/embedded.json, or json_str, or fallback to EMBEDDED_JSON."""
    if json_str is not None:
        return json.loads(json_str)
    for p in (path, "ncaa_data/embedded_21_25.json", "ncaa_data/embedded.json"):
        if p and os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return json.loads(EMBEDDED_JSON)


def get_team_id_map(payload=None):
    """Return dict short_name -> id and dict id -> short_name."""
    p = payload or load_embedded()
    id2name = {t[0]: t[1] for t in p["teams"]}
    name2id = {t[1]: t[0] for t in p["teams"]}
    return name2id, id2name


if __name__ == "__main__":
    # Full dataset
    data = build_embedded()
    n_teams = len(data["teams"])
    n_games = len(data["games"])
    n_stats = len(data["team_stats"])
    print(f"embedded.json: {n_teams} teams, {n_games} games, {n_stats} team_stats")

    # 21–25 only, organized by year (same rounding/schema)
    data_21_25 = build_embedded(years=TARGET_YEARS, out_path="ncaa_data/embedded_21_25.json")
    print(f"embedded_21_25.json: years {data_21_25.get('years', [])}")
    for y in data_21_25.get("by_year", {}):
        g = len(data_21_25["by_year"][y]["games"])
        s = len(data_21_25["by_year"][y]["team_stats"])
        print(f"  {y}: {g} games, {s} team_stats")
