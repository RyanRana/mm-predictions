"""
STEP 2: Run this AFTER the R script finishes.
Combines yearly CSVs into single files using chunked reading
so it won't blow up your RAM.
Option: --embed writes lossy-compressed ncaa_data/embedded.json (IDs, 1-decimal stats).

Usage: python ncaa_combine_csvs.py [--embed]

Requires: pip install pandas
"""
import pandas as pd
import glob
import os
import json
import sys

def combine_chunked(input_pattern, output_file):
    """Combine CSVs without loading all into memory at once."""
    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"  No files matching {input_pattern}")
        return

    first = True
    total_rows = 0
    for f in files:
        chunk = pd.read_csv(f, low_memory=False)
        total_rows += len(chunk)
        chunk.to_csv(output_file, mode='w' if first else 'a',
                     header=first, index=False)
        first = False
        del chunk

    print(f"  {output_file}: {total_rows:,} rows from {len(files)} files")

print("=== Combining player box scores ===")
combine_chunked("ncaa_data/player_box/player_box_*.csv",
                "ncaa_data/ALL_player_boxscores.csv")

print("=== Combining team box scores ===")
combine_chunked("ncaa_data/team_box/team_box_*.csv",
                "ncaa_data/ALL_team_boxscores.csv")

print("=== Combining schedules ===")
combine_chunked("ncaa_data/schedules/schedule_*.csv",
                "ncaa_data/ALL_schedules.csv")

print("\nDone! Combined files are in ncaa_data/")

# Show final sizes
for f in glob.glob("ncaa_data/ALL_*.csv"):
    size_mb = os.path.getsize(f) / 1e6
    print(f"  {os.path.basename(f)}: {size_mb:.0f} MB")

# Optional: write lossy embedded payload (IDs, low precision) for small/offline use
if "--embed" in sys.argv:
    try:
        from ncaa_embedded_data import build_embedded
        data = build_embedded(from_ncaa_data_dir=True)
        n = len(data.get("teams", [])) + len(data.get("games", [])) + len(data.get("team_stats", []))
        if n == 0:
            data = build_embedded(from_ncaa_data_dir=False)
        print(f"\nEmbedded ncaa_data/embedded.json written ({len(data.get('teams',[]))} teams, {len(data.get('games',[]))} games, {len(data.get('team_stats',[]))} team_stats).")
    except Exception as e:
        print("\nEmbed skipped:", e)
