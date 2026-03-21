# ============================================================
# NCAA BASKETBALL COMPREHENSIVE DATA DOWNLOADER
# ============================================================
# This script downloads data ONE YEAR AT A TIME to avoid
# crashing your machine. Each year saves to disk immediately,
# then gets cleared from memory before the next year loads.
#
# REQUIREMENTS:
#   install.packages(c("hoopR", "dplyr"))
#   # Optional for BartTorvik data:
#   devtools::install_github("andreweatherman/cbbdata")
#
# EXPECTED OUTPUT: ~500MB-1GB of CSV files total
# RAM USAGE: ~500MB peak (one season at a time)
# TIME: ~15-30 minutes depending on internet speed
# ============================================================

library(hoopR)

# Create output folder
dir.create("ncaa_data", showWarnings = FALSE)

# -----------------------------------------------------------
# 1. PLAYER BOX SCORES (every player, every game)
#    ~80,000-100,000 rows per season
# -----------------------------------------------------------
cat("=== DOWNLOADING PLAYER BOX SCORES ===\n")
dir.create("ncaa_data/player_box", showWarnings = FALSE)

for (year in 2003:2025) {
  cat(sprintf("  Player box %d... ", year))
  tryCatch({
    df <- load_mbb_player_box(seasons = year)
    write.csv(df, sprintf("ncaa_data/player_box/player_box_%d.csv", year),
              row.names = FALSE)
    cat(sprintf("OK (%d rows)\n", nrow(df)))
    rm(df)          # free memory immediately
    gc()            # force garbage collection
  }, error = function(e) {
    cat(sprintf("SKIP - %s\n", e$message))
  })
  Sys.sleep(1)      # be polite to the server
}

# -----------------------------------------------------------
# 2. TEAM BOX SCORES (every game, team-level stats)
#    ~10,000-12,000 rows per season
# -----------------------------------------------------------
cat("\n=== DOWNLOADING TEAM BOX SCORES ===\n")
dir.create("ncaa_data/team_box", showWarnings = FALSE)

for (year in 2003:2025) {
  cat(sprintf("  Team box %d... ", year))
  tryCatch({
    df <- load_mbb_team_box(seasons = year)
    write.csv(df, sprintf("ncaa_data/team_box/team_box_%d.csv", year),
              row.names = FALSE)
    cat(sprintf("OK (%d rows)\n", nrow(df)))
    rm(df); gc()
  }, error = function(e) {
    cat(sprintf("SKIP - %s\n", e$message))
  })
  Sys.sleep(1)
}

# -----------------------------------------------------------
# 3. SCHEDULES (game dates, matchups, locations)
# -----------------------------------------------------------
cat("\n=== DOWNLOADING SCHEDULES ===\n")
dir.create("ncaa_data/schedules", showWarnings = FALSE)

for (year in 2003:2025) {
  cat(sprintf("  Schedule %d... ", year))
  tryCatch({
    df <- load_mbb_schedule(seasons = year)
    write.csv(df, sprintf("ncaa_data/schedules/schedule_%d.csv", year),
              row.names = FALSE)
    cat(sprintf("OK (%d rows)\n", nrow(df)))
    rm(df); gc()
  }, error = function(e) {
    cat(sprintf("SKIP - %s\n", e$message))
  })
  Sys.sleep(1)
}

# -----------------------------------------------------------
# 4. CURRENT TEAMS LIST
# -----------------------------------------------------------
cat("\n=== DOWNLOADING TEAM LIST ===\n")
tryCatch({
  teams <- espn_mbb_teams()
  write.csv(teams, "ncaa_data/espn_teams.csv", row.names = FALSE)
  cat(sprintf("  Teams: %d\n", nrow(teams)))
}, error = function(e) cat(sprintf("  SKIP - %s\n", e$message)))

# -----------------------------------------------------------
# DONE - Print summary
# -----------------------------------------------------------
cat("\n============ DONE ============\n")
cat("Files saved in: ncaa_data/\n\n")

# Count what we got
files <- list.files("ncaa_data", recursive = TRUE, pattern = "\\.csv$")
total_size <- sum(file.size(file.path("ncaa_data", files))) / 1e6
cat(sprintf("Total files: %d\n", length(files)))
cat(sprintf("Total size:  %.0f MB\n", total_size))
cat("\nFolder structure:\n")
cat("  ncaa_data/player_box/   <- individual player stats per game\n")
cat("  ncaa_data/team_box/     <- team totals per game\n")
cat("  ncaa_data/schedules/    <- game dates & matchups\n")
cat("  ncaa_data/espn_teams.csv <- all D1 teams\n")

# -----------------------------------------------------------
# Optional: write lossy-compressed embedded.json (IDs, low data)
# Schema: teams [[id, short_name]], games [[y,r,w_id,l_id,ws,ls]]
# -----------------------------------------------------------
if (requireNamespace("jsonlite", quietly = TRUE)) {
  cat("\n=== Writing embedded.json (lossy: IDs, minimal fields) ===\n")
  tryCatch({
    teams_df <- read.csv("ncaa_data/espn_teams.csv", stringsAsFactors = FALSE)
    nm_col <- if ("display_name" %in% names(teams_df)) teams_df$display_name else if ("name" %in% names(teams_df)) teams_df$name else teams_df[[1]]
    team_names <- unique(substr(trimws(nm_col[!is.na(nm_col) & nm_col != ""]), 1, 20))
    teams_list <- lapply(seq_along(team_names), function(i) list(as.integer(i), team_names[i]))
    name2id <- setNames(as.integer(seq_along(team_names)), team_names)

    games_list <- list()
    sched_files <- list.files("ncaa_data/schedules", pattern = "schedule_.*\\.csv$", full.names = TRUE)
    for (f in sched_files) {
      d <- read.csv(f, stringsAsFactors = FALSE)
      y <- if ("season" %in% names(d)) as.integer(d$season[1]) else as.integer(gsub(".*_(\\d+)\\.csv", "\\1", basename(f)))
      hn <- names(d)[grepl("home.*(name|team|display)", names(d), ignore.case = TRUE)][1]
      an <- names(d)[grepl("away.*(name|team|display)", names(d), ignore.case = TRUE)][1]
      hs_col <- names(d)[grepl("home.*score", names(d), ignore.case = TRUE)][1]
      as_col <- names(d)[grepl("away.*score", names(d), ignore.case = TRUE)][1]
      if (is.na(hn) || is.na(an) || is.na(hs_col) || is.na(as_col)) next
      home_s <- as.integer(d[[hs_col]])
      away_s <- as.integer(d[[as_col]])
      home_s[is.na(home_s)] <- 0
      away_s[is.na(away_s)] <- 0
      hnames <- trimws(substr(as.character(d[[hn]]), 1, 20))
      anames <- trimws(substr(as.character(d[[an]]), 1, 20))
      for (i in seq_len(nrow(d))) {
        if (hnames[i] == "" && anames[i] == "") next
        if (!hnames[i] %in% names(name2id)) { name2id[[hnames[i]]] <- length(name2id) + 1; teams_list[[length(teams_list) + 1]] <- list(as.integer(name2id[[hnames[i]]]), hnames[i]) }
        if (!anames[i] %in% names(name2id)) { name2id[[anames[i]]] <- length(name2id) + 1; teams_list[[length(teams_list) + 1]] <- list(as.integer(name2id[[anames[i]]]), anames[i]) }
        hid <- as.integer(name2id[[hnames[i]]])
        aid <- as.integer(name2id[[anames[i]]])
        if (home_s[i] >= away_s[i]) games_list[[length(games_list) + 1]] <- list(y, 0L, hid, aid, as.integer(home_s[i]), as.integer(away_s[i]))
        else games_list[[length(games_list) + 1]] <- list(y, 0L, aid, hid, as.integer(away_s[i]), as.integer(home_s[i]))
      }
    }
    payload <- list(v = 1L, teams = teams_list, games = games_list, team_stats = list())
    writeLines(jsonlite::toJSON(payload, auto_unbox = TRUE), "ncaa_data/embedded.json")
    cat(sprintf("  embedded.json: %d teams, %d games\n", length(teams_list), length(games_list)))
  }, error = function(e) cat("  Embed skipped:", conditionMessage(e), "\n"))
} else {
  cat("\n  Install jsonlite to write ncaa_data/embedded.json: install.packages('jsonlite')\n")
}
