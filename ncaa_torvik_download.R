# ============================================================
# OPTIONAL: BartTorvik Advanced Stats via cbbdata
# ============================================================
# Run this SEPARATELY after the main download.
# Requires a free cbbdata account.
#
# FIRST TIME ONLY - uncomment and run:
#   devtools::install_github("andreweatherman/cbbdata")
#   cbbdata::cbd_create_account(username="you", password="pass")
#
# After that, just run this script.
# ============================================================

library(cbbdata)

dir.create("ncaa_data/torvik", showWarnings = FALSE)

# Team-level advanced stats (ADJOE, ADJDE, BARTHAG, etc.)
cat("=== BartTorvik Team Stats ===\n")
for (year in 2008:2025) {
  cat(sprintf("  Team factors %d... ", year))
  tryCatch({
    df <- cbd_torvik_team_factors(year = year)
    write.csv(df, sprintf("ncaa_data/torvik/team_factors_%d.csv", year),
              row.names = FALSE)
    cat(sprintf("OK (%d teams)\n", nrow(df)))
    rm(df); gc()
  }, error = function(e) cat(sprintf("SKIP - %s\n", e$message)))
  Sys.sleep(0.5)
}

# Player season stats
cat("\n=== BartTorvik Player Season Stats ===\n")
for (year in 2008:2025) {
  cat(sprintf("  Players %d... ", year))
  tryCatch({
    df <- cbd_torvik_player_season(year = year)
    write.csv(df, sprintf("ncaa_data/torvik/players_%d.csv", year),
              row.names = FALSE)
    cat(sprintf("OK (%d players)\n", nrow(df)))
    rm(df); gc()
  }, error = function(e) cat(sprintf("SKIP - %s\n", e$message)))
  Sys.sleep(0.5)
}

# Game results
cat("\n=== BartTorvik Game Results ===\n")
for (year in 2008:2025) {
  cat(sprintf("  Games %d... ", year))
  tryCatch({
    df <- cbd_torvik_game_stats(year = year)
    write.csv(df, sprintf("ncaa_data/torvik/games_%d.csv", year),
              row.names = FALSE)
    cat(sprintf("OK (%d games)\n", nrow(df)))
    rm(df); gc()
  }, error = function(e) cat(sprintf("SKIP - %s\n", e$message)))
  Sys.sleep(0.5)
}

# -----------------------------------------------------------
# Optional: merge lossy team_stats into ncaa_data/embedded.json (IDs, 1 decimal)
# Run after ncaa_data_download.R to get one file with teams + games + team_stats.
# -----------------------------------------------------------
if (requireNamespace("jsonlite", quietly = TRUE)) {
  cat("\n=== Updating embedded.json with Torvik team_stats ===\n")
  tryCatch({
    payload <- list(v = 1L, teams = list(), games = list(), team_stats = list())
    embed_path <- "ncaa_data/embedded.json"
    if (file.exists(embed_path)) {
      payload <- jsonlite::fromJSON(embed_path, simplifyVector = FALSE)
    }
    name2id <- setNames(sapply(payload$teams, `[[`, 1), sapply(payload$teams, `[[`, 2))
    next_id <- max(0, sapply(payload$teams, `[[`, 1), na.rm = TRUE) + 1L
    tf_files <- list.files("ncaa_data/torvik", pattern = "team_factors_.*\\.csv$", full.names = TRUE)
    for (f in tf_files) {
      d <- read.csv(f, stringsAsFactors = FALSE)
      y <- if ("year" %in% names(d)) as.integer(d$year[1]) else as.integer(gsub(".*_(\\d+)\\.csv", "\\1", basename(f)))
      team_col <- names(d)[grepl("^team|team_name", names(d), ignore.case = TRUE)][1]
      if (is.na(team_col)) team_col <- names(d)[1]
      g_col <- names(d)[tolower(names(d)) == "g"][1]
      w_col <- names(d)[tolower(names(d)) == "w"][1]
      for (i in seq_len(nrow(d))) {
        tname <- substr(trimws(as.character(d[[team_col]][i])), 1, 20)
        if (tname == "") next
        if (!tname %in% names(name2id)) { name2id[[tname]] <- next_id; payload$teams[[length(payload$teams) + 1]] <- list(next_id, tname); next_id <- next_id + 1L }
        id <- name2id[[tname]]
        g <- if (!is.na(g_col) && g_col %in% names(d)) as.integer(d[[g_col]][i]) else 0L; if (is.na(g)) g <- 0L
        w <- if (!is.na(w_col) && w_col %in% names(d)) as.integer(d[[w_col]][i]) else 0L; if (is.na(w)) w <- 0L
        adjoe <- round(as.numeric(if ("adjoe" %in% names(d)) d$adjoe[i] else if ("ADJOE" %in% names(d)) d$ADJOE[i] else 0), 1)
        adjde <- round(as.numeric(if ("adjde" %in% names(d)) d$adjde[i] else if ("ADJDE" %in% names(d)) d$ADJDE[i] else 0), 1)
        barthag <- round(as.numeric(if ("barthag" %in% names(d)) d$barthag[i] else if ("BARTHAG" %in% names(d)) d$BARTHAG[i] else 0), 3)
        payload$team_stats[[length(payload$team_stats) + 1]] <- list(id, y, g, w, adjoe, adjde, barthag)
      }
    }
    writeLines(jsonlite::toJSON(payload, auto_unbox = TRUE), embed_path)
    cat(sprintf("  embedded.json: %d teams, %d games, %d team_stats\n",
                length(payload$teams), length(payload$games), length(payload$team_stats)))
  }, error = function(e) cat("  Embed skipped:", conditionMessage(e), "\n"))
}

cat("\nDone! Torvik data saved in ncaa_data/torvik/\n")
