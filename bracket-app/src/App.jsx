import { useState, useMemo } from "react";
import bracketData from "./bracket_preds.json";
import { predictProba, getFeatureNames } from "./predictor.js";
import "./App.css";

const ROUND_NAMES = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"];

function Matchup({ game, roundIdx }) {
  const confidence = game.pick_prob;
  const isUpset = game.pick_seed > Math.min(game.seed_a, game.seed_b);
  return (
    <div className={`matchup ${isUpset ? "upset" : ""}`}>
      <Team
        name={game.team_a}
        seed={game.seed_a}
        prob={game.prob_a}
        isWinner={game.pick === game.team_a}
      />
      <Team
        name={game.team_b}
        seed={game.seed_b}
        prob={game.prob_b}
        isWinner={game.pick === game.team_b}
      />
      <div className="confidence-bar">
        <div
          className="confidence-fill"
          style={{ width: `${confidence * 100}%` }}
        />
        <span className="confidence-label">{(confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

function Team({ name, seed, prob, isWinner }) {
  return (
    <div className={`team-row ${isWinner ? "winner" : "loser"}`}>
      <span className="seed">{seed}</span>
      <span className="team-name">{name}</span>
      <span className="prob">{(prob * 100).toFixed(1)}%</span>
    </div>
  );
}

function Region({ name, data }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <div className="region">
      <h2 className="region-title" onClick={() => setExpanded(!expanded)}>
        {name}
        <span className="region-winner">
          → ({data.winner.seed}) {data.winner.name}
        </span>
        <span className="toggle">{expanded ? "▾" : "▸"}</span>
      </h2>
      {expanded && (
        <div className="rounds">
          {data.rounds.map((round, ri) => (
            <div key={ri} className="round">
              <h3 className="round-title">{ROUND_NAMES[ri] || `Round ${ri + 1}`}</h3>
              <div className="matchups">
                {round.map((game, gi) => (
                  <Matchup key={gi} game={game} roundIdx={ri} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function FinalFour({ data }) {
  return (
    <div className="final-four-section">
      <h2 className="section-title">Final Four</h2>
      <div className="matchups ff-matchups">
        {data.final_four.map((game, i) => (
          <Matchup key={i} game={game} roundIdx={4} />
        ))}
      </div>

      <h2 className="section-title">Championship</h2>
      <div className="matchups ff-matchups">
        <Matchup game={data.championship} roundIdx={5} />
      </div>

      <div className="champion-banner">
        <div className="champion-label">Predicted Champion</div>
        <div className="champion-name">
          <span className="champion-seed">({data.champion.seed})</span>{" "}
          {data.champion.name}
        </div>
      </div>
    </div>
  );
}

function ModelInfo() {
  return (
    <div className="model-info">
      <h3>Model Details — v2 Pipeline</h3>
      <ul>
        <li><strong>Algorithm:</strong> 4-model Ensemble (GBM + Random Forest + Logistic Regression + MLP) with stacked meta-learner + calibration</li>
        <li><strong>Features:</strong> 27 features — 18 base differentials (ADJOE, ADJDE, BARTHAG, EFG, TOR, ORB/DRB, FTR, 3P%, 2P%, Tempo, WAB, Seed) + 6 interaction (offense-vs-defense, tempo clash, 3PT matchup, rebound margin) + seed prior + win% diff + log BARTHAG ratio</li>
        <li><strong>Training:</strong> NCAA tournament games 2013–2023 + synthetic regular-season matchups, with temporal weighting (recent years weighted ~2.5x) and round weighting (late rounds up to 3x)</li>
        <li><strong>Data Leak Fix:</strong> Pre-tournament stats approximated by regressing out postseason games from end-of-season stats</li>
        <li><strong>Tuning:</strong> Optuna hyperparameter search (40 trials) on weighted cross-validated log-loss</li>
        <li><strong>Backtested Accuracy:</strong> 75.0% ensemble (LOO by year, 2013–2023), champion correct 8/10 years</li>
        <li><strong>2026 Stats:</strong> Pre-tournament adjusted efficiency from DeepMetricAnalytics (March 2026)</li>
      </ul>
    </div>
  );
}

function CustomMatchup() {
  const allTeams = useMemo(() => {
    const teams = {};
    for (const region of Object.values(bracketData.regions)) {
      for (const t of region.teams) {
        teams[t.name] = t.seed;
      }
    }
    return Object.entries(teams).sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0]));
  }, []);

  const [teamA, setTeamA] = useState(allTeams[0]?.[0] || "");
  const [teamB, setTeamB] = useState(allTeams[1]?.[0] || "");

  const allGames = useMemo(() => {
    const flat = [];
    for (const region of Object.values(bracketData.regions)) {
      for (const round of region.rounds) {
        for (const g of round) flat.push(g);
      }
    }
    for (const g of bracketData.final_four) flat.push(g);
    flat.push(bracketData.championship);
    return flat;
  }, []);

  const result = useMemo(() => {
    if (teamA === teamB) return null;
    const game = allGames.find(
      (g) =>
        (g.team_a === teamA && g.team_b === teamB) ||
        (g.team_a === teamB && g.team_b === teamA)
    );
    if (game) {
      const probA = game.team_a === teamA ? game.prob_a : game.prob_b;
      return { probA, source: "bracket" };
    }
    return null;
  }, [teamA, teamB, allGames]);

  const seedA = allTeams.find((t) => t[0] === teamA)?.[1] || 16;
  const seedB = allTeams.find((t) => t[0] === teamB)?.[1] || 16;

  return (
    <div className="custom-matchup">
      <h2 className="section-title">Custom Matchup — Live Model</h2>
      <div className="custom-controls">
        <select value={teamA} onChange={(e) => setTeamA(e.target.value)}>
          {allTeams.map(([name, seed]) => (
            <option key={name} value={name}>({seed}) {name}</option>
          ))}
        </select>
        <span className="vs">vs</span>
        <select value={teamB} onChange={(e) => setTeamB(e.target.value)}>
          {allTeams.map(([name, seed]) => (
            <option key={name} value={name}>({seed}) {name}</option>
          ))}
        </select>
      </div>
      {teamA !== teamB && result && (
        <div className="custom-result">
          <div className="custom-bar-container">
            <div className="custom-team-label left">
              <span className="seed">{seedA}</span> {teamA}
            </div>
            <div className="custom-bar">
              <div
                className="custom-bar-a"
                style={{ width: `${result.probA * 100}%` }}
              >
                {(result.probA * 100).toFixed(1)}%
              </div>
              <div
                className="custom-bar-b"
                style={{ width: `${(1 - result.probA) * 100}%` }}
              >
                {((1 - result.probA) * 100).toFixed(1)}%
              </div>
            </div>
            <div className="custom-team-label right">
              {teamB} <span className="seed">{seedB}</span>
            </div>
          </div>
        </div>
      )}
      {teamA === teamB && (
        <div className="custom-result dim">Select two different teams</div>
      )}
      {teamA !== teamB && !result && (
        <div className="custom-result dim">This matchup was not in the bracket simulation</div>
      )}
    </div>
  );
}

export default function App() {
  const [showInfo, setShowInfo] = useState(false);
  const regions = bracketData.regions;

  return (
    <div className="app">
      <header className="header">
        <h1>
          March Madness {bracketData.year}
          <span className="subtitle">ML Bracket Predictions</span>
        </h1>
        <button className="info-toggle" onClick={() => setShowInfo(!showInfo)}>
          {showInfo ? "Hide" : "Show"} Model Info
        </button>
      </header>

      {showInfo && <ModelInfo />}

      <FinalFour data={bracketData} />

      <CustomMatchup />

      <div className="regions-grid">
        {Object.entries(regions).map(([name, data]) => (
          <Region key={name} name={name} data={data} />
        ))}
      </div>
    </div>
  );
}
