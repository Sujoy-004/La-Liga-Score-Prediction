"use client";

import { useState } from "react";

const TEAMS = [
  "Real Madrid",
  "Barcelona",
  "Atletico Madrid",
  "Sevilla",
  "Valencia",
  "Athletic Club",
  "Real Sociedad",
  "Villarreal",
  "Real Betis",
  "Girona"
];

export default function Home() {
  const [homeTeam, setHomeTeam] = useState(TEAMS[0]);
  const [awayTeam, setAwayTeam] = useState(TEAMS[1]);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [pHome, setPHome] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (homeTeam === awayTeam) {
      setError("Home and Away teams must be different");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      setPrediction(data.prediction);
      setPHome(data.p_home);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceLevel = (p: number) => {
    if (p >= 0.75) return { label: "High", color: "text-green-600" };
    if (p >= 0.60) return { label: "Medium", color: "text-yellow-600" };
    return { label: "Low", color: "text-red-600" };
  };

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center p-4 font-sans">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
        <h1 className="text-3xl font-extrabold text-center text-gray-900 mb-8 tracking-tight">
          ⚽ La Liga Predictor
        </h1>

        <div className="space-y-6">
          {/* Team Selection */}
          <div className="grid grid-cols-1 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">🏠 Home Team</label>
              <select
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
                className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-orange-500 focus:outline-none transition-all"
              >
                {TEAMS.map((team) => (
                  <option key={team} value={team}>{team}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">✈️ Away Team</label>
              <select
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
                className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-orange-500 focus:outline-none transition-all"
              >
                {TEAMS.map((team) => (
                  <option key={team} value={team}>{team}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={loading}
            className={`w-full py-4 rounded-xl text-white font-bold text-lg shadow-lg transition-all ${
              loading 
                ? "bg-gray-400 cursor-not-allowed" 
                : "bg-orange-600 hover:bg-orange-700 active:scale-[0.98]"
            }`}
          >
            {loading ? "🔮 Predicting..." : "🔮 Predict Result"}
          </button>

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 text-red-700 rounded-lg text-sm font-medium border border-red-100 text-center">
              {error}
            </div>
          )}

          {/* Results Display */}
          {prediction && pHome !== null && (
            <div className="mt-8 p-6 bg-orange-50 rounded-2xl border border-orange-100 text-center animate-in fade-in slide-in-from-bottom-4 duration-500">
              <h3 className="text-gray-600 text-sm font-bold uppercase tracking-widest mb-2">Match Prediction</h3>
              <div className="text-4xl font-black text-orange-600 mb-4">{prediction}</div>
              
              <div className="flex justify-between items-center bg-white p-4 rounded-xl shadow-sm border border-orange-50">
                <div className="text-left">
                  <div className="text-xs text-gray-500 font-bold uppercase">Home Win Prob.</div>
                  <div className="text-xl font-bold text-gray-900">{(pHome * 100).toFixed(1)}%</div>
                </div>
                
                <div className="text-right">
                  <div className="text-xs text-gray-500 font-bold uppercase">Confidence</div>
                  <div className={`text-xl font-bold ${getConfidenceLevel(pHome).color}`}>
                    {getConfidenceLevel(pHome).label}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <footer className="mt-8 text-center text-gray-400 text-xs">
          Built with Senior Architect Protocol Standards
        </footer>
      </div>
    </main>
  );
}
