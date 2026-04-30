"use client";

import { useState, useEffect } from "react";
import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Target, 
  RefreshCcw, 
  Info, 
  ShieldAlert,
  ChevronDown,
  Trophy,
  Zap,
  BarChart3
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from "recharts";

const teams = Array.from(new Set([
  "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Villarreal", 
  "Real Betis", "Rayo Vallecano", "Mallorca", "Real Sociedad", "Celta Vigo", 
  "Osasuna", "Girona", "Getafe", "Espanyol", "Leganes", 
  "Las Palmas", "Valencia", "Alaves", "Valladolid"
])).sort();

export default function PredictorPage() {
  const [homeTeam, setHomeTeam] = useState("Real Madrid");
  const [awayTeam, setAwayTeam] = useState("Barcelona");
  const [isCalibrating, setIsCalibrating] = useState(false);
  const searchParams = useSearchParams();

  useEffect(() => {
    const home = searchParams.get("home");
    const away = searchParams.get("away");
    if (home && teams.includes(home)) setHomeTeam(home);
    if (away && teams.includes(away)) setAwayTeam(away);
  }, [searchParams]);

  const { data: prediction, refetch, isFetching } = useQuery({
    queryKey: ["prediction", homeTeam, awayTeam],
    queryFn: async () => {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${baseUrl}/predict?home_team=${homeTeam}&away_team=${awayTeam}`);
      if (!res.ok) throw new Error("Failed to fetch prediction");
      return res.json();
    },
    enabled: false,
  });

  const handlePredict = () => {
    setIsCalibrating(true);
    setTimeout(() => {
      refetch();
      setIsCalibrating(false);
    }, 800); // Artificial delay for "Quiet Luxury" feeling
  };

  return (
    <div className="flex h-full bg-background text-foreground overflow-hidden">
      <Sidebar />
      
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          <div className="max-w-4xl mx-auto space-y-8">
            {/* Header */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center justify-between"
            >
              <div>
                <h1 className="text-3xl font-bold tracking-tight">Tactical Predictor</h1>
                <p className="text-slate-500 font-medium">Calibrated Stacked Ensemble Inference</p>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-arctic-blue/10 border border-arctic-blue/20 rounded-full">
                <div className="w-2 h-2 bg-arctic-blue rounded-full animate-pulse" />
                <span className="text-[10px] font-bold uppercase tracking-widest text-arctic-blue">V0.6.0 ENGINE</span>
              </div>
            </motion.div>

            {/* Selection Grid */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              <TeamSelect 
                label="Home Team" 
                value={homeTeam} 
                onChange={setHomeTeam} 
                icon={<ShieldAlert className="w-4 h-4 text-arctic-blue" />}
              />
              
              <div className="flex flex-col items-center justify-center">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handlePredict}
                  disabled={isCalibrating || isFetching}
                  className="w-16 h-16 rounded-full bg-arctic-blue flex items-center justify-center shadow-[0_0_30px_rgba(59,130,246,0.3)] hover:shadow-[0_0_40px_rgba(59,130,246,0.5)] transition-all disabled:opacity-50 group"
                >
                  <RefreshCcw className={cn("w-8 h-8 text-white", (isCalibrating || isFetching) && "animate-spin")} />
                </motion.button>
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mt-4">Run Inference</span>
              </div>

              <TeamSelect 
                label="Away Team" 
                value={awayTeam} 
                onChange={setAwayTeam} 
                icon={<Target className="w-4 h-4 text-loss" />}
              />
            </section>

            {/* Results Section */}
            <AnimatePresence mode="wait">
              {prediction ? (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="space-y-8"
                >
                  {/* Result Card */}
                  <div className="glass p-8 rounded-3xl space-y-6 bg-gradient-to-br from-white/[0.03] to-transparent">
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                      <div className="space-y-1">
                        <p className="text-[10px] font-black text-arctic-blue uppercase tracking-[0.2em]">Predicted Outcome</p>
                        <h2 className="text-5xl font-black text-white tracking-tight">
                          {prediction.prediction === "Home Win" ? homeTeam : awayTeam} <span className="text-slate-500">to win</span>
                        </h2>
                      </div>
                      <div className="bg-arctic-blue/10 border border-arctic-blue/20 px-6 py-4 rounded-2xl text-center min-w-[140px]">
                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Confidence</p>
                        <h2 className="text-3xl font-black text-arctic-blue">
                          {prediction.prediction === "Home Win" 
                            ? prediction.probability_home_win 
                            : `${(100 - parseFloat(prediction.probability_home_win)).toFixed(1)}%`}
                        </h2>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="relative h-3 bg-white/5 rounded-full overflow-hidden border border-white/10">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ 
                            width: prediction.prediction === "Home Win" 
                              ? prediction.probability_home_win 
                              : `${(100 - parseFloat(prediction.probability_home_win)).toFixed(1)}%` 
                          }}
                          className="absolute inset-y-0 left-0 bg-arctic-blue rounded-full shadow-[0_0_20px_rgba(59,130,246,0.4)]"
                        />
                      </div>
                      <div className="flex justify-between text-[9px] font-black uppercase tracking-[0.15em] text-slate-500 px-1">
                        <span>Win Probability Index</span>
                        <span>100% Certainty</span>
                      </div>
                    </div>
                  </div>

                  {/* Insights Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {prediction.insights.map((insight: string, i: number) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="glass p-5 rounded-2xl flex gap-4 border-l-4 border-l-arctic-blue"
                      >
                        <div className="p-2 bg-arctic-blue/10 rounded-lg h-fit">
                          <Zap className="w-4 h-4 text-arctic-blue" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400">Tactical Insight</h4>
                          <p className="text-sm text-slate-200 leading-relaxed font-medium">{insight}</p>
                        </div>
                      </motion.div>
                    ))}
                    {prediction.insights.length === 0 && (
                      <div className="glass p-5 rounded-2xl flex gap-4 border-l-4 border-l-slate-500 col-span-2">
                        <div className="p-2 bg-slate-500/10 rounded-lg h-fit">
                          <Info className="w-4 h-4 text-slate-500" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400">Baseline Stability</h4>
                          <p className="text-sm text-slate-200 leading-relaxed font-medium">
                            No significant tactical anomalies detected. Historical priors and rolling form dominate the inference.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Tactical Attribution Chart */}
                  {prediction.attribution && !prediction.attribution.error && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="glass p-8 rounded-3xl space-y-6"
                    >
                      <div className="flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-arctic-blue" />
                        <h2 className="text-xl font-bold">Tactical Attribution (SHAP)</h2>
                      </div>
                      <p className="text-sm text-slate-500">
                        Feature weights indicating the mathematical influence on the "Home Win" probability.
                      </p>
                      
                      <div className="h-[300px] w-full mt-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={Object.entries(prediction.attribution)
                              .map(([name, value]) => ({ 
                                name: name.replace(/_/g, ' ').toUpperCase(), 
                                value: value as number 
                              }))
                              .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
                              .slice(0, 8)
                            }
                            margin={{ left: 120, right: 40 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                            <XAxis type="number" hide />
                            <YAxis 
                              dataKey="name" 
                              type="category" 
                              axisLine={false}
                              tickLine={false}
                              tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700 }}
                              width={120}
                            />
                            <Tooltip 
                              cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  return (
                                    <div className="glass px-3 py-2 rounded-lg border-arctic-blue/20">
                                      <p className="text-[10px] font-bold text-arctic-blue uppercase tracking-widest">{payload[0].payload.name}</p>
                                      <p className="text-sm font-bold text-white">{(payload[0].value as number).toFixed(4)}</p>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                              {Object.entries(prediction.attribution).map((entry, index) => (
                                <Cell 
                                  key={`cell-${index}`} 
                                  fill={(entry[1] as number) >= 0 ? '#10B981' : '#EF4444'} 
                                  fillOpacity={0.8}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="glass p-12 rounded-3xl border-dashed border-2 flex flex-col items-center justify-center text-center space-y-4"
                >
                  <div className="p-4 bg-white/5 rounded-full">
                    <Trophy className="w-12 h-12 text-slate-700" />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-xl font-bold text-slate-300">Ready for Inference</h3>
                    <p className="text-sm text-slate-500 max-w-xs">
                      Select two teams and hit the calibration engine to generate deep tactical probabilities.
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}

function TeamSelect({ label, value, onChange, icon }: any) {
  return (
    <div className="space-y-3">
      <label className="text-xs font-bold uppercase tracking-widest text-slate-500 flex items-center gap-2">
        {icon}
        {label}
      </label>
      <div className="relative group">
        <select 
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-white/5 border border-white/10 rounded-2xl py-4 px-5 text-sm font-bold focus:outline-none focus:border-arctic-blue/50 focus:ring-4 ring-arctic-blue/5 transition-all appearance-none cursor-pointer group-hover:bg-white/10"
        >
          {teams.map(team => (
            <option key={team} value={team} className="bg-[#0B0F1A] text-white">{team}</option>
          ))}
        </select>
        <ChevronDown className="absolute right-5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none group-focus-within:text-arctic-blue transition-colors" />
      </div>
    </div>
  );
}
