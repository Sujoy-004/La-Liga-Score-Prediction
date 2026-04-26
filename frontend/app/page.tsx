"use client";

import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import { motion } from "framer-motion";
import { 
  TrendingUp, 
  ShieldCheck, 
  Users, 
  Cpu,
  ArrowUpRight,
  Target
} from "lucide-react";
import { cn } from "@/lib/utils";

const stats = [
  { name: "Model Calibration", value: "0.2370", label: "Brier Score", icon: Cpu, trend: "+2.1%", positive: true },
  { name: "Prediction Accuracy", value: "57.35%", label: "CV Accuracy", icon: Target, trend: "+0.8%", positive: true },
  { name: "Data Points", value: "3,840", label: "Matches Synced", icon: Users, trend: "Weekly", positive: true },
  { name: "Engine Status", value: "Optimal", label: "Stacked Ensemble", icon: ShieldCheck, trend: "Online", positive: true },
];

export default function Dashboard() {
  return (
    <div className="flex h-full bg-background text-foreground overflow-hidden">
      <Sidebar />
      
      <main className="flex-1 flex flex-col overflow-hidden relative">
        <Header />
        
        <div className="flex-1 overflow-y-auto p-8 space-y-8 custom-scrollbar">
          {/* Hero Section */}
          <section>
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col gap-1"
            >
              <h1 className="text-3xl font-bold tracking-tight">Intelligence Dashboard</h1>
              <p className="text-slate-500 font-medium">Welcome back, Sujoy. The La Liga engine is currently operating at peak calibration.</p>
            </motion.div>
          </section>

          {/* Stats Grid */}
          <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {stats.map((stat, i) => (
              <motion.div
                key={stat.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="glass p-6 rounded-2xl group hover:border-arctic-blue/30 transition-all cursor-default"
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="p-2.5 bg-arctic-blue/10 rounded-xl group-hover:bg-arctic-blue/20 transition-colors">
                    <stat.icon className="w-5 h-5 text-arctic-blue" />
                  </div>
                  <div className={cn(
                    "flex items-center gap-1 text-xs font-bold px-2 py-1 rounded-full",
                    stat.positive ? "bg-win/10 text-win" : "bg-loss/10 text-loss"
                  )}>
                    {stat.trend}
                    <ArrowUpRight className="w-3 h-3" />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-500">{stat.name}</p>
                  <h3 className="text-2xl font-bold mt-1 tracking-tight">{stat.value}</h3>
                  <p className="text-[10px] uppercase tracking-widest font-bold text-slate-600 mt-2">{stat.label}</p>
                </div>
              </motion.div>
            ))}
          </section>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Recent Predictions */}
            <section className="lg:col-span-2 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-arctic-blue" />
                  Recent Calibrations
                </h2>
                <button className="text-sm font-bold text-arctic-blue hover:text-arctic-blue-soft transition-colors">View All</button>
              </div>
              
              <div className="glass rounded-2xl overflow-hidden border-arctic-border/50">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-white/5 border-b border-arctic-border/50">
                      <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-slate-500">Fixture</th>
                      <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-slate-500">Model Output</th>
                      <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-slate-500">Probability</th>
                      <th className="px-6 py-4 text-xs font-bold uppercase tracking-widest text-slate-500">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-arctic-border/30">
                    {[
                      { fixture: "Real Madrid vs Barcelona", output: "Home Win", prob: "64.2%", conf: "High" },
                      { fixture: "Atletico Madrid vs Sevilla", output: "Home Win", prob: "51.8%", conf: "Moderate" },
                      { fixture: "Valencia vs Villarreal", output: "Draw", prob: "38.4%", conf: "Low" },
                      { fixture: "Real Betis vs Real Sociedad", output: "Away Win", prob: "42.1%", conf: "Moderate" },
                    ].map((row, i) => (
                      <tr key={i} className="hover:bg-white/5 transition-colors cursor-pointer group">
                        <td className="px-6 py-4">
                          <span className="font-bold text-sm group-hover:text-arctic-blue transition-colors">{row.fixture}</span>
                        </td>
                        <td className="px-6 py-4">
                          <span className={cn(
                            "px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-tighter",
                            row.output.includes("Win") ? "bg-win/10 text-win" : "bg-draw/10 text-draw"
                          )}>
                            {row.output}
                          </span>
                        </td>
                        <td className="px-6 py-4 font-mono text-sm">{row.prob}</td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                              <div 
                                className={cn(
                                  "h-full rounded-full bg-arctic-blue",
                                  row.conf === "High" ? "w-full" : row.conf === "Moderate" ? "w-2/3" : "w-1/3"
                                )} 
                              />
                            </div>
                            <span className="text-xs font-medium text-slate-500">{row.conf}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Tactical Insights Card */}
            <section className="space-y-4">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-arctic-blue" />
                Tactical Insights
              </h2>
              <div className="glass p-6 rounded-2xl space-y-6">
                <div className="space-y-2">
                  <h4 className="text-sm font-bold text-slate-300">Bogey Team Detection</h4>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Villarreal currently exhibits a 1.4x risk multiplier when facing high-press blocks like Real Sociedad.
                  </p>
                </div>
                <div className="h-px bg-white/5" />
                <div className="space-y-2">
                  <h4 className="text-sm font-bold text-slate-300">Home Fortress Index</h4>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    San Mamés (Athletic Club) has entered a "High Stability" window, increasing Home Win priors by 8.4%.
                  </p>
                </div>
                <button className="w-full py-2.5 rounded-xl bg-arctic-blue/10 border border-arctic-blue/20 text-arctic-blue text-xs font-bold hover:bg-arctic-blue/20 transition-all">
                  Deep Insight Report
                </button>
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}
