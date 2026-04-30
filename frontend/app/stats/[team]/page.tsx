"use client";

import { use } from "react";
import Header from "@/components/header";
import Sidebar from "@/components/sidebar";
import { motion } from "framer-motion";
import { 
  ShieldAlert, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  BarChart2,
  Calendar,
  Activity
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from "recharts";

export default function TeamStatsPage({ params }: { params: Promise<{ team: string }> }) {
  const { team } = use(params);
  const decodedTeam = decodeURIComponent(team);

  const { data: stats, isLoading } = useQuery({
    queryKey: ["stats", decodedTeam],
    queryFn: async () => {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";
      const res = await fetch(`${baseUrl}/stats/${decodedTeam}`);
      if (!res.ok) throw new Error("Team not found");
      return res.json();
    }
  });

  if (isLoading) return <LoadingState />;

  return (
    <div className="flex h-full bg-background text-foreground overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <div className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          <div className="max-w-5xl mx-auto space-y-8">
            {/* Team Header */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-end gap-6 pb-4 border-b border-white/5"
            >
              <div className="w-24 h-24 rounded-3xl bg-arctic-blue/10 border border-arctic-blue/20 flex items-center justify-center shadow-2xl">
                <ShieldAlert className="w-12 h-12 text-arctic-blue" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded-full bg-arctic-blue/10 border border-arctic-blue/20 text-[10px] font-black text-arctic-blue uppercase tracking-widest">
                    La Liga Entity
                  </span>
                </div>
                <h1 className="text-5xl font-black text-white tracking-tighter">{decodedTeam}</h1>
              </div>
            </motion.div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <MetricCard 
                title="Rolling Goals For" 
                value={stats?.rolling_gf.toFixed(2)} 
                icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
                trend={stats?.stability_trend}
              />
              <MetricCard 
                title="Rolling Goals Against" 
                value={stats?.rolling_ga.toFixed(2)} 
                icon={<TrendingDown className="w-4 h-4 text-loss" />}
                trend="Engine Calibrated"
              />
              <MetricCard 
                title="Tactical Stability" 
                value={stats?.tactical_stability} 
                icon={<Activity className="w-4 h-4 text-arctic-blue" />}
                trend="Engine Optimized"
              />
            </div>

            {/* Recent Form & Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 glass p-8 rounded-3xl space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-arctic-blue" />
                    <h2 className="text-xl font-bold">Goal Distribution</h2>
                  </div>
                </div>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={[
                      { name: 'M1', gf: stats?.rolling_gf * 0.8, ga: stats?.rolling_ga * 1.2 },
                      { name: 'M2', gf: stats?.rolling_gf * 1.1, ga: stats?.rolling_ga * 0.9 },
                      { name: 'M3', gf: stats?.rolling_gf * 1.3, ga: stats?.rolling_ga * 1.1 },
                      { name: 'M4', gf: stats?.rolling_gf * 0.9, ga: stats?.rolling_ga * 0.7 },
                      { name: 'M5', gf: stats?.rolling_gf, ga: stats?.rolling_ga },
                    ]}>
                      <defs>
                        <linearGradient id="colorGf" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                      <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0B0F1A', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                        itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
                      />
                      <Area type="monotone" dataKey="gf" stroke="#3b82f6" fillOpacity={1} fill="url(#colorGf)" strokeWidth={3} />
                      <Area type="monotone" dataKey="ga" stroke="#EF4444" fill="transparent" strokeWidth={2} strokeDasharray="5 5" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="glass p-8 rounded-3xl space-y-6">
                <div className="flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-arctic-blue" />
                  <h2 className="text-xl font-bold">Recent H2H Form</h2>
                </div>
                <div className="space-y-4">
                  {stats?.recent_results.map((res: string, i: number) => (
                    <div key={i} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/5">
                      <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Match {i+1}</span>
                      <div className={`px-3 py-1 rounded-lg text-[10px] font-black uppercase ${
                        res === 'W' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 
                        res === 'L' ? 'bg-rose-500/10 text-rose-500 border border-rose-500/20' : 
                        'bg-slate-500/10 text-slate-500 border border-slate-500/20'
                      }`}>
                        {res === 'W' ? 'Victory' : res === 'L' ? 'Defeat' : 'Draw'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function MetricCard({ title, value, icon, trend }: any) {
  return (
    <div className="glass p-6 rounded-3xl space-y-4 relative overflow-hidden group">
      <div className="flex items-center justify-between">
        <div className="p-2 bg-white/5 rounded-xl group-hover:bg-arctic-blue/10 transition-colors">
          {icon}
        </div>
        <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{trend}</span>
      </div>
      <div className="space-y-1">
        <p className="text-xs font-black text-slate-500 uppercase tracking-widest">{title}</p>
        <p className="text-3xl font-black text-white">{value}</p>
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="h-full w-full bg-background flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-arctic-blue border-t-transparent rounded-full animate-spin" />
        <p className="text-xs font-black text-arctic-blue uppercase tracking-[0.3em] animate-pulse">Synchronizing Tactical Data...</p>
      </div>
    </div>
  );
}
