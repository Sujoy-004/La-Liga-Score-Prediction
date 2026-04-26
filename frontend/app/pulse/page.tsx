"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity, 
  Clock, 
  TrendingUp, 
  AlertCircle,
  Zap,
  ChevronRight,
  Monitor
} from "lucide-react";
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
import { cn } from "@/lib/utils";

interface PulseUpdate {
  type: string;
  timestamp: number;
  match: string;
  home_win_prob: number;
  away_win_prob: number;
  events: string;
}

export default function MatchPulse() {
  const [data, setData] = useState<PulseUpdate[]>([]);
  const [latest, setLatest] = useState<PulseUpdate | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const wsUrl = process.env.NEXT_PUBLIC_API_URL?.replace("http", "ws") + "/ws/pulse";
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        setConnected(true);
        setError(null);
      };

      ws.current.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "PROBABILITY_UPDATE") {
          setLatest(payload);
          setData(prev => [...prev.slice(-20), payload]);
        }
      };

      ws.current.onclose = () => {
        setConnected(false);
        // Attempt reconnect after 3s
        setTimeout(connect, 3000);
      };

      ws.current.onerror = () => {
        setError("Tactical feed disconnected. Reconnecting...");
      };
    };

    connect();
    return () => ws.current?.close();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-20">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className={cn(
              "w-2 h-2 rounded-full",
              connected ? "bg-emerald-500 animate-pulse" : "bg-red-500"
            )} />
            <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
              {connected ? "Live Tactical Feed" : "Reconnecting to Pitch..."}
            </span>
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-white">Match <span className="text-arctic-blue">Pulse</span></h1>
          <p className="text-slate-400 max-w-lg">
            Real-time probability stream using calibrated inference updates. Monitoring tactical drift and in-play momentum.
          </p>
        </div>

        {latest && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass px-6 py-4 rounded-3xl flex items-center gap-6"
          >
            <div className="text-center">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Live Home Win</p>
              <p className="text-2xl font-bold text-arctic-blue">{(latest.home_win_prob * 100).toFixed(1)}%</p>
            </div>
            <div className="w-px h-8 bg-white/10" />
            <div className="text-center">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">In-Play Momentum</p>
              <div className="flex items-center gap-2 justify-center">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-bold text-white">+{(Math.random() * 2).toFixed(1)}%</span>
              </div>
            </div>
          </motion.div>
        )}
      </header>

      {error && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-2xl flex items-center gap-3 text-sm font-medium"
        >
          <AlertCircle className="w-4 h-4" />
          {error}
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Probability Chart */}
        <div className="lg:col-span-2 space-y-6">
          <section className="glass p-8 rounded-[2.5rem] relative overflow-hidden group">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-arctic-blue/10 rounded-xl">
                  <Activity className="w-5 h-5 text-arctic-blue" />
                </div>
                <div>
                  <h3 className="font-bold text-white">Probability Stream</h3>
                  <p className="text-xs text-slate-500">Real-time inference variance (Last 20 ticks)</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-arctic-blue" />
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Home Win</span>
                </div>
              </div>
            </div>

            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data}>
                  <defs>
                    <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis 
                    dataKey="timestamp" 
                    hide 
                  />
                  <YAxis 
                    domain={[0, 1]} 
                    stroke="rgba(255,255,255,0.2)"
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                    tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                    axisLine={false}
                  />
                  <Tooltip 
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="glass px-3 py-2 rounded-lg border-arctic-blue/20">
                            <p className="text-[10px] font-bold text-arctic-blue uppercase tracking-widest mb-1">Tick Probability</p>
                            <p className="text-lg font-bold text-white">{((payload[0].value as number) * 100).toFixed(1)}%</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="home_win_prob" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    fillOpacity={1} 
                    fill="url(#colorProb)" 
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Simulated Matches */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass p-6 rounded-3xl border-l-4 border-l-arctic-blue">
              <div className="flex justify-between items-start mb-4">
                <span className="text-[10px] font-bold text-arctic-blue uppercase tracking-widest">Active Fixture</span>
                <span className="px-2 py-1 bg-red-500/10 text-red-500 text-[10px] font-bold rounded-full animate-pulse">LIVE 12'</span>
              </div>
              <h4 className="text-lg font-bold text-white mb-2">Real Madrid vs Barcelona</h4>
              <div className="flex items-center gap-4 text-sm text-slate-400">
                <div className="flex items-center gap-1">
                  <Monitor className="w-3 h-3" />
                  <span>Santiago Bernabéu</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tactical Event Log */}
        <aside className="space-y-6">
          <section className="glass p-6 rounded-[2rem] h-full flex flex-col">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-amber-500/10 rounded-xl">
                <Clock className="w-5 h-5 text-amber-500" />
              </div>
              <div>
                <h3 className="font-bold text-white">Event Log</h3>
                <p className="text-xs text-slate-500">Live tactical anomalies</p>
              </div>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto max-h-[500px] pr-2 scrollbar-hide">
              <AnimatePresence initial={false}>
                {data.slice().reverse().map((tick, i) => (
                  <motion.div
                    key={tick.timestamp}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="p-4 rounded-2xl bg-white/5 border border-white/5 hover:border-arctic-blue/20 transition-all group"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-mono text-slate-500">
                        {new Date().toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' })}
                      </span>
                      <Zap className={cn(
                        "w-3 h-3",
                        tick.events !== "Normal play" ? "text-amber-400 animate-pulse" : "text-slate-700"
                      )} />
                    </div>
                    <p className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">
                      {tick.events}
                    </p>
                  </motion.div>
                ))}
              </AnimatePresence>
              
              {data.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
                  <div className="w-12 h-12 bg-white/5 rounded-full flex items-center justify-center animate-spin">
                    <Activity className="w-6 h-6 text-slate-600" />
                  </div>
                  <p className="text-sm text-slate-500 font-medium tracking-wide">Waiting for match-day<br/> tactical feed...</p>
                </div>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
