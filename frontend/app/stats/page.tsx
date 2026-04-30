"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Sidebar from "@/components/sidebar";
import Header from "@/components/header";
import { motion } from "framer-motion";
import { Users, ChevronRight, Search } from "lucide-react";

export default function StatsIndexPage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/teams`)
      .then((res) => res.json())
      .then((data) => {
        setTeams(data.sort());
        setLoading(false);
      })
      .catch((err) => console.error("Failed to fetch teams:", err));
  }, []);

  const filteredTeams = teams.filter((t) =>
    t.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex h-screen bg-void text-slate-100 overflow-hidden font-outfit">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {/* Arctic Pulse Background */}
        <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-arctic-blue/5 blur-[120px] rounded-full pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-arctic-blue/3 blur-[100px] rounded-full pointer-events-none" />
        
        <Header />

        <main className="flex-1 overflow-y-auto p-8 relative z-10 custom-scrollbar">
          <div className="max-w-7xl mx-auto space-y-8">
            <header>
              <h1 className="text-4xl font-bold tracking-tight">
                Deep <span className="text-arctic-blue">Stats</span>
              </h1>
              <p className="text-slate-400 mt-2 text-lg">
                Explore advanced analytics and tactical stability metrics for all La Liga entities.
              </p>
            </header>

            {/* Search Bar */}
            <div className="relative group max-w-md">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 h-5 text-slate-500 group-focus-within:text-arctic-blue transition-colors" />
              </div>
              <input
                type="text"
                placeholder="Search team..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="block w-full pl-12 pr-4 py-3 bg-white/5 border border-white/10 rounded-2xl focus:ring-2 focus:ring-arctic-blue/50 focus:border-arctic-blue outline-none transition-all placeholder:text-slate-600 text-lg"
              />
            </div>

            {loading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {[...Array(12)].map((_, i) => (
                  <div key={i} className="h-24 bg-white/5 animate-pulse rounded-2xl border border-white/10" />
                ))}
              </div>
            ) : (
              <motion.div 
                layout
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
              >
                {filteredTeams.map((team, index) => (
                  <motion.div
                    key={team}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                  >
                    <Link href={`/stats/${encodeURIComponent(team)}`}>
                      <div className="group bg-white/5 hover:bg-white/10 border border-white/10 hover:border-arctic-blue/30 p-5 rounded-2xl transition-all duration-300 flex items-center justify-between cursor-pointer relative overflow-hidden shadow-sm hover:shadow-arctic-blue/10">
                        <div className="flex items-center gap-4 relative z-10">
                          <div className="w-12 h-12 bg-white/5 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform">
                            <Users className="w-6 h-6 text-arctic-blue" />
                          </div>
                          <div>
                            <h3 className="font-bold text-lg group-hover:text-arctic-blue transition-colors">{team}</h3>
                            <p className="text-xs text-slate-500 uppercase tracking-widest">La Liga Entity</p>
                          </div>
                        </div>
                        <ChevronRight className="w-5 h-5 text-slate-600 group-hover:text-white group-hover:translate-x-1 transition-all" />
                        
                        {/* Hover Gradient Overlay */}
                        <div className="absolute inset-0 bg-gradient-to-r from-arctic-blue/0 to-arctic-blue/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </Link>
                  </motion.div>
                ))}
              </motion.div>
            )}

            {filteredTeams.length === 0 && !loading && (
              <div className="text-center py-20 bg-white/5 rounded-3xl border border-dashed border-white/10">
                <p className="text-slate-500 text-lg italic">No teams matching "{search}" found in our database.</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
