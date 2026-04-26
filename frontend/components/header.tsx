"use client";

import { Bell, Search, User } from "lucide-react";
import { motion } from "framer-motion";

export default function Header() {
  return (
    <header className="h-16 glass border-b px-8 flex items-center justify-between z-40 sticky top-0">
      <div className="flex items-center gap-4 flex-1">
        <div className="relative group max-w-md w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-focus-within:text-arctic-blue transition-colors" />
          <input 
            type="text" 
            placeholder="Search fixtures, teams, or insights..." 
            className="w-full bg-white/5 border border-white/10 rounded-full py-1.5 pl-10 pr-4 text-sm focus:outline-none focus:border-arctic-blue/50 focus:ring-4 ring-arctic-blue/5 transition-all"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <motion.button 
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="p-2 rounded-full hover:bg-white/5 transition-colors relative"
        >
          <Bell className="w-5 h-5 text-slate-400" />
          <span className="absolute top-2 right-2 w-2 h-2 bg-arctic-blue rounded-full border-2 border-background" />
        </motion.button>
        
        <div className="h-8 w-px bg-white/10 mx-2" />
        
        <button className="flex items-center gap-3 pl-2 pr-4 py-1.5 rounded-full hover:bg-white/5 transition-all">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-arctic-blue to-purple-600 flex items-center justify-center text-xs font-bold shadow-lg">
            S
          </div>
          <div className="text-left hidden sm:block">
            <p className="text-xs font-bold text-white">Sujoy</p>
            <p className="text-[10px] text-slate-500 font-medium">Lead Architect</p>
          </div>
        </button>
      </div>
    </header>
  );
}
