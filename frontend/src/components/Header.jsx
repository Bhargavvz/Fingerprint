import React from 'react';

export default function Header() {
    return (
        <header className="relative z-20 py-6">
            <div className="container mx-auto px-4 max-w-6xl">
                <div className="glass px-6 py-4 flex items-center justify-between">
                    {/* Logo */}
                    <div className="flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-red-500 via-primary to-pink-500 flex items-center justify-center text-2xl shadow-lg">
                            🩸
                        </div>
                        <div>
                            <h1 className="text-xl font-bold gradient-text-primary">BloodType AI</h1>
                            <p className="text-xs text-gray-400">Fingerprint Analysis</p>
                        </div>
                    </div>

                    {/* Nav Links */}
                    <nav className="hidden md:flex items-center gap-6">
                        <a href="#" className="text-gray-300 hover:text-white transition-colors">Home</a>
                        <a href="#" className="text-gray-300 hover:text-white transition-colors">About</a>
                        <a href="#" className="text-gray-300 hover:text-white transition-colors">Research</a>
                    </nav>

                    {/* Status Badge */}
                    <div className="flex items-center gap-3">
                        <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/20 border border-green-500/30">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span className="text-green-400 text-sm font-medium">Model Active</span>
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
}
