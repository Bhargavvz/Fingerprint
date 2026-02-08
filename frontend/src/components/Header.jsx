import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const NAV_LINKS = [
    { path: '/', label: 'Home' },
    { path: '/detect', label: 'Detection' },
    { path: '/about', label: 'About' },
    { path: '/research', label: 'Research' },
];

export default function Header() {
    const [isOpen, setIsOpen] = useState(false);
    const location = useLocation();

    return (
        <header className="relative z-20 py-6">
            <div className="container mx-auto px-4 max-w-6xl">
                <div className="glass px-6 py-4 flex items-center justify-between">
                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-3 group">
                        <div className="w-11 h-11 rounded-xl bg-white flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
                            <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none">
                                <defs>
                                    <linearGradient id="bloodGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#ef4444" />
                                        <stop offset="100%" stopColor="#1e3a8a" />
                                    </linearGradient>
                                </defs>
                                <path d="M12 2.5C12 2.5 5.5 10.5 5.5 15C5.5 18.59 8.41 21.5 12 21.5C15.59 21.5 18.5 18.59 18.5 15C18.5 10.5 12 2.5 12 2.5Z" stroke="url(#bloodGradient)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                        </div>
                        <div>
                            <h1 className="text-xl font-bold text-white">BloodType AI</h1>
                            <p className="text-xs text-gray-400">Fingerprint Analysis</p>
                        </div>
                    </Link>

                    {/* Nav Links - Desktop */}
                    <nav className="hidden md:flex items-center gap-1">
                        {NAV_LINKS.map(({ path, label }) => (
                            <Link
                                key={path}
                                to={path}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${location.pathname === path
                                    ? 'bg-white/10 text-white'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                {label}
                            </Link>
                        ))}
                    </nav>

                    {/* Right Section */}
                    <div className="flex items-center gap-3">
                        {/* GitHub */}
                        <a
                            href="https://github.com/Bhargavvz/Fingerprint"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="hidden sm:flex p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
                        >
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                            </svg>
                        </a>

                        {/* Status Badge */}
                        <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/20 border border-green-500/30">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span className="text-green-400 text-sm font-medium">Model Active</span>
                        </div>

                        {/* Try Now Button */}
                        <Link
                            to="/detect"
                            className="hidden md:block btn-primary px-4 py-2 text-sm"
                        >
                            Try Now
                        </Link>

                        {/* Mobile Menu Button */}
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="md:hidden p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5"
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                {isOpen ? (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                ) : (
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                                )}
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Mobile Menu */}
                {isOpen && (
                    <div className="md:hidden mt-2 glass p-4 fade-in">
                        {NAV_LINKS.map(({ path, label }) => (
                            <Link
                                key={path}
                                to={path}
                                onClick={() => setIsOpen(false)}
                                className={`block px-4 py-3 rounded-lg text-sm font-medium transition-all ${location.pathname === path
                                    ? 'bg-white/10 text-white'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                {label}
                            </Link>
                        ))}
                        <div className="mt-4 pt-4 border-t border-white/10">
                            <Link
                                to="/detect"
                                onClick={() => setIsOpen(false)}
                                className="btn-primary w-full py-3 text-center block"
                            >
                                Try Detection
                            </Link>
                        </div>
                    </div>
                )}
            </div>
        </header>
    );
}
