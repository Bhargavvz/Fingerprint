import React from 'react';
import { Link } from 'react-router-dom';

export default function Footer() {
    return (
        <footer className="relative z-10 border-t border-white/10 mt-16">
            <div className="container mx-auto px-4 max-w-6xl py-12">
                <div className="grid md:grid-cols-4 gap-8">
                    {/* Brand */}
                    <div className="md:col-span-2">
                        <Link to="/" className="flex items-center gap-3 mb-4">
                            <div className="w-10 h-10 rounded-xl bg-white flex items-center justify-center">
                                <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none">
                                    <defs>
                                        <linearGradient id="bloodGradientFooter" x1="0%" y1="0%" x2="100%" y2="100%">
                                            <stop offset="0%" stopColor="#ef4444" />
                                            <stop offset="100%" stopColor="#1e3a8a" />
                                        </linearGradient>
                                    </defs>
                                    <path d="M12 2.5C12 2.5 5.5 10.5 5.5 15C5.5 18.59 8.41 21.5 12 21.5C15.59 21.5 18.5 18.59 18.5 15C18.5 10.5 12 2.5 12 2.5Z" stroke="url(#bloodGradientFooter)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            </div>
                            <span className="text-lg font-bold text-white">BloodType AI</span>
                        </Link>
                        <p className="text-gray-400 text-sm max-w-md mb-4">
                            AI-powered blood group prediction from fingerprint images using EfficientNet-B3 with CBAM attention mechanism.
                        </p>
                        <div className="flex gap-4">
                            <a
                                href="https://github.com/Bhargavvz/Fingerprint"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-gray-400 hover:text-white transition-colors"
                            >
                                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                                </svg>
                            </a>
                        </div>
                    </div>

                    {/* Quick Links */}
                    <div>
                        <h4 className="text-white font-semibold mb-4">Quick Links</h4>
                        <ul className="space-y-2">
                            <li><Link to="/detect" className="text-gray-400 hover:text-white text-sm transition-colors">Detection</Link></li>
                            <li><Link to="/about" className="text-gray-400 hover:text-white text-sm transition-colors">About</Link></li>
                            <li><Link to="/research" className="text-gray-400 hover:text-white text-sm transition-colors">Research</Link></li>
                        </ul>
                    </div>

                    {/* Legal */}
                    <div>
                        <h4 className="text-white font-semibold mb-4">Legal</h4>
                        <ul className="space-y-2">
                            <li><Link to="/privacy" className="text-gray-400 hover:text-white text-sm transition-colors">Privacy Policy</Link></li>
                            <li><Link to="/terms" className="text-gray-400 hover:text-white text-sm transition-colors">Terms of Use</Link></li>
                            <li>
                                <a href="https://github.com/Bhargavvz/Fingerprint" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white text-sm transition-colors">
                                    GitHub
                                </a>
                            </li>
                        </ul>
                    </div>
                </div>

                {/* Bottom */}
                <div className="mt-12 pt-8 border-t border-white/10">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                        <p className="text-gray-500 text-sm">© 2026 BloodType AI. All rights reserved.</p>
                        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                            <span className="text-yellow-400">⚠️</span>
                            <span className="text-yellow-300 text-xs">For research purposes only. Not for medical diagnosis.</span>
                        </div>
                    </div>
                </div>
            </div>
        </footer>
    );
}
