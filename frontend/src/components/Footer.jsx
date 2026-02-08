import React from 'react';

export default function Footer() {
    return (
        <footer className="relative z-10 py-8 mt-16">
            <div className="container mx-auto px-4 max-w-6xl">
                <div className="glass p-8 text-center">
                    {/* Brand */}
                    <div className="flex items-center justify-center gap-3 mb-6">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500 via-primary to-pink-500 flex items-center justify-center text-xl">
                            🩸
                        </div>
                        <span className="text-xl font-bold gradient-text-primary">BloodType AI</span>
                    </div>

                    {/* Disclaimer */}
                    <p className="text-gray-400 text-sm max-w-2xl mx-auto mb-6">
                        This is an academic research project. Blood group determination requires proper
                        laboratory testing by qualified healthcare professionals. This tool should not
                        be used for medical diagnosis.
                    </p>

                    {/* Links */}
                    <div className="flex flex-wrap justify-center gap-6 mb-6">
                        <a href="#" className="text-gray-400 hover:text-primary transition-colors text-sm">
                            Privacy Policy
                        </a>
                        <a href="#" className="text-gray-400 hover:text-primary transition-colors text-sm">
                            Terms of Use
                        </a>
                        <a href="#" className="text-gray-400 hover:text-primary transition-colors text-sm">
                            Research Paper
                        </a>
                        <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-primary transition-colors text-sm">
                            GitHub
                        </a>
                    </div>

                    {/* Credit */}
                    <div className="pt-6 border-t border-white/10">
                        <p className="text-gray-500 text-sm">
                            © 2026 CMR College of Engineering and Technology
                        </p>
                        <p className="text-gray-600 text-xs mt-2">
                            D. Saketh Reddy • G. Surya Kiran • G. Bhavana Reddy | Guide: Dr. P. Senthil
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}
