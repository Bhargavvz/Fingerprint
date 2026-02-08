import React from 'react';
import { Link } from 'react-router-dom';

const FEATURES = [
    {
        icon: '🧬',
        title: 'Deep Learning',
        description: 'EfficientNet-B3 with CBAM attention mechanism for high accuracy predictions'
    },
    {
        icon: '🔬',
        title: 'Explainable AI',
        description: 'Grad-CAM visualizations show exactly what the model focuses on'
    },
    {
        icon: '🩸',
        title: '8 Blood Groups',
        description: 'Classifies A+, A-, B+, B-, AB+, AB-, O+, O- with confidence scores'
    },
    {
        icon: '⚡',
        title: 'Real-time',
        description: 'Get predictions in seconds with our optimized inference pipeline'
    },
    {
        icon: '📊',
        title: '89% Accuracy',
        description: 'Trained on thousands of fingerprint samples with rigorous validation'
    },
    {
        icon: '🔒',
        title: 'Privacy First',
        description: 'Images are processed locally and never stored on our servers'
    }
];

const STEPS = [
    { step: 1, title: 'Upload', description: 'Upload a clear fingerprint image', icon: '📤' },
    { step: 2, title: 'Analyze', description: 'AI processes the ridge patterns', icon: '🔍' },
    { step: 3, title: 'Result', description: 'Get blood group prediction with explanation', icon: '✨' }
];

export default function LandingPage() {
    return (
        <div className="overflow-hidden">
            {/* Hero Section */}
            <section className="relative min-h-[90vh] flex items-center justify-center px-4">
                <div className="max-w-5xl mx-auto text-center">
                    {/* Badge */}
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 mb-8 fade-in">
                        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        <span className="text-indigo-300 text-sm">Research Project • 89% Accuracy</span>
                    </div>

                    {/* Heading */}
                    <h1 className="text-5xl md:text-7xl font-black mb-6 leading-tight fade-in" style={{ animationDelay: '0.1s' }}>
                        <span className="gradient-text">
                            Blood Group Detection
                        </span>
                        <br />
                        <span className="bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
                            from Fingerprints
                        </span>
                    </h1>

                    {/* Description */}
                    <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-10 fade-in" style={{ animationDelay: '0.2s' }}>
                        Revolutionary AI-powered system that predicts blood groups by analyzing fingerprint patterns using deep learning and computer vision.
                    </p>

                    {/* CTA Buttons */}
                    <div className="flex flex-col sm:flex-row gap-4 justify-center fade-in" style={{ animationDelay: '0.3s' }}>
                        <Link
                            to="/detect"
                            className="btn-primary px-8 py-4 text-lg"
                        >
                            Try Detection →
                        </Link>
                        <Link
                            to="/research"
                            className="btn-secondary px-8 py-4 text-lg"
                        >
                            View Research
                        </Link>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-8 mt-16 max-w-lg mx-auto fade-in" style={{ animationDelay: '0.4s' }}>
                        <div className="text-center">
                            <p className="text-3xl font-bold text-white">89%</p>
                            <p className="text-gray-500 text-sm">Accuracy</p>
                        </div>
                        <div className="text-center">
                            <p className="text-3xl font-bold text-white">8</p>
                            <p className="text-gray-500 text-sm">Blood Groups</p>
                        </div>
                        <div className="text-center">
                            <p className="text-3xl font-bold text-white">12M+</p>
                            <p className="text-gray-500 text-sm">Parameters</p>
                        </div>
                    </div>
                </div>

                {/* Scroll indicator */}
                <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
                    <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                    </svg>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-24 px-4">
                <div className="max-w-6xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                            Powered by Advanced AI
                        </h2>
                        <p className="text-gray-400 max-w-2xl mx-auto">
                            Combining state-of-the-art deep learning with explainable AI for transparent predictions.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {FEATURES.map((feature, index) => (
                            <div
                                key={index}
                                className="glass glass-hover p-6 text-center"
                            >
                                <div className="text-4xl mb-4">{feature.icon}</div>
                                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                                <p className="text-gray-400 text-sm">{feature.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* How it Works */}
            <section className="py-24 px-4 bg-gradient-to-b from-transparent via-indigo-950/20 to-transparent">
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                            How It Works
                        </h2>
                        <p className="text-gray-400">
                            Simple three-step process to get your blood group prediction.
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {STEPS.map((item) => (
                            <div key={item.step} className="text-center">
                                <div className="relative inline-block mb-6">
                                    <div className="w-20 h-20 rounded-2xl bg-primary flex items-center justify-center text-3xl shadow-xl shadow-indigo-500/30">
                                        {item.icon}
                                    </div>
                                    <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-dark border-2 border-primary flex items-center justify-center text-sm font-bold text-white">
                                        {item.step}
                                    </div>
                                </div>
                                <h3 className="text-xl font-semibold text-white mb-2">{item.title}</h3>
                                <p className="text-gray-400 text-sm">{item.description}</p>
                            </div>
                        ))}
                    </div>

                    <div className="text-center mt-12">
                        <Link
                            to="/detect"
                            className="btn-primary px-8 py-4 inline-flex items-center gap-2"
                        >
                            Get Started Now
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Disclaimer Banner */}
            <section className="py-12 px-4">
                <div className="max-w-4xl mx-auto">
                    <div className="glass p-6 border-l-4 border-yellow-500">
                        <div className="flex items-start gap-4">
                            <span className="text-3xl">⚠️</span>
                            <div>
                                <h3 className="text-lg font-semibold text-yellow-300 mb-2">Research Project Disclaimer</h3>
                                <p className="text-yellow-200/80 text-sm">
                                    This is an academic research project exploring the correlation between fingerprint patterns and blood groups.
                                    The predictions made by this system are for research and educational purposes only.
                                    <strong> This is NOT a substitute for proper medical blood testing.</strong> Always consult healthcare professionals for blood group determination.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
}
