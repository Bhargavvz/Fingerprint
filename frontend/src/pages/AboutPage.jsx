import React from 'react';
import { Link } from 'react-router-dom';

const TECH_STACK = [
    { name: 'PyTorch', icon: '🔥', description: 'Deep learning framework' },
    { name: 'EfficientNet-B3', icon: '🧠', description: 'Backbone architecture' },
    { name: 'CBAM', icon: '👁️', description: 'Attention mechanism' },
    { name: 'Grad-CAM', icon: '🔍', description: 'Explainability' },
    { name: 'FastAPI', icon: '⚡', description: 'Backend API' },
    { name: 'React', icon: '⚛️', description: 'Frontend UI' },
];

const METHODOLOGY = [
    {
        title: 'Data Collection',
        description: 'Fingerprint images collected and labeled with corresponding blood groups from verified sources.',
        icon: '📊'
    },
    {
        title: 'Preprocessing',
        description: 'Images normalized, resized to 224x224, and augmented with elastic transforms and geometric variations.',
        icon: '🔧'
    },
    {
        title: 'Model Architecture',
        description: 'EfficientNet-B3 backbone with CBAM attention module for enhanced feature extraction.',
        icon: '🏗️'
    },
    {
        title: 'Training',
        description: 'Trained with Focal Loss, MixUp/CutMix augmentation, and cosine annealing scheduler.',
        icon: '🎯'
    },
];

export default function AboutPage() {
    return (
        <div className="container mx-auto px-4 py-12 max-w-4xl">
            {/* Header */}
            <div className="text-center mb-16 fade-in">
                <h1 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
                    About This Project
                </h1>
                <p className="text-gray-400 max-w-2xl mx-auto">
                    An academic research project exploring the correlation between fingerprint patterns and blood groups using deep learning.
                </p>
            </div>

            {/* Overview */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.1s' }}>
                <div className="glass p-8">
                    <h2 className="text-2xl font-bold text-white mb-4">Project Overview</h2>
                    <p className="text-gray-300 leading-relaxed mb-4">
                        This research project investigates the hypothesis that fingerprint patterns may contain latent features correlated with blood group types. Using state-of-the-art deep learning techniques, we've developed an AI system that analyzes fingerprint images to predict blood groups.
                    </p>
                    <p className="text-gray-300 leading-relaxed">
                        The model achieves <strong className="text-primary">89% accuracy</strong> across 8 blood group classes (A+, A-, B+, B-, AB+, AB-, O+, O-) and provides visual explanations through Grad-CAM heatmaps.
                    </p>
                </div>
            </section>

            {/* Methodology */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.2s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Methodology</h2>
                <div className="grid md:grid-cols-2 gap-6">
                    {METHODOLOGY.map((step, index) => (
                        <div key={index} className="glass glass-hover p-6">
                            <div className="flex items-start gap-4">
                                <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center text-2xl shrink-0">
                                    {step.icon}
                                </div>
                                <div>
                                    <h3 className="text-lg font-semibold text-white mb-2">{step.title}</h3>
                                    <p className="text-gray-400 text-sm">{step.description}</p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Tech Stack */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.3s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Technology Stack</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {TECH_STACK.map((tech, index) => (
                        <div key={index} className="glass glass-hover p-4 text-center">
                            <span className="text-3xl mb-2 block">{tech.icon}</span>
                            <h4 className="font-semibold text-white">{tech.name}</h4>
                            <p className="text-gray-500 text-xs">{tech.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Disclaimer */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.4s' }}>
                <div className="glass p-6 border-l-4 border-yellow-500">
                    <h3 className="text-lg font-semibold text-yellow-300 mb-3 flex items-center gap-2">
                        <span>⚠️</span> Important Disclaimer
                    </h3>
                    <p className="text-yellow-200/80 text-sm leading-relaxed">
                        This is a research project and its predictions should NOT be used for medical purposes. The correlation between fingerprints and blood groups is a subject of ongoing scientific investigation. Blood group determination must be performed through proper laboratory testing by qualified medical professionals.
                    </p>
                </div>
            </section>

            {/* CTA */}
            <div className="text-center fade-in" style={{ animationDelay: '0.5s' }}>
                <Link to="/detect" className="btn-primary px-8 py-4 inline-flex items-center gap-2">
                    Try the Detection Tool
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                </Link>
            </div>
        </div>
    );
}
