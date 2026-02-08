import React from 'react';
import { Link } from 'react-router-dom';

// Real metrics from training output
const METRICS = [
    { label: 'Test Accuracy', value: '89.33%', icon: '🎯' },
    { label: 'Best Val F1', value: '92.60%', icon: '📊' },
    { label: 'Total Epochs', value: '110', icon: '🔄' },
    { label: 'Test Samples', value: '900', icon: '🧪' },
];

// Real per-class results from classification_report.json
const CLASS_RESULTS = [
    { name: 'A+', precision: 0.927, recall: 0.894, f1: 0.910, support: 85 },
    { name: 'A-', precision: 0.896, recall: 0.914, f1: 0.905, support: 151 },
    { name: 'B+', precision: 0.918, recall: 0.918, f1: 0.918, support: 98 },
    { name: 'B-', precision: 0.927, recall: 0.910, f1: 0.918, support: 111 },
    { name: 'AB+', precision: 0.904, recall: 0.887, f1: 0.895, support: 106 },
    { name: 'AB-', precision: 0.849, recall: 0.886, f1: 0.867, support: 114 },
    { name: 'O+', precision: 0.893, recall: 0.914, f1: 0.903, support: 128 },
    { name: 'O-', precision: 0.845, recall: 0.813, f1: 0.829, support: 107 },
];

export default function ResearchPage() {
    return (
        <div className="container mx-auto px-4 py-12 max-w-5xl">
            {/* Header */}
            <div className="text-center mb-16 fade-in">
                <h1 className="text-4xl md:text-5xl font-bold mb-4 gradient-text">
                    Research Results
                </h1>
                <p className="text-gray-400 max-w-2xl mx-auto">
                    Training metrics, performance analysis, and model evaluation results from EfficientNet-B3 with CBAM attention.
                </p>
            </div>

            {/* Key Metrics */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.1s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Key Metrics</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {METRICS.map((metric, index) => (
                        <div key={index} className="glass p-6 text-center">
                            <span className="text-3xl mb-2 block">{metric.icon}</span>
                            <p className="text-3xl font-bold text-white mb-1">{metric.value}</p>
                            <p className="text-gray-500 text-sm">{metric.label}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Training Curves */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.2s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Training Curves</h2>
                <div className="glass p-6">
                    <img
                        src="/outputs/training_curves.png"
                        alt="Training Curves - Loss and Accuracy over epochs"
                        className="w-full rounded-xl"
                    />
                </div>
            </section>

            {/* Confusion Matrix */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.3s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Confusion Matrix</h2>
                <div className="glass p-6">
                    <img
                        src="/outputs/confusion_matrix.png"
                        alt="Confusion Matrix - Predicted vs Actual blood groups"
                        className="w-full rounded-xl"
                    />
                </div>
            </section>

            {/* Per-Class Results */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.4s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Per-Class Performance</h2>
                <div className="glass overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-white/10">
                                    <th className="text-left py-4 px-6 text-gray-400 font-medium">Blood Group</th>
                                    <th className="text-center py-4 px-6 text-gray-400 font-medium">Precision</th>
                                    <th className="text-center py-4 px-6 text-gray-400 font-medium">Recall</th>
                                    <th className="text-center py-4 px-6 text-gray-400 font-medium">F1 Score</th>
                                    <th className="text-center py-4 px-6 text-gray-400 font-medium">Samples</th>
                                </tr>
                            </thead>
                            <tbody>
                                {CLASS_RESULTS.map((cls, index) => (
                                    <tr key={index} className="border-b border-white/5 transition-colors hover:bg-white/5">
                                        <td className="py-4 px-6">
                                            <span className="px-3 py-1 rounded-lg bg-primary/20 text-primary font-semibold">
                                                {cls.name}
                                            </span>
                                        </td>
                                        <td className="text-center py-4 px-6 text-white">{(cls.precision * 100).toFixed(1)}%</td>
                                        <td className="text-center py-4 px-6 text-white">{(cls.recall * 100).toFixed(1)}%</td>
                                        <td className="text-center py-4 px-6">
                                            <span className={`font-semibold ${cls.f1 >= 0.90 ? 'text-green-400' : cls.f1 >= 0.85 ? 'text-yellow-400' : 'text-orange-400'}`}>
                                                {(cls.f1 * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="text-center py-4 px-6 text-gray-400">{cls.support}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            {/* Per-Class Metrics Chart */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.45s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Per-Class Metrics Chart</h2>
                <div className="glass p-6">
                    <img
                        src="/outputs/per_class_metrics.png"
                        alt="Per-class precision, recall, and F1 scores"
                        className="w-full rounded-xl"
                    />
                </div>
            </section>

            {/* ROC Curves */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.5s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">ROC Curves</h2>
                <div className="glass p-6">
                    <img
                        src="/outputs/roc_curves.png"
                        alt="ROC curves for each blood group class"
                        className="w-full rounded-xl"
                    />
                </div>
            </section>

            {/* Model Info */}
            <section className="mb-16 fade-in" style={{ animationDelay: '0.55s' }}>
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Model Architecture</h2>
                <div className="glass p-8">
                    <div className="grid md:grid-cols-2 gap-6">
                        <div>
                            <h3 className="text-lg font-semibold text-white mb-4">Backbone</h3>
                            <ul className="space-y-2 text-gray-400 text-sm">
                                <li>• EfficientNet-B3 pretrained on ImageNet</li>
                                <li>• CBAM (Convolutional Block Attention Module)</li>
                                <li>• Custom classification head with dropout</li>
                                <li>• ~12M trainable parameters</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-white mb-4">Training</h3>
                            <ul className="space-y-2 text-gray-400 text-sm">
                                <li>• Focal Loss with label smoothing (0.1)</li>
                                <li>• AdamW optimizer with weight decay</li>
                                <li>• Cosine annealing with warm restarts</li>
                                <li>• MixUp and CutMix augmentation</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            {/* Links */}
            <section className="text-center fade-in" style={{ animationDelay: '0.6s' }}>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <a
                        href="https://github.com/Bhargavvz/Fingerprint"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn-secondary px-8 py-4 inline-flex items-center justify-center gap-2"
                    >
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                        </svg>
                        View on GitHub
                    </a>
                    <Link to="/detect" className="btn-primary px-8 py-4">
                        Try the Model
                    </Link>
                </div>
            </section>
        </div>
    );
}
