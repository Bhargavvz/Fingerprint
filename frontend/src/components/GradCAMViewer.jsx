import React, { useState } from 'react';

export default function GradCAMViewer({ originalImage, gradcamImage, explanation, bloodGroup }) {
    const [showOverlay, setShowOverlay] = useState(true);
    const [opacity, setOpacity] = useState(0.6);

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-2xl font-bold gradient-text">🔍 Visual Explanation</h3>
                    <p className="text-gray-400 mt-1">
                        Grad-CAM highlights regions that influenced the prediction
                    </p>
                </div>
            </div>

            {/* Image Viewer */}
            <div className="grid md:grid-cols-2 gap-6">
                {/* Original Image */}
                <div className="glass p-4 rounded-2xl">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                        <span>📷</span>
                        Original Fingerprint
                    </h4>
                    <div className="relative rounded-xl overflow-hidden bg-black/30">
                        <img
                            src={originalImage}
                            alt="Original fingerprint"
                            className="w-full h-64 object-contain"
                        />
                    </div>
                </div>

                {/* Grad-CAM Overlay */}
                <div className="glass p-4 rounded-2xl">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                        <span>🔥</span>
                        Attention Heatmap
                    </h4>
                    <div className="relative rounded-xl overflow-hidden bg-black/30">
                        <img
                            src={originalImage}
                            alt="Base fingerprint"
                            className="w-full h-64 object-contain"
                        />
                        {showOverlay && gradcamImage && (
                            <img
                                src={`data:image/png;base64,${gradcamImage}`}
                                alt="Grad-CAM overlay"
                                className="absolute inset-0 w-full h-64 object-contain mix-blend-multiply"
                                style={{ opacity }}
                            />
                        )}
                    </div>
                </div>
            </div>

            {/* Controls */}
            <div className="glass p-4 rounded-2xl">
                <div className="flex flex-wrap items-center gap-6">
                    {/* Toggle Overlay */}
                    <label className="flex items-center gap-3 cursor-pointer">
                        <div
                            className={`w-12 h-6 rounded-full transition-colors ${showOverlay ? 'bg-primary' : 'bg-gray-600'
                                }`}
                            onClick={() => setShowOverlay(!showOverlay)}
                        >
                            <div
                                className={`w-5 h-5 rounded-full bg-white shadow-md transform transition-transform ${showOverlay ? 'translate-x-6' : 'translate-x-0.5'
                                    } mt-0.5`}
                            />
                        </div>
                        <span className="text-gray-300 text-sm">Show Overlay</span>
                    </label>

                    {/* Opacity Slider */}
                    <div className="flex items-center gap-3">
                        <span className="text-gray-400 text-sm">Opacity:</span>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={opacity}
                            onChange={(e) => setOpacity(parseFloat(e.target.value))}
                            className="w-32 accent-primary"
                        />
                        <span className="text-gray-300 text-sm w-12">{(opacity * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>

            {/* Explanation Text */}
            {explanation?.explanation && (
                <div className="glass p-6 rounded-2xl">
                    <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <span>💡</span>
                        AI Interpretation
                    </h4>
                    <div className="space-y-4 text-gray-300">
                        <p>{explanation.explanation}</p>

                        {/* Key Features */}
                        <div className="grid sm:grid-cols-3 gap-4 mt-6">
                            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                                <p className="text-gray-400 text-xs uppercase mb-1">Confidence</p>
                                <p className="text-2xl font-bold text-primary">
                                    {((explanation.confidence || 0) * 100).toFixed(1)}%
                                </p>
                            </div>
                            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                                <p className="text-gray-400 text-xs uppercase mb-1">Predicted Class</p>
                                <p className="text-2xl font-bold text-white">{bloodGroup}</p>
                            </div>
                            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                                <p className="text-gray-400 text-xs uppercase mb-1">Model</p>
                                <p className="text-lg font-bold text-accent">EfficientNet-B3</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="glass p-4 rounded-2xl">
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Color Legend</h4>
                <div className="flex items-center gap-4">
                    <div className="flex-1 h-4 rounded-lg" style={{
                        background: 'linear-gradient(to right, #3b82f6, #06b6d4, #22c55e, #eab308, #f97316, #ef4444)'
                    }} />
                </div>
                <div className="flex justify-between mt-2 text-xs text-gray-400">
                    <span>Low Importance</span>
                    <span>Medium</span>
                    <span>High Importance</span>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                    Red/yellow regions indicate areas the model focused on when making its prediction.
                    These typically highlight ridge patterns, core formations, and delta points.
                </p>
            </div>
        </div>
    );
}
