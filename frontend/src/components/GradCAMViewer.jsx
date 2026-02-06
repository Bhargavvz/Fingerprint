import { useState } from 'react'
import { Eye, Info, Layers } from 'lucide-react'

function GradCAMViewer({ originalImage, gradcamImage, explanation }) {
    const [showOriginal, setShowOriginal] = useState(false)

    return (
        <div className="glass rounded-2xl p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/20">
                    <Layers className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                    <h2 className="text-xl font-semibold text-white">Explainable AI</h2>
                    <p className="text-sm text-slate-400">Grad-CAM Visualization</p>
                </div>
            </div>

            {/* Image Viewer */}
            <div className="relative aspect-square rounded-xl overflow-hidden bg-slate-800/50 border border-white/10">
                {gradcamImage ? (
                    <img
                        src={showOriginal ? originalImage : gradcamImage}
                        alt={showOriginal ? 'Original fingerprint' : 'Grad-CAM heatmap'}
                        className="w-full h-full object-contain"
                    />
                ) : originalImage ? (
                    <img
                        src={originalImage}
                        alt="Original fingerprint"
                        className="w-full h-full object-contain"
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-slate-500">
                        No image available
                    </div>
                )}

                {/* Toggle Button */}
                {gradcamImage && (
                    <button
                        onClick={() => setShowOriginal(!showOriginal)}
                        className="absolute bottom-4 right-4 flex items-center gap-2 px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm text-white text-sm hover:bg-black/80 transition-colors"
                    >
                        <Eye className="w-4 h-4" />
                        {showOriginal ? 'Show Heatmap' : 'Show Original'}
                    </button>
                )}

                {/* Overlay Labels */}
                <div className="absolute top-4 left-4 flex gap-2">
                    {gradcamImage && (
                        <div className="px-2 py-1 rounded bg-purple-500/80 text-white text-xs font-medium">
                            {showOriginal ? 'Original' : 'Grad-CAM'}
                        </div>
                    )}
                </div>
            </div>

            {/* Explanation */}
            <div className="space-y-3">
                <div className="flex items-center gap-2">
                    <Info className="w-4 h-4 text-blue-400" />
                    <h3 className="text-sm font-medium text-slate-400">Model Explanation</h3>
                </div>
                <p className="text-slate-300 text-sm leading-relaxed">
                    {explanation || 'The model analyzes fingerprint ridge patterns and their unique characteristics to predict blood group. The heatmap shows which regions of the fingerprint the model focused on most heavily for its prediction.'}
                </p>
            </div>

            {/* Color Legend */}
            {gradcamImage && (
                <div className="space-y-2">
                    <h3 className="text-sm font-medium text-slate-400">Heatmap Legend</h3>
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-gradient-to-r from-blue-500 to-cyan-400"></div>
                            <span className="text-xs text-slate-400">Low Importance</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-gradient-to-r from-yellow-500 to-orange-400"></div>
                            <span className="text-xs text-slate-400">Medium</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-gradient-to-r from-red-500 to-red-600"></div>
                            <span className="text-xs text-slate-400">High Importance</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default GradCAMViewer
