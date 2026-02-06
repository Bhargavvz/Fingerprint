import { RefreshCw, Droplet } from 'lucide-react'

function ResultCard({ result, uploadedImage, onReset }) {
    const getBloodGroupColor = (group) => {
        if (group.startsWith('A')) return 'from-red-500 to-red-600'
        if (group.startsWith('B')) return 'from-blue-500 to-blue-600'
        if (group.startsWith('AB')) return 'from-purple-500 to-purple-600'
        return 'from-green-500 to-green-600'
    }

    const getBloodGroupGlow = (group) => {
        if (group.startsWith('A')) return 'shadow-red-500/30'
        if (group.startsWith('B')) return 'shadow-blue-500/30'
        if (group.startsWith('AB')) return 'shadow-purple-500/30'
        return 'shadow-green-500/30'
    }

    const confidencePercent = (result.confidence * 100).toFixed(1)

    return (
        <div className="glass rounded-2xl p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">Prediction Result</h2>
                <button
                    onClick={onReset}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-slate-300 hover:text-white"
                >
                    <RefreshCw className="w-4 h-4" />
                    New Analysis
                </button>
            </div>

            {/* Main Result */}
            <div className="flex items-center gap-6">
                {/* Uploaded Image */}
                {uploadedImage && (
                    <div className="w-24 h-24 rounded-xl overflow-hidden border border-white/10 flex-shrink-0">
                        <img
                            src={uploadedImage}
                            alt="Uploaded fingerprint"
                            className="w-full h-full object-cover"
                        />
                    </div>
                )}

                {/* Blood Group Badge */}
                <div className={`
          flex items-center gap-4 px-6 py-4 rounded-2xl
          bg-gradient-to-r ${getBloodGroupColor(result.blood_group)}
          shadow-lg ${getBloodGroupGlow(result.blood_group)}
        `}>
                    <Droplet className="w-8 h-8 text-white" />
                    <div>
                        <div className="text-3xl font-bold text-white">{result.blood_group}</div>
                        <div className="text-white/80 text-sm">Blood Group</div>
                    </div>
                </div>
            </div>

            {/* Confidence */}
            <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">Confidence</span>
                    <span className="text-white font-medium">{confidencePercent}%</span>
                </div>
                <div className="confidence-meter">
                    <div
                        className={`confidence-fill bg-gradient-to-r ${getBloodGroupColor(result.blood_group)}`}
                        style={{ width: `${confidencePercent}%` }}
                    />
                </div>
            </div>

            {/* All Probabilities */}
            <div className="space-y-3">
                <h3 className="text-sm font-medium text-slate-400">All Probabilities</h3>
                <div className="grid grid-cols-4 gap-2">
                    {Object.entries(result.all_probabilities || {})
                        .sort((a, b) => b[1] - a[1])
                        .map(([group, prob]) => (
                            <div
                                key={group}
                                className={`
                  px-3 py-2 rounded-lg text-center
                  ${group === result.blood_group
                                        ? `bg-gradient-to-r ${getBloodGroupColor(group)} text-white`
                                        : 'bg-white/5 text-slate-400'}
                `}
                            >
                                <div className="font-semibold">{group}</div>
                                <div className="text-xs opacity-80">{(prob * 100).toFixed(1)}%</div>
                            </div>
                        ))}
                </div>
            </div>

            {/* Disclaimer */}
            <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
                <p className="text-amber-400/90 text-sm text-center">
                    {result.disclaimer}
                </p>
            </div>
        </div>
    )
}

export default ResultCard
