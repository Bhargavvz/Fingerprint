import React from 'react';

const BLOOD_GROUP_COLORS = {
    'A+': { gradient: 'from-red-500 to-red-600', text: 'text-red-500', bg: 'blood-a-plus' },
    'A-': { gradient: 'from-orange-500 to-orange-600', text: 'text-orange-500', bg: 'blood-a-minus' },
    'B+': { gradient: 'from-blue-500 to-blue-600', text: 'text-blue-500', bg: 'blood-b-plus' },
    'B-': { gradient: 'from-indigo-500 to-indigo-600', text: 'text-indigo-500', bg: 'blood-b-minus' },
    'AB+': { gradient: 'from-violet-500 to-violet-600', text: 'text-violet-500', bg: 'blood-ab-plus' },
    'AB-': { gradient: 'from-purple-500 to-purple-600', text: 'text-purple-500', bg: 'blood-ab-minus' },
    'O+': { gradient: 'from-green-500 to-green-600', text: 'text-green-500', bg: 'blood-o-plus' },
    'O-': { gradient: 'from-teal-500 to-teal-600', text: 'text-teal-500', bg: 'blood-o-minus' },
};

const BLOOD_GROUP_INFO = {
    'A+': { canGiveTo: ['A+', 'AB+'], canReceiveFrom: ['A+', 'A-', 'O+', 'O-'], population: '35.7%' },
    'A-': { canGiveTo: ['A+', 'A-', 'AB+', 'AB-'], canReceiveFrom: ['A-', 'O-'], population: '6.3%' },
    'B+': { canGiveTo: ['B+', 'AB+'], canReceiveFrom: ['B+', 'B-', 'O+', 'O-'], population: '8.5%' },
    'B-': { canGiveTo: ['B+', 'B-', 'AB+', 'AB-'], canReceiveFrom: ['B-', 'O-'], population: '1.5%' },
    'AB+': { canGiveTo: ['AB+'], canReceiveFrom: ['All'], population: '3.4%', label: 'Universal Recipient' },
    'AB-': { canGiveTo: ['AB+', 'AB-'], canReceiveFrom: ['A-', 'B-', 'AB-', 'O-'], population: '0.6%' },
    'O+': { canGiveTo: ['A+', 'B+', 'AB+', 'O+'], canReceiveFrom: ['O+', 'O-'], population: '37.4%' },
    'O-': { canGiveTo: ['All'], canReceiveFrom: ['O-'], population: '6.6%', label: 'Universal Donor' },
};

export default function ResultCard({ result }) {
    if (!result) return null;

    const bloodGroup = result.blood_group || result.predicted_class;
    const confidence = result.confidence;
    const probabilities = result.all_probabilities || result.probabilities || {};

    // Guard against undefined bloodGroup
    if (!bloodGroup) return null;

    const colors = BLOOD_GROUP_COLORS[bloodGroup] || BLOOD_GROUP_COLORS['O+'];
    const info = BLOOD_GROUP_INFO[bloodGroup] || {};

    const sortedProbs = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    return (
        <div className="result-card">
            {/* Header */}
            <div className="text-center mb-8">
                <p className="text-gray-400 text-sm uppercase tracking-wider mb-2">Predicted Blood Group</p>

                {/* Large Blood Group Badge */}
                <div className={`inline-flex items-center justify-center w-32 h-32 rounded-3xl bg-gradient-to-br ${colors.gradient} shadow-2xl mb-4`}>
                    <span className="text-5xl font-black text-white">{bloodGroup}</span>
                </div>

                {/* Special Label */}
                {info.label && (
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-yellow-500/20 border border-yellow-500/30 mb-2">
                        <span className="text-yellow-400">⭐</span>
                        <span className="text-yellow-300 text-sm font-medium">{info.label}</span>
                    </div>
                )}

                {/* Confidence */}
                <div className="mt-4">
                    <p className="text-gray-400 text-sm mb-2">Confidence Level</p>
                    <div className="flex items-center justify-center gap-4">
                        <div className="w-48">
                            <div className="confidence-bar">
                                <div
                                    className={`confidence-fill bg-gradient-to-r ${colors.gradient}`}
                                    style={{ width: `${confidence * 100}%` }}
                                />
                            </div>
                        </div>
                        <span className={`text-2xl font-bold ${colors.text}`}>
                            {(confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>

            {/* Blood Group Info */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="glass p-4 rounded-xl text-center">
                    <p className="text-gray-400 text-xs uppercase mb-2">Population</p>
                    <p className="text-2xl font-bold text-white">{info.population || 'N/A'}</p>
                </div>
                <div className="glass p-4 rounded-xl text-center">
                    <p className="text-gray-400 text-xs uppercase mb-2">Rh Factor</p>
                    <p className="text-2xl font-bold text-white">
                        {bloodGroup.includes('+') ? 'Positive' : 'Negative'}
                    </p>
                </div>
            </div>

            {/* Compatibility Info */}
            <div className="glass p-4 rounded-xl mb-6">
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Compatibility Overview</h4>
                <div className="space-y-3">
                    <div>
                        <p className="text-xs text-gray-400 mb-1">Can Donate To:</p>
                        <div className="flex flex-wrap gap-2">
                            {(Array.isArray(info.canGiveTo) ? info.canGiveTo : ['All']).map((bg) => (
                                <span key={bg} className={`px-2 py-1 rounded-lg text-xs font-medium ${bg === 'All' ? 'bg-green-500/20 text-green-300' : 'bg-white/10 text-white'
                                    }`}>
                                    {bg}
                                </span>
                            ))}
                        </div>
                    </div>
                    <div>
                        <p className="text-xs text-gray-400 mb-1">Can Receive From:</p>
                        <div className="flex flex-wrap gap-2">
                            {(Array.isArray(info.canReceiveFrom) ? info.canReceiveFrom : ['All']).map((bg) => (
                                <span key={bg} className={`px-2 py-1 rounded-lg text-xs font-medium ${bg === 'All' ? 'bg-blue-500/20 text-blue-300' : 'bg-white/10 text-white'
                                    }`}>
                                    {bg}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Probability Distribution */}
            <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Probability Distribution</h4>
                <div className="space-y-2">
                    {sortedProbs.map(([group, prob]) => {
                        const groupColors = BLOOD_GROUP_COLORS[group] || colors;
                        const isTop = group === bloodGroup;

                        return (
                            <div key={group} className="prob-bar">
                                <span className={`prob-label ${isTop ? 'text-white font-bold' : 'text-gray-400'}`}>
                                    {group}
                                </span>
                                <div className="prob-track">
                                    <div
                                        className={`prob-fill bg-gradient-to-r ${groupColors.gradient} ${isTop ? 'opacity-100' : 'opacity-60'}`}
                                        style={{ width: `${prob * 100}%` }}
                                    />
                                </div>
                                <span className={`prob-value ${isTop ? 'text-white font-semibold' : ''}`}>
                                    {(prob * 100).toFixed(1)}%
                                </span>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
