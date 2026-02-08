import React from 'react';

export default function AnimatedBackground() {
    return (
        <div className="fixed inset-0 z-0 overflow-hidden">
            {/* Base Gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-indigo-950/50 to-slate-950" />

            {/* Static Orbs - no animation */}
            <div
                className="absolute w-[500px] h-[500px] rounded-full blur-[100px] opacity-20"
                style={{
                    background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                    top: '-10%',
                    left: '-10%'
                }}
            />
            <div
                className="absolute w-[400px] h-[400px] rounded-full blur-[100px] opacity-15"
                style={{
                    background: 'linear-gradient(135deg, #ec4899, #f43f5e)',
                    bottom: '-5%',
                    right: '-5%'
                }}
            />
            <div
                className="absolute w-[300px] h-[300px] rounded-full blur-[80px] opacity-10"
                style={{
                    background: 'linear-gradient(135deg, #06b6d4, #22d3ee)',
                    top: '40%',
                    left: '50%',
                    transform: 'translateX(-50%)'
                }}
            />

            {/* Subtle Grid Pattern */}
            <div
                className="absolute inset-0 opacity-[0.03]"
                style={{
                    backgroundImage: `
                        linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)
                    `,
                    backgroundSize: '80px 80px'
                }}
            />
        </div>
    );
}
