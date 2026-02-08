import React from 'react';
import { useDropzone } from 'react-dropzone';

export default function FileUpload({ onDrop, loading, preview }) {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.gif']
        },
        maxFiles: 1,
        disabled: loading
    });

    return (
        <div
            {...getRootProps()}
            className={`upload-zone relative ${isDragActive ? 'dragover' : ''} ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
            <input {...getInputProps()} />

            {loading ? (
                <div className="flex flex-col items-center gap-6">
                    {/* Loading Animation */}
                    <div className="relative">
                        <div className="spinner"></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-2xl">🔬</span>
                        </div>
                    </div>

                    <div>
                        <p className="text-xl font-semibold text-white mb-2">Analyzing Fingerprint...</p>
                        <div className="loading-dots justify-center">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>

                    <p className="text-gray-400 text-sm">
                        Our AI is examining ridge patterns and minutiae features
                    </p>
                </div>
            ) : preview ? (
                <div className="flex flex-col items-center gap-6">
                    {/* Preview Image */}
                    <div className="relative group">
                        <img
                            src={preview}
                            alt="Fingerprint preview"
                            className="max-h-48 rounded-xl object-contain shadow-2xl transition-transform group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent rounded-xl opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-center pb-4">
                            <span className="text-white text-sm">Click or drop to replace</span>
                        </div>
                    </div>

                    <div>
                        <p className="text-lg font-medium text-white">Image Ready</p>
                        <p className="text-gray-400 text-sm mt-1">Click to upload a different image</p>
                    </div>
                </div>
            ) : (
                <div className="flex flex-col items-center gap-6">
                    {/* Upload Icon */}
                    <div className="relative">
                        <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                            <span className="text-5xl">{isDragActive ? '📥' : '👆'}</span>
                        </div>
                        {/* Pulse Ring */}
                        <div className="absolute inset-0 rounded-2xl border-2 border-primary/40 animate-ping opacity-20"></div>
                    </div>

                    <div>
                        <p className="text-xl font-semibold text-white mb-2">
                            {isDragActive ? 'Drop your fingerprint here!' : 'Upload Fingerprint Image'}
                        </p>
                        <p className="text-gray-400">
                            Drag & drop or <span className="text-primary font-medium">browse files</span>
                        </p>
                    </div>

                    {/* Supported Formats */}
                    <div className="flex flex-wrap justify-center gap-2">
                        {['JPG', 'PNG', 'BMP', 'GIF'].map((format) => (
                            <span
                                key={format}
                                className="px-3 py-1 rounded-full bg-white/5 text-gray-400 text-xs font-medium"
                            >
                                .{format}
                            </span>
                        ))}
                    </div>

                    {/* Tips */}
                    <div className="flex gap-6 text-xs text-gray-500 mt-4">
                        <div className="flex items-center gap-2">
                            <span>✨</span>
                            <span>High resolution preferred</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span>🔒</span>
                            <span>Processed locally</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
