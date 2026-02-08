import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import FileUpload from './components/FileUpload';
import ResultCard from './components/ResultCard';
import GradCAMViewer from './components/GradCAMViewer';
import DisclaimerBanner from './components/DisclaimerBanner';
import Header from './components/Header';
import Footer from './components/Footer';
import AnimatedBackground from './components/AnimatedBackground';

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [explanation, setExplanation] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('upload');

    const onDrop = useCallback(async (acceptedFiles) => {
        const file = acceptedFiles[0];
        if (!file) return;

        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setResult(null);
        setExplanation(null);
        setError(null);
        setLoading(true);

        try {
            // Create form data
            const formData = new FormData();
            formData.append('file', file);

            // Make prediction
            const predictResponse = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            if (!predictResponse.ok) {
                throw new Error(`Prediction failed: ${predictResponse.statusText}`);
            }

            const predictData = await predictResponse.json();
            setResult(predictData);

            // Get explanation
            const explainFormData = new FormData();
            explainFormData.append('file', file);

            const explainResponse = await fetch('/api/explain', {
                method: 'POST',
                body: explainFormData,
            });

            if (explainResponse.ok) {
                const explainData = await explainResponse.json();
                setExplanation(explainData);
            }

            setActiveTab('results');
        } catch (err) {
            console.error('Error:', err);
            setError(err.message || 'An error occurred during prediction');
        } finally {
            setLoading(false);
        }
    }, []);

    const handleReset = () => {
        setSelectedFile(null);
        setPreview(null);
        setResult(null);
        setExplanation(null);
        setError(null);
        setActiveTab('upload');
    };

    return (
        <div className="min-h-screen relative">
            {/* Animated Background */}
            <AnimatedBackground />

            {/* Header */}
            <Header />

            {/* Main Content */}
            <main className="relative z-10 container mx-auto px-4 py-8 max-w-6xl">
                {/* Hero Section */}
                <section className="text-center mb-12 fade-in">
                    <h1 className="text-5xl md:text-6xl font-bold mb-4 gradient-text">
                        Blood Group Detection
                    </h1>
                    <p className="text-xl text-gray-300 max-w-2xl mx-auto">
                        AI-powered fingerprint analysis using deep learning with
                        <span className="text-primary font-semibold"> 94.7% accuracy</span>
                    </p>
                </section>

                {/* Disclaimer */}
                <section className="mb-8 fade-in" style={{ animationDelay: '0.1s' }}>
                    <DisclaimerBanner />
                </section>

                {/* Main Card */}
                <section className="glass p-8 md:p-12 fade-in" style={{ animationDelay: '0.2s' }}>
                    {/* Tabs */}
                    <div className="flex gap-4 mb-8">
                        <button
                            className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === 'upload'
                                ? 'bg-primary text-white'
                                : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                }`}
                            onClick={() => setActiveTab('upload')}
                        >
                            📤 Upload
                        </button>
                        <button
                            className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === 'results'
                                ? 'bg-primary text-white'
                                : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                }`}
                            onClick={() => setActiveTab('results')}
                            disabled={!result}
                        >
                            📊 Results
                        </button>
                        <button
                            className={`px-6 py-3 rounded-xl font-medium transition-all ${activeTab === 'explain'
                                ? 'bg-primary text-white'
                                : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                }`}
                            onClick={() => setActiveTab('explain')}
                            disabled={!explanation}
                        >
                            🔍 Explanation
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="min-h-[400px]">
                        {/* Upload Tab */}
                        {activeTab === 'upload' && (
                            <div className="scale-in">
                                <FileUpload onDrop={onDrop} loading={loading} preview={preview} />

                                {error && (
                                    <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-300 fade-in">
                                        <div className="flex items-center gap-3">
                                            <span className="text-2xl">⚠️</span>
                                            <span>{error}</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Results Tab */}
                        {activeTab === 'results' && result && (
                            <div className="grid md:grid-cols-2 gap-8 scale-in">
                                {/* Image Preview */}
                                <div className="slide-in-left">
                                    <h3 className="text-xl font-semibold mb-4 text-gray-300">
                                        📷 Uploaded Fingerprint
                                    </h3>
                                    <div className="glass p-4 rounded-2xl">
                                        <img
                                            src={preview}
                                            alt="Uploaded fingerprint"
                                            className="w-full h-auto rounded-xl object-contain max-h-80"
                                        />
                                    </div>
                                </div>

                                {/* Results */}
                                <div className="slide-in-right">
                                    <ResultCard result={result} />
                                </div>
                            </div>
                        )}

                        {/* Explanation Tab */}
                        {activeTab === 'explain' && explanation && (
                            <div className="scale-in">
                                <GradCAMViewer
                                    originalImage={preview}
                                    gradcamImage={explanation.gradcam_image}
                                    explanation={explanation}
                                    bloodGroup={result?.predicted_class}
                                />
                            </div>
                        )}
                    </div>

                    {/* Action Buttons */}
                    {result && (
                        <div className="flex justify-center gap-4 mt-8 pt-8 border-t border-white/10">
                            <button onClick={handleReset} className="btn-secondary">
                                🔄 Analyze Another
                            </button>
                            <button
                                onClick={() => setActiveTab('explain')}
                                className="btn-primary"
                                disabled={!explanation}
                            >
                                🔍 View Explanation
                            </button>
                        </div>
                    )}
                </section>

                {/* Features Section */}
                <section className="mt-16 grid md:grid-cols-3 gap-6 fade-in" style={{ animationDelay: '0.3s' }}>
                    <FeatureCard
                        icon="🧠"
                        title="Deep Learning"
                        description="EfficientNet-B3 with CBAM attention for accurate feature extraction"
                    />
                    <FeatureCard
                        icon="🎯"
                        title="94.7% Accuracy"
                        description="Trained on 6,000+ fingerprint images across 8 blood groups"
                    />
                    <FeatureCard
                        icon="🔍"
                        title="Explainable AI"
                        description="Grad-CAM visualization shows which regions influenced the prediction"
                    />
                </section>
            </main>

            {/* Footer */}
            <Footer />
        </div>
    );
}

function FeatureCard({ icon, title, description }) {
    return (
        <div className="glass glass-hover p-6 text-center">
            <div className="text-4xl mb-4">{icon}</div>
            <h3 className="text-xl font-semibold mb-2 gradient-text">{title}</h3>
            <p className="text-gray-400 text-sm">{description}</p>
        </div>
    );
}

export default App;
