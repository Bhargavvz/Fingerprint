import React from 'react';
import { Link } from 'react-router-dom';

export default function TermsPage() {
    return (
        <div className="container mx-auto px-4 py-12 max-w-3xl">
            <h1 className="text-4xl font-bold mb-8 gradient-text fade-in">
                Terms of Use
            </h1>

            <div className="glass p-8 fade-in" style={{ animationDelay: '0.1s' }}>
                <p className="text-gray-500 text-sm mb-6">Last updated: February 2026</p>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">1. Acceptance of Terms</h2>
                    <p className="text-gray-300 leading-relaxed">
                        By accessing and using BloodType AI, you accept and agree to be bound by these Terms of Use. If you do not agree, please do not use this service.
                    </p>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">2. Research Purpose Disclaimer</h2>
                    <div className="p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30 mb-4">
                        <p className="text-yellow-300 font-medium">
                            ⚠️ This is an academic research project. Predictions are for educational and research purposes ONLY.
                        </p>
                    </div>
                    <p className="text-gray-300 leading-relaxed mb-4">
                        This application is NOT a medical device and should NOT be used for:
                    </p>
                    <ul className="list-disc list-inside text-gray-400 space-y-2">
                        <li>Medical diagnosis or treatment decisions</li>
                        <li>Blood transfusion compatibility determination</li>
                        <li>Any clinical or healthcare purposes</li>
                        <li>Legal identification or verification</li>
                    </ul>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">3. No Medical Advice</h2>
                    <p className="text-gray-300 leading-relaxed">
                        The predictions provided by this system do not constitute medical advice. Blood group determination must be performed through proper laboratory testing by qualified healthcare professionals.
                    </p>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">4. Accuracy Limitations</h2>
                    <p className="text-gray-300 leading-relaxed">
                        While our model achieves reasonable accuracy in research settings, AI predictions are not 100% accurate. The correlation between fingerprints and blood groups is a subject of ongoing scientific investigation.
                    </p>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">5. Limitation of Liability</h2>
                    <p className="text-gray-300 leading-relaxed">
                        The developers and researchers are not liable for any damages or consequences arising from the use or misuse of this application. Use at your own risk.
                    </p>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">6. Intellectual Property</h2>
                    <p className="text-gray-300 leading-relaxed">
                        This project is open source and available under the terms specified in the{' '}
                        <a href="https://github.com/Bhargavvz/Fingerprint" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                            GitHub repository
                        </a>.
                    </p>
                </section>

                <section>
                    <h2 className="text-xl font-semibold text-white mb-4">7. Changes to Terms</h2>
                    <p className="text-gray-300 leading-relaxed">
                        We reserve the right to modify these terms at any time. Continued use of the service constitutes acceptance of updated terms.
                    </p>
                </section>
            </div>

            <div className="mt-8 fade-in" style={{ animationDelay: '0.2s' }}>
                <Link to="/" className="text-primary hover:underline">
                    ← Back to Home
                </Link>
            </div>
        </div>
    );
}
