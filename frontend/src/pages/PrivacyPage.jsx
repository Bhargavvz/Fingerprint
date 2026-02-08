import React from 'react';
import { Link } from 'react-router-dom';

export default function PrivacyPage() {
    return (
        <div className="container mx-auto px-4 py-12 max-w-3xl">
            <h1 className="text-4xl font-bold mb-8 gradient-text fade-in">
                Privacy Policy
            </h1>

            <div className="glass p-8 fade-in" style={{ animationDelay: '0.1s' }}>
                <p className="text-gray-500 text-sm mb-6">Last updated: February 2026</p>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">1. Information We Collect</h2>
                    <p className="text-gray-300 leading-relaxed mb-4">
                        BloodType AI is designed with privacy in mind. When you use our fingerprint detection service:
                    </p>
                    <ul className="list-disc list-inside text-gray-400 space-y-2">
                        <li>Fingerprint images are processed in real-time and are <strong className="text-white">NOT stored</strong> on our servers</li>
                        <li>We do not collect personal identification information</li>
                        <li>No cookies are used for tracking purposes</li>
                        <li>We may collect anonymous usage statistics to improve the service</li>
                    </ul>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">2. How We Use Information</h2>
                    <p className="text-gray-300 leading-relaxed mb-4">
                        Any information collected is used solely for:
                    </p>
                    <ul className="list-disc list-inside text-gray-400 space-y-2">
                        <li>Providing the blood group prediction service</li>
                        <li>Improving model accuracy and user experience</li>
                        <li>Technical debugging and service maintenance</li>
                    </ul>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">3. Data Security</h2>
                    <p className="text-gray-300 leading-relaxed">
                        We implement appropriate security measures to protect your information. Fingerprint images are transmitted securely and processed without permanent storage.
                    </p>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">4. Third-Party Services</h2>
                    <p className="text-gray-300 leading-relaxed">
                        This application may use third-party services for hosting and analytics. These services have their own privacy policies.
                    </p>
                </section>

                <section className="mb-8">
                    <h2 className="text-xl font-semibold text-white mb-4">5. Research Purpose</h2>
                    <p className="text-gray-300 leading-relaxed">
                        This is an academic research project. Results and aggregated, anonymized data may be used for academic publications and research purposes.
                    </p>
                </section>

                <section>
                    <h2 className="text-xl font-semibold text-white mb-4">6. Contact</h2>
                    <p className="text-gray-300 leading-relaxed">
                        For privacy concerns, please contact us through our{' '}
                        <a href="https://github.com/Bhargavvz/Fingerprint" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                            GitHub repository
                        </a>.
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
