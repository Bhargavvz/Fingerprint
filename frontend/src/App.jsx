import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import AnimatedBackground from './components/AnimatedBackground';
import LandingPage from './pages/LandingPage';
import DetectionPage from './pages/DetectionPage';
import AboutPage from './pages/AboutPage';
import ResearchPage from './pages/ResearchPage';
import PrivacyPage from './pages/PrivacyPage';
import TermsPage from './pages/TermsPage';

function App() {
    return (
        <Router>
            <div className="min-h-screen relative">
                {/* Animated Background */}
                <AnimatedBackground />

                {/* Header */}
                <Header />

                {/* Main Content */}
                <main className="relative z-10">
                    <Routes>
                        <Route path="/" element={<LandingPage />} />
                        <Route path="/detect" element={<DetectionPage />} />
                        <Route path="/about" element={<AboutPage />} />
                        <Route path="/research" element={<ResearchPage />} />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                    </Routes>
                </main>

                {/* Footer */}
                <Footer />
            </div>
        </Router>
    );
}

export default App;
