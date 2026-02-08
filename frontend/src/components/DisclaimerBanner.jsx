import React from 'react';

export default function DisclaimerBanner() {
    return (
        <div className="disclaimer">
            <div className="disclaimer-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                    <path fillRule="evenodd" d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003zM12 8.25a.75.75 0 01.75.75v3.75a.75.75 0 01-1.5 0V9a.75.75 0 01.75-.75zm0 8.25a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd" />
                </svg>
            </div>
            <div>
                <h4 className="font-semibold text-yellow-300 mb-1">Research Purpose Only</h4>
                <p className="text-sm text-gray-300">
                    This tool is for <strong>educational and research purposes only</strong>.
                    Blood group determination for medical decisions must be performed by certified
                    laboratory professionals using approved serological methods.
                    Do not use this for medical diagnosis, blood transfusion decisions, or any healthcare purposes.
                </p>
            </div>
        </div>
    );
}
