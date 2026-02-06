import { AlertTriangle } from 'lucide-react'

function DisclaimerBanner() {
    return (
        <div className="bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10 border-b border-amber-500/20">
            <div className="max-w-6xl mx-auto px-4 py-3">
                <div className="flex items-center justify-center gap-2 text-amber-400 text-sm">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                    <p>
                        <strong>Disclaimer:</strong> This is an academic research project.
                        Not intended for medical diagnosis. Always consult healthcare professionals.
                    </p>
                </div>
            </div>
        </div>
    )
}

export default DisclaimerBanner
