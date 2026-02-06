"""
Explanation generation for fingerprint blood group predictions.
Provides human-readable explanations alongside Grad-CAM visualizations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# Fingerprint pattern information for explanations
FINGERPRINT_PATTERNS = {
    "loop": {
        "description": "Loop patterns with ridges entering from one side and exiting the other",
        "features": ["curved ridges", "single delta point", "asymmetric flow"]
    },
    "whorl": {
        "description": "Whorl patterns with ridges forming circular or spiral formations",
        "features": ["circular ridges", "two delta points", "central core"]
    },
    "arch": {
        "description": "Arch patterns with ridges flowing from one side to another",
        "features": ["wave-like ridges", "no delta points", "smooth curves"]
    }
}

# Blood group characteristics for academic explanation
BLOOD_GROUP_INFO = {
    "A+": {
        "antigen": "A antigen on red cells, Rh positive",
        "antibody": "Anti-B antibodies in plasma",
        "frequency": "Common (~30% of population)"
    },
    "A-": {
        "antigen": "A antigen on red cells, Rh negative",
        "antibody": "Anti-B antibodies in plasma",
        "frequency": "Less common (~6% of population)"
    },
    "B+": {
        "antigen": "B antigen on red cells, Rh positive",
        "antibody": "Anti-A antibodies in plasma",
        "frequency": "Common (~9% of population)"
    },
    "B-": {
        "antigen": "B antigen on red cells, Rh negative",
        "antibody": "Anti-A antibodies in plasma",
        "frequency": "Rare (~2% of population)"
    },
    "AB+": {
        "antigen": "Both A and B antigens, Rh positive",
        "antibody": "No anti-A or anti-B antibodies",
        "frequency": "Less common (~3% of population)"
    },
    "AB-": {
        "antigen": "Both A and B antigens, Rh negative",
        "antibody": "No anti-A or anti-B antibodies",
        "frequency": "Very rare (~1% of population)"
    },
    "O+": {
        "antigen": "No A or B antigens, Rh positive",
        "antibody": "Both anti-A and anti-B antibodies",
        "frequency": "Most common (~39% of population)"
    },
    "O-": {
        "antigen": "No A or B antigens, Rh negative",
        "antibody": "Both anti-A and anti-B antibodies",
        "frequency": "Uncommon (~7% of population)"
    }
}


class ExplanationGenerator:
    """
    Generate human-readable explanations for blood group predictions.
    """
    
    DISCLAIMER = (
        "⚠️ IMPORTANT DISCLAIMER: This is an AI-based research project and should NOT "
        "be used for medical diagnosis. Blood group determination requires proper "
        "laboratory testing by qualified healthcare professionals. This prediction "
        "is for educational and research purposes only."
    )
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Args:
            class_names: List of blood group class names
        """
        self.class_names = class_names or ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    
    def generate(
        self,
        predicted_class: int,
        confidence: float,
        probabilities: Optional[Dict[str, float]] = None,
        heatmap_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a complete explanation for a prediction.
        
        Args:
            predicted_class: Index of predicted blood group
            confidence: Prediction confidence (0-1)
            probabilities: Dictionary of all class probabilities
            heatmap_analysis: Optional analysis of Grad-CAM heatmap
            
        Returns:
            Dictionary containing the explanation
        """
        blood_group = self.class_names[predicted_class]
        
        explanation = {
            "prediction": {
                "blood_group": blood_group,
                "confidence": confidence,
                "confidence_level": self._get_confidence_level(confidence)
            },
            "blood_group_info": BLOOD_GROUP_INFO.get(blood_group, {}),
            "model_reasoning": self._generate_reasoning(blood_group, confidence, heatmap_analysis),
            "limitations": self._get_limitations(),
            "disclaimer": self.DISCLAIMER
        }
        
        if probabilities:
            explanation["all_probabilities"] = probabilities
            explanation["alternative_predictions"] = self._get_alternatives(probabilities, blood_group)
        
        return explanation
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence level."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        elif confidence >= 0.25:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_reasoning(
        self,
        blood_group: str,
        confidence: float,
        heatmap_analysis: Optional[Dict]
    ) -> str:
        """Generate reasoning text for the prediction."""
        
        reasoning_parts = [
            f"The model analyzed the fingerprint ridge patterns and predicted blood group {blood_group} "
            f"with {confidence:.1%} confidence."
        ]
        
        if heatmap_analysis:
            if heatmap_analysis.get("high_activation_regions"):
                regions = heatmap_analysis["high_activation_regions"]
                reasoning_parts.append(
                    f"The model focused primarily on {', '.join(regions)} regions of the fingerprint."
                )
        
        reasoning_parts.append(
            "The prediction is based on learned correlations between fingerprint features "
            "and blood group labels in the training data. This approach has limitations and "
            "should not be considered medically reliable."
        )
        
        return " ".join(reasoning_parts)
    
    def _get_alternatives(
        self,
        probabilities: Dict[str, float],
        predicted: str,
        top_k: int = 3
    ) -> List[Dict]:
        """Get alternative predictions."""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        alternatives = []
        for bg, prob in sorted_probs[:top_k]:
            if bg != predicted:
                alternatives.append({
                    "blood_group": bg,
                    "probability": prob
                })
        
        return alternatives[:top_k-1]
    
    def _get_limitations(self) -> List[str]:
        """Get list of known limitations."""
        return [
            "The correlation between fingerprints and blood groups is not scientifically established",
            "Model accuracy depends on dataset quality and distribution",
            "Environmental factors can affect fingerprint quality",
            "The model may have learned spurious correlations",
            "Results should never be used for medical decisions"
        ]
    
    def generate_summary(
        self,
        predicted_class: int,
        confidence: float
    ) -> str:
        """Generate a brief one-line summary."""
        blood_group = self.class_names[predicted_class]
        conf_level = self._get_confidence_level(confidence)
        
        return f"Predicted blood group: {blood_group} ({conf_level} confidence: {confidence:.1%})"


def analyze_heatmap(heatmap: np.ndarray) -> Dict:
    """
    Analyze a Grad-CAM heatmap to identify regions of interest.
    
    Args:
        heatmap: Grad-CAM heatmap array
        
    Returns:
        Dictionary with heatmap analysis
    """
    # Normalize heatmap
    if heatmap.max() > 1:
        heatmap = heatmap / 255.0
    
    # Find high activation regions
    threshold = 0.5
    high_activation = heatmap > threshold
    
    h, w = heatmap.shape[:2]
    
    # Divide into regions
    regions = {
        "top-left": high_activation[:h//2, :w//2],
        "top-right": high_activation[:h//2, w//2:],
        "bottom-left": high_activation[h//2:, :w//2],
        "bottom-right": high_activation[h//2:, w//2:],
        "center": high_activation[h//4:3*h//4, w//4:3*w//4]
    }
    
    # Identify regions with significant activation
    significant_regions = []
    for name, region in regions.items():
        if region.mean() > 0.3:
            significant_regions.append(name)
    
    # Compute statistics
    analysis = {
        "high_activation_regions": significant_regions if significant_regions else ["distributed"],
        "max_activation": float(heatmap.max()),
        "mean_activation": float(heatmap.mean()),
        "coverage": float((heatmap > 0.3).mean()),  # Percentage of image with significant activation
    }
    
    return analysis


def generate_explanation(
    predicted_class: int,
    confidence: float,
    probabilities: Optional[Dict[str, float]] = None,
    heatmap: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to generate a complete explanation.
    
    Args:
        predicted_class: Index of predicted class
        confidence: Prediction confidence
        probabilities: All class probabilities
        heatmap: Optional Grad-CAM heatmap
        class_names: List of class names
        
    Returns:
        Complete explanation dictionary
    """
    generator = ExplanationGenerator(class_names)
    
    heatmap_analysis = None
    if heatmap is not None:
        heatmap_analysis = analyze_heatmap(heatmap)
    
    return generator.generate(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        heatmap_analysis=heatmap_analysis
    )


if __name__ == "__main__":
    # Test explanation generation
    explanation = generate_explanation(
        predicted_class=0,
        confidence=0.87,
        probabilities={
            "A+": 0.87, "A-": 0.05, "B+": 0.03, "B-": 0.02,
            "AB+": 0.01, "AB-": 0.01, "O+": 0.005, "O-": 0.005
        }
    )
    
    print("Generated Explanation:")
    print(f"\nPrediction: {explanation['prediction']['blood_group']}")
    print(f"Confidence: {explanation['prediction']['confidence']:.1%}")
    print(f"Level: {explanation['prediction']['confidence_level']}")
    
    print(f"\nBlood Group Info:")
    for key, value in explanation['blood_group_info'].items():
        print(f"  {key}: {value}")
    
    print(f"\nModel Reasoning:")
    print(f"  {explanation['model_reasoning']}")
    
    print(f"\nLimitations:")
    for limitation in explanation['limitations']:
        print(f"  - {limitation}")
    
    print(f"\nDisclaimer:")
    print(f"  {explanation['disclaimer']}")
