using System.Collections.Generic;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents the result of an explainability analysis
    /// </summary>
    public class ExplanationResult
    {
        /// <summary>
        /// The predicted value
        /// </summary>
        public double Prediction { get; set; }

        /// <summary>
        /// Confidence level of the prediction (0-1)
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Feature contributions to the prediction
        /// </summary>
        public Dictionary<int, double> FeatureContributions { get; set; } = new Dictionary<int, double>();

        /// <summary>
        /// Type of explanation (e.g., "SHAP", "LIME", "Permutation")
        /// </summary>
        public string ExplanationType { get; set; } = "Unknown";

        /// <summary>
        /// Additional metadata about the explanation
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }
}