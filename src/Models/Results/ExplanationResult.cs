using System.Collections.Generic;

namespace AiDotNet.Models.Results
{
    /// <summary>
    /// Result of a model explanation showing how features contributed to a prediction
    /// </summary>
    public class ExplanationResult
    {
        /// <summary>
        /// Gets or sets the prediction value
        /// </summary>
        public double Prediction { get; set; }
        
        /// <summary>
        /// Gets or sets the confidence level of the prediction (0-1)
        /// </summary>
        public double Confidence { get; set; }
        
        /// <summary>
        /// Gets or sets the contribution of each feature to the prediction
        /// </summary>
        public Dictionary<int, double> FeatureContributions { get; set; } = new Dictionary<int, double>();
        
        /// <summary>
        /// Gets or sets the type of explanation method used
        /// </summary>
        public string ExplanationType { get; set; } = string.Empty;
    }
}