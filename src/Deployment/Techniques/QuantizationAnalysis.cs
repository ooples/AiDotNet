using System.Collections.Generic;

namespace AiDotNet.Deployment.Techniques
{
    /// <summary>
    /// Quantization analysis results.
    /// </summary>
    public class QuantizationAnalysis
    {
        /// <summary>
        /// Gets or sets the original model size in MB.
        /// </summary>
        public double OriginalSize { get; set; }
        
        /// <summary>
        /// Gets or sets the recommended quantization strategy.
        /// </summary>
        public string RecommendedStrategy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the list of supported strategies with their recommendations.
        /// </summary>
        public List<StrategyRecommendation> SupportedStrategies { get; set; } = new();
        
        /// <summary>
        /// Gets or sets expected metrics after quantization.
        /// </summary>
        public Dictionary<string, double> ExpectedMetrics { get; set; } = new Dictionary<string, double>();
    }
}