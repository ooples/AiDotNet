using System.Collections.Generic;

namespace AiDotNet.Deployment.Techniques
{
    /// <summary>
    /// Strategy recommendation for quantization.
    /// </summary>
    public class StrategyRecommendation
    {
        /// <summary>
        /// Gets or sets the strategy name.
        /// </summary>
        public string StrategyName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the expected compression ratio.
        /// </summary>
        public double ExpectedCompressionRatio { get; set; }
        
        /// <summary>
        /// Gets or sets the expected accuracy drop (as a fraction).
        /// </summary>
        public double ExpectedAccuracyDrop { get; set; }
        
        /// <summary>
        /// Gets or sets the expected speedup factor.
        /// </summary>
        public double ExpectedSpeedup { get; set; }
        
        /// <summary>
        /// Gets or sets any warnings about this strategy.
        /// </summary>
        public List<string> Warnings { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets additional metadata about this strategy.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }
}