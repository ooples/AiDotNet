using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents a partial dependence plot for feature analysis
    /// </summary>
    public class PartialDependencePlot
    {
        /// <summary>
        /// Gets or sets the feature index
        /// </summary>
        public int FeatureIndex { get; set; }
        
        /// <summary>
        /// Gets or sets the feature name
        /// </summary>
        public string FeatureName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the x values (feature values)
        /// </summary>
        public List<double> XValues { get; set; } = new List<double>();
        
        /// <summary>
        /// Gets or sets the y values (average predictions)
        /// </summary>
        public List<double> YValues { get; set; } = new List<double>();
        
        /// <summary>
        /// Gets or sets the confidence intervals
        /// </summary>
        public List<double> ConfidenceIntervals { get; set; } = new List<double>();
    }
}