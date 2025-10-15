using System.Collections.Generic;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents a partial dependence plot for a feature
    /// </summary>
    public class PartialDependencePlot
    {
        /// <summary>
        /// Index of the feature
        /// </summary>
        public int FeatureIndex { get; set; }

        /// <summary>
        /// Name of the feature
        /// </summary>
        public string FeatureName { get; set; } = string.Empty;

        /// <summary>
        /// Feature values (x-axis)
        /// </summary>
        public List<double> XValues { get; set; } = new List<double>();

        /// <summary>
        /// Average predictions (y-axis)
        /// </summary>
        public List<double> YValues { get; set; } = new List<double>();

        /// <summary>
        /// Confidence intervals for each point
        /// </summary>
        public List<double> ConfidenceIntervals { get; set; } = new List<double>();

        /// <summary>
        /// Lower confidence bounds
        /// </summary>
        public List<double> LowerBounds { get; set; } = new List<double>();

        /// <summary>
        /// Upper confidence bounds
        /// </summary>
        public List<double> UpperBounds { get; set; } = new List<double>();

        /// <summary>
        /// Number of samples used for each point
        /// </summary>
        public List<int> SampleCounts { get; set; } = new List<int>();
    }
}