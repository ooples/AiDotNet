using System.Collections.Generic;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents the results of a bias detection analysis.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class BiasDetectionResult<T>
    {
        /// <summary>
        /// Gets or sets whether bias was detected.
        /// </summary>
        public bool HasBias { get; set; }

        /// <summary>
        /// Gets or sets the message describing the bias detection results.
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// Gets or sets the positive prediction rates for each group.
        /// </summary>
        public Dictionary<string, T> GroupPositiveRates { get; set; }

        /// <summary>
        /// Gets or sets the sizes of each group.
        /// </summary>
        public Dictionary<string, int> GroupSizes { get; set; }

        /// <summary>
        /// Gets or sets the disparate impact ratio (min rate / max rate).
        /// </summary>
        public T DisparateImpactRatio { get; set; }

        /// <summary>
        /// Gets or sets the statistical parity difference (max rate - min rate).
        /// </summary>
        public T StatisticalParityDifference { get; set; }

        /// <summary>
        /// Gets or sets the equal opportunity difference (max TPR - min TPR).
        /// </summary>
        public T EqualOpportunityDifference { get; set; }

        /// <summary>
        /// Gets or sets the True Positive Rates for each group.
        /// </summary>
        public Dictionary<string, T> GroupTruePositiveRates { get; set; }

        /// <summary>
        /// Gets or sets the False Positive Rates for each group.
        /// </summary>
        public Dictionary<string, T> GroupFalsePositiveRates { get; set; }

        /// <summary>
        /// Gets or sets the Precision values for each group.
        /// </summary>
        public Dictionary<string, T> GroupPrecisions { get; set; }

        /// <summary>
        /// Initializes a new instance of the BiasDetectionResult class.
        /// </summary>
        public BiasDetectionResult()
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            Message = string.Empty;
            GroupPositiveRates = new Dictionary<string, T>();
            GroupSizes = new Dictionary<string, int>();
            GroupTruePositiveRates = new Dictionary<string, T>();
            GroupFalsePositiveRates = new Dictionary<string, T>();
            GroupPrecisions = new Dictionary<string, T>();
            DisparateImpactRatio = numOps.Zero;
            StatisticalParityDifference = numOps.Zero;
            EqualOpportunityDifference = numOps.Zero;
        }
    }
}
