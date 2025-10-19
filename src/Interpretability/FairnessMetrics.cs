using System.Collections.Generic;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Represents fairness metrics for model evaluation.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class FairnessMetrics<T>
    {
        /// <summary>
        /// Gets or sets the demographic parity metric value.
        /// </summary>
        public T DemographicParity { get; set; }

        /// <summary>
        /// Gets or sets the equal opportunity metric value.
        /// </summary>
        public T EqualOpportunity { get; set; }

        /// <summary>
        /// Gets or sets the equalized odds metric value.
        /// </summary>
        public T EqualizedOdds { get; set; }

        /// <summary>
        /// Gets or sets the predictive parity metric value.
        /// </summary>
        public T PredictiveParity { get; set; }

        /// <summary>
        /// Gets or sets the disparate impact metric value.
        /// </summary>
        public T DisparateImpact { get; set; }

        /// <summary>
        /// Gets or sets the statistical parity difference metric value.
        /// </summary>
        public T StatisticalParityDifference { get; set; }

        /// <summary>
        /// Gets or sets additional fairness metrics.
        /// </summary>
        public Dictionary<string, T> AdditionalMetrics { get; set; }

        /// <summary>
        /// Gets or sets the sensitive feature index used for fairness evaluation.
        /// </summary>
        public int SensitiveFeatureIndex { get; set; }

        /// <summary>
        /// Initializes a new instance of the FairnessMetrics class.
        /// </summary>
        public FairnessMetrics()
        {
            AdditionalMetrics = new Dictionary<string, T>();
        }
    }
}
