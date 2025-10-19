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
        /// Initializes a new instance of the FairnessMetrics class with all metric values.
        /// </summary>
        /// <param name="demographicParity">The demographic parity metric value.</param>
        /// <param name="equalOpportunity">The equal opportunity metric value.</param>
        /// <param name="equalizedOdds">The equalized odds metric value.</param>
        /// <param name="predictiveParity">The predictive parity metric value.</param>
        /// <param name="disparateImpact">The disparate impact metric value.</param>
        /// <param name="statisticalParityDifference">The statistical parity difference metric value.</param>
        /// <exception cref="ArgumentNullException">Thrown when any parameter is null and T is a reference type.</exception>
        public FairnessMetrics(
            T demographicParity,
            T equalOpportunity,
            T equalizedOdds,
            T predictiveParity,
            T disparateImpact,
            T statisticalParityDifference)
        {
            // Validate parameters for reference types to prevent null assignment
            if (!typeof(T).IsValueType)
            {
                ArgumentNullException.ThrowIfNull(demographicParity);
                ArgumentNullException.ThrowIfNull(equalOpportunity);
                ArgumentNullException.ThrowIfNull(equalizedOdds);
                ArgumentNullException.ThrowIfNull(predictiveParity);
                ArgumentNullException.ThrowIfNull(disparateImpact);
                ArgumentNullException.ThrowIfNull(statisticalParityDifference);
            }

            DemographicParity = demographicParity;
            EqualOpportunity = equalOpportunity;
            EqualizedOdds = equalizedOdds;
            PredictiveParity = predictiveParity;
            DisparateImpact = disparateImpact;
            StatisticalParityDifference = statisticalParityDifference;
            AdditionalMetrics = new Dictionary<string, T>();
        }
    }
}
