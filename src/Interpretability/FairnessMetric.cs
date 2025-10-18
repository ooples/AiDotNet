namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Enumeration of fairness metrics for model evaluation.
    /// </summary>
    public enum FairnessMetric
    {
        /// <summary>
        /// Demographic parity: equal positive prediction rates across groups.
        /// </summary>
        DemographicParity,

        /// <summary>
        /// Equal opportunity: equal true positive rates across groups.
        /// </summary>
        EqualOpportunity,

        /// <summary>
        /// Equalized odds: equal true positive and false positive rates across groups.
        /// </summary>
        EqualizedOdds,

        /// <summary>
        /// Predictive parity: equal precision across groups.
        /// </summary>
        PredictiveParity,

        /// <summary>
        /// Disparate impact: ratio of positive prediction rates between groups.
        /// </summary>
        DisparateImpact,

        /// <summary>
        /// Statistical parity difference: difference in positive prediction rates.
        /// </summary>
        StatisticalParityDifference
    }
}
