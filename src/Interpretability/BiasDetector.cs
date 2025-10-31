using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Provides methods for detecting bias in model predictions using Disparate Impact metrics.
    /// Uses the 80% rule: Disparate Impact Ratio < 0.8 indicates bias.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class BiasDetector<T> : BiasDetectorBase<T>
    {
        /// <summary>
        /// Initializes a new instance of the BiasDetector class.
        /// </summary>
        public BiasDetector() : base(isLowerBiasBetter: true)
        {
        }

        /// <summary>
        /// Implements bias detection logic using Disparate Impact metrics.
        /// </summary>
        protected override BiasDetectionResult<T> GetBiasDetectionResult(
            Vector<T> predictions,
            Vector<T> sensitiveFeature,
            Vector<T>? actualLabels)
        {
            var result = new BiasDetectionResult<T>();

            // Identify unique groups in the sensitive feature
            var groups = InterpretabilityMetricsHelper<T>.GetUniqueGroups(sensitiveFeature);

            if (groups.Count < 2)
            {
                result.HasBias = false;
                result.Message = "Insufficient groups for bias detection. At least 2 groups are required.";
                return result;
            }

            // Compute positive prediction rates for each group
            var groupPositiveRates = new Dictionary<string, T>();
            var groupSizes = new Dictionary<string, int>();

            foreach (var group in groups)
            {
                var groupIndices = InterpretabilityMetricsHelper<T>.GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = InterpretabilityMetricsHelper<T>.GetSubset(predictions, groupIndices);
                var positiveRate = InterpretabilityMetricsHelper<T>.ComputePositiveRate(groupPredictions);
                string groupKey = group?.ToString() ?? "unknown";

                groupPositiveRates[groupKey] = positiveRate;
                groupSizes[groupKey] = groupIndices.Count;
            }

            result.GroupPositiveRates = groupPositiveRates;
            result.GroupSizes = groupSizes;

            // Compute disparate impact
            var orderedRates = groupPositiveRates.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
            T minRate = orderedRates.First();
            T maxRate = orderedRates.Last();

            // Avoid division by zero
            if (_numOps.Equals(maxRate, _numOps.Zero))
            {
                result.DisparateImpactRatio = _numOps.One;
            }
            else
            {
                result.DisparateImpactRatio = _numOps.Divide(minRate, maxRate);
            }

            // Compute statistical parity difference
            result.StatisticalParityDifference = _numOps.Subtract(maxRate, minRate);

            // Check for bias using 80% rule (disparate impact < 0.8 indicates bias)
            double disparateImpactValue = Convert.ToDouble(result.DisparateImpactRatio);
            double statisticalParityValue = Convert.ToDouble(result.StatisticalParityDifference);

            result.HasBias = disparateImpactValue < 0.8 || Math.Abs(statisticalParityValue) > 0.1;

            if (result.HasBias)
            {
                result.Message = $"Bias detected: Disparate impact ratio = {disparateImpactValue:F3}, Statistical parity difference = {statisticalParityValue:F3}";
            }
            else
            {
                result.Message = "No significant bias detected based on disparate impact and statistical parity.";
            }

            // If actual labels are provided, compute additional metrics
            if (actualLabels != null)
            {
                ComputeAdditionalMetrics(predictions, sensitiveFeature, actualLabels, groups, result);
            }

            return result;
        }

        /// <summary>
        /// Computes additional bias metrics when actual labels are available.
        /// </summary>
        private void ComputeAdditionalMetrics(Vector<T> predictions, Vector<T> sensitiveFeature,
            Vector<T> actualLabels, List<T> groups, BiasDetectionResult<T> result)
        {
            var groupTPRs = new Dictionary<string, T>();
            var groupFPRs = new Dictionary<string, T>();
            var groupPrecisions = new Dictionary<string, T>();

            foreach (var group in groups)
            {
                var groupIndices = InterpretabilityMetricsHelper<T>.GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = InterpretabilityMetricsHelper<T>.GetSubset(predictions, groupIndices);
                var groupActualLabels = InterpretabilityMetricsHelper<T>.GetSubset(actualLabels, groupIndices);
                string groupKey = group?.ToString() ?? "unknown";

                // Compute True Positive Rate (TPR)
                var tpr = InterpretabilityMetricsHelper<T>.ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                groupTPRs[groupKey] = tpr;

                // Compute False Positive Rate (FPR)
                var fpr = InterpretabilityMetricsHelper<T>.ComputeFalsePositiveRate(groupPredictions, groupActualLabels);
                groupFPRs[groupKey] = fpr;

                // Compute Precision
                var precision = InterpretabilityMetricsHelper<T>.ComputePrecision(groupPredictions, groupActualLabels);
                groupPrecisions[groupKey] = precision;
            }

            result.GroupTruePositiveRates = groupTPRs;
            result.GroupFalsePositiveRates = groupFPRs;
            result.GroupPrecisions = groupPrecisions;

            // Compute equal opportunity difference (max TPR difference)
            if (groupTPRs.Count >= 2)
            {
                var orderedTPRs = groupTPRs.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
                T minTPR = orderedTPRs.First();
                T maxTPR = orderedTPRs.Last();
                result.EqualOpportunityDifference = _numOps.Subtract(maxTPR, minTPR);

                double eoValue = Convert.ToDouble(result.EqualOpportunityDifference);
                if (Math.Abs(eoValue) > 0.1)
                {
                    result.HasBias = true;
                    result.Message += $" Equal opportunity difference = {eoValue:F3}.";
                }
            }
        }
    }

    /// <summary>
    /// Represents the result of a bias detection analysis.
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
