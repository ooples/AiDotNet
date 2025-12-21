using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Detects bias using the Disparate Impact metric (80% rule).
    /// Disparate Impact Ratio = (Min Positive Rate) / (Max Positive Rate).
    /// A ratio below 0.8 indicates potential bias.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class DisparateImpactBiasDetector<T> : BiasDetectorBase<T>
    {
        private readonly double _threshold;

        /// <summary>
        /// Initializes a new instance of the DisparateImpactBiasDetector class.
        /// </summary>
        /// <param name="threshold">The threshold for detecting bias (default is 0.8, representing the 80% rule from fair lending guidelines).</param>
        public DisparateImpactBiasDetector(double threshold = 0.8) : base(isLowerBiasBetter: false)
        {
            if (threshold <= 0 || threshold > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be between 0 and 1.");
            }
            _threshold = threshold;
        }

        /// <summary>
        /// Implements bias detection using Disparate Impact ratio (80% rule).
        /// </summary>
        protected override BiasDetectionResult<T> GetBiasDetectionResult(
            Vector<T> predictions,
            Vector<T> sensitiveFeature,
            Vector<T>? actualLabels)
        {
            if (predictions.Length != sensitiveFeature.Length)
            {
                throw new ArgumentException("Predictions and sensitive feature lengths must match.", nameof(predictions));
            }

            if (actualLabels is not null && actualLabels.Length != predictions.Length)
            {
                throw new ArgumentException("Actual labels and predictions lengths must match.", nameof(actualLabels));
            }

            var result = new BiasDetectionResult<T>();

            // Identify unique groups
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
            var groupTruePositiveRates = new Dictionary<string, T>();
            var groupFalsePositiveRates = new Dictionary<string, T>();
            var groupPrecisions = new Dictionary<string, T>();

            foreach (var group in groups)
            {
                var groupIndices = InterpretabilityMetricsHelper<T>.GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = InterpretabilityMetricsHelper<T>.GetSubset(predictions, groupIndices);
                var positiveRate = InterpretabilityMetricsHelper<T>.ComputePositiveRate(groupPredictions);
                string groupKey = group?.ToString() ?? "unknown";

                groupPositiveRates[groupKey] = positiveRate;
                groupSizes[groupKey] = groupIndices.Count;

                // Compute additional metrics if actual labels are provided
                if (actualLabels is not null)
                {
                    var groupActualLabels = InterpretabilityMetricsHelper<T>.GetSubset(actualLabels, groupIndices);
                    groupTruePositiveRates[groupKey] = InterpretabilityMetricsHelper<T>.ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                    groupFalsePositiveRates[groupKey] = InterpretabilityMetricsHelper<T>.ComputeFalsePositiveRate(groupPredictions, groupActualLabels);
                    groupPrecisions[groupKey] = InterpretabilityMetricsHelper<T>.ComputePrecision(groupPredictions, groupActualLabels);
                }
            }

            result.GroupPositiveRates = groupPositiveRates;
            result.GroupSizes = groupSizes;

            // Add additional metrics if computed
            if (actualLabels is not null)
            {
                result.GroupTruePositiveRates = groupTruePositiveRates;
                result.GroupFalsePositiveRates = groupFalsePositiveRates;
                result.GroupPrecisions = groupPrecisions;

                // Compute Equal Opportunity Difference (max TPR - min TPR)
                var orderedTprs = groupTruePositiveRates.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
                if (orderedTprs.Count >= 2)
                {
                    T minTpr = orderedTprs.First();
                    T maxTpr = orderedTprs.Last();
                    result.EqualOpportunityDifference = _numOps.Subtract(maxTpr, minTpr);
                }
            }

            // Compute disparate impact ratio
            var orderedRates = groupPositiveRates.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
            T minRate = orderedRates.First();
            T maxRate = orderedRates.Last();
            result.StatisticalParityDifference = _numOps.Subtract(maxRate, minRate);

            // Avoid division by zero
            if (_numOps.Equals(maxRate, _numOps.Zero))
            {
                result.DisparateImpactRatio = _numOps.One;
                result.StatisticalParityDifference = _numOps.Zero;
                result.Message = "All groups have zero positive predictions. Disparate impact ratio set to 1.0.";
                result.HasBias = false;
                return result;
            }
            else
            {
                result.DisparateImpactRatio = _numOps.Divide(minRate, maxRate);
            }

            // Check for bias using configured threshold
            double disparateImpactValue = Convert.ToDouble(result.DisparateImpactRatio);
            result.HasBias = disparateImpactValue < _threshold;

            // Also check for Equal Opportunity bias if actualLabels provided
            if (actualLabels is not null)
            {
                double eod = Convert.ToDouble(result.EqualOpportunityDifference);
                // If EOD is significant (> 0.2), also flag as bias
                if (eod > 0.2)
                {
                    result.HasBias = true;
                }
            }

            if (result.HasBias)
            {
                result.Message = $"Bias detected: Disparate impact ratio = {disparateImpactValue:F3} (below {_threshold} threshold)";
            }
            else
            {
                result.Message = $"No significant bias detected: Disparate impact ratio = {disparateImpactValue:F3}";
            }

            return result;
        }
    }
}
