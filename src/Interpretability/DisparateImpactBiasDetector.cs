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
                throw new ArgumentException("Predictions and sensitive feature vectors must have the same length.", nameof(predictions));
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

            // Compute disparate impact ratio
            var orderedRates = groupPositiveRates.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
            T minRate = orderedRates.First();
            T maxRate = orderedRates.Last();

            // Avoid division by zero
            if (_numOps.Equals(maxRate, _numOps.Zero))
            {
                result.DisparateImpactRatio = _numOps.One;
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
