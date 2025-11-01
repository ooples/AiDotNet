using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Detects bias using Demographic Parity (Statistical Parity Difference).
    /// Measures the difference in positive prediction rates between groups.
    /// A difference greater than 0.1 (10%) indicates potential bias.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class DemographicParityBiasDetector<T> : BiasDetectorBase<T>
    {
        /// <summary>
        /// Initializes a new instance of the DemographicParityBiasDetector class.
        /// </summary>
        public DemographicParityBiasDetector() : base(isLowerBiasBetter: true)
        {
            // For statistical parity difference, lower is better (0 means perfect parity)
        }

        /// <summary>
        /// Implements bias detection using Statistical Parity Difference.
        /// </summary>
        protected override BiasDetectionResult<T> GetBiasDetectionResult(
            Vector<T> predictions,
            Vector<T> sensitiveFeature,
            Vector<T>? actualLabels)
        {
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

            // Compute statistical parity difference (max rate - min rate)
            var orderedRates = groupPositiveRates.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
            T minRate = orderedRates.First();
            T maxRate = orderedRates.Last();

            result.StatisticalParityDifference = _numOps.Subtract(maxRate, minRate);

            // Check for bias using 10% threshold
            double statisticalParityValue = Convert.ToDouble(result.StatisticalParityDifference);
            result.HasBias = Math.Abs(statisticalParityValue) > 0.1;

            if (result.HasBias)
            {
                result.Message = $"Bias detected: Statistical parity difference = {statisticalParityValue:F3} (exceeds 0.1 threshold)";
            }
            else
            {
                result.Message = $"No significant bias detected: Statistical parity difference = {statisticalParityValue:F3}";
            }

            return result;
        }
    }
}
