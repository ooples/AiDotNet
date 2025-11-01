using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Detects bias using Equal Opportunity metric (True Positive Rate difference).
    /// Requires actual labels to compute TPR for each group.
    /// A TPR difference greater than 0.1 (10%) indicates potential bias.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class EqualOpportunityBiasDetector<T> : BiasDetectorBase<T>
    {
        /// <summary>
        /// Initializes a new instance of the EqualOpportunityBiasDetector class.
        /// </summary>
        public EqualOpportunityBiasDetector() : base(isLowerBiasBetter: true)
        {
            // For equal opportunity difference, lower is better (0 means perfect equality)
        }

        /// <summary>
        /// Implements bias detection using Equal Opportunity (TPR difference).
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
            var groupTPRs = new Dictionary<string, T>();

            foreach (var group in groups)
            {
                var groupIndices = InterpretabilityMetricsHelper<T>.GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = InterpretabilityMetricsHelper<T>.GetSubset(predictions, groupIndices);
                var positiveRate = InterpretabilityMetricsHelper<T>.ComputePositiveRate(groupPredictions);
                string groupKey = group?.ToString() ?? "unknown";

                groupPositiveRates[groupKey] = positiveRate;
                groupSizes[groupKey] = groupIndices.Count;

                // Compute TPR if actual labels provided
                if (actualLabels != null)
                {
                    var groupActualLabels = InterpretabilityMetricsHelper<T>.GetSubset(actualLabels, groupIndices);
                    var tpr = InterpretabilityMetricsHelper<T>.ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                    groupTPRs[groupKey] = tpr;
                }
            }

            result.GroupPositiveRates = groupPositiveRates;
            result.GroupSizes = groupSizes;
            result.GroupTruePositiveRates = groupTPRs;

            // Check if actual labels were provided
            if (actualLabels == null || groupTPRs.Count == 0)
            {
                result.HasBias = false;
                result.Message = "Cannot compute equal opportunity without actual labels. Provide actualLabels parameter.";
                return result;
            }

            // Compute equal opportunity difference (max TPR - min TPR)
            var orderedTPRs = groupTPRs.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
            T minTPR = orderedTPRs.First();
            T maxTPR = orderedTPRs.Last();

            result.EqualOpportunityDifference = _numOps.Subtract(maxTPR, minTPR);

            // Check for bias using 10% threshold
            double eoValue = Convert.ToDouble(result.EqualOpportunityDifference);
            result.HasBias = Math.Abs(eoValue) > 0.1;

            if (result.HasBias)
            {
                result.Message = $"Bias detected: Equal opportunity difference = {eoValue:F3} (exceeds 0.1 threshold)";
            }
            else
            {
                result.Message = $"No significant bias detected: Equal opportunity difference = {eoValue:F3}";
            }

            return result;
        }
    }
}
