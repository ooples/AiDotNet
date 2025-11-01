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
        private readonly double _threshold;

        /// <summary>
        /// Initializes a new instance of the EqualOpportunityBiasDetector class.
        /// </summary>
        /// <param name="threshold">The threshold for detecting bias (default is 0.1, representing 10% difference threshold).</param>
        public EqualOpportunityBiasDetector(double threshold = 0.1) : base(isLowerBiasBetter: true)
        {
            if (threshold <= 0 || threshold > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be between 0 and 1.");
            }
            _threshold = threshold;
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

            // Check for bias using configured threshold
            // Note: eoValue is always non-negative since it's max - min, so Math.Abs is unnecessary
            double eoValue = Convert.ToDouble(result.EqualOpportunityDifference);
            result.HasBias = eoValue > _threshold;

            if (result.HasBias)
            {
                result.Message = $"Bias detected: Equal opportunity difference = {eoValue:F3} (exceeds {_threshold} threshold)";
            }
            else
            {
                result.Message = $"No significant bias detected: Equal opportunity difference = {eoValue:F3}";
            }

            return result;
        }
    }
}
