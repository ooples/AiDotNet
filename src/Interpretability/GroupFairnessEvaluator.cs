using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Group-level fairness evaluator that focuses on equalized performance across groups.
    /// Computes equal opportunity and equalized odds when actual labels are available.
    /// Focuses on ensuring similar error rates across demographic groups.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class GroupFairnessEvaluator<T> : FairnessEvaluatorBase<T>
    {
        /// <summary>
        /// Initializes a new instance of the GroupFairnessEvaluator class.
        /// </summary>
        public GroupFairnessEvaluator() : base(isHigherFairnessBetter: false)
        {
            // Lower fairness scores indicate better fairness (closer to 0 means less disparity)
        }

        /// <summary>
        /// Computes group-level fairness metrics focusing on performance equity.
        /// </summary>
        protected override FairnessMetrics<T> GetFairnessMetrics(
            IFullModel<T, Matrix<T>, Vector<T>> model,
            Matrix<T> inputs,
            int sensitiveFeatureIndex,
            Vector<T>? actualLabels)
        {
            // Get predictions from the model
            Vector<T> predictions = model.Predict(inputs);

            // Extract sensitive feature column
            Vector<T> sensitiveFeature = inputs.GetColumn(sensitiveFeatureIndex);

            // Identify unique groups
            var groups = InterpretabilityMetricsHelper<T>.GetUniqueGroups(sensitiveFeature);

            if (groups.Count < 2)
            {
                // Return zero metrics if insufficient groups
                return new FairnessMetrics<T>(
                    demographicParity: _numOps.Zero,
                    equalOpportunity: _numOps.Zero,
                    equalizedOdds: _numOps.Zero,
                    predictiveParity: _numOps.Zero,
                    disparateImpact: _numOps.One,
                    statisticalParityDifference: _numOps.Zero)
                {
                    SensitiveFeatureIndex = sensitiveFeatureIndex
                };
            }

            // Compute group statistics
            var groupTPRs = new Dictionary<string, T>();
            var groupFPRs = new Dictionary<string, T>();
            var groupPrecisions = new Dictionary<string, T>();
            var groupSizes = new Dictionary<string, int>();

            foreach (var group in groups)
            {
                var groupIndices = InterpretabilityMetricsHelper<T>.GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = InterpretabilityMetricsHelper<T>.GetSubset(predictions, groupIndices);
                string groupKey = group?.ToString() ?? "unknown";

                groupSizes[groupKey] = groupIndices.Count;

                if (actualLabels != null)
                {
                    var groupActualLabels = InterpretabilityMetricsHelper<T>.GetSubset(actualLabels, groupIndices);

                    // Compute TPR (True Positive Rate)
                    var tpr = InterpretabilityMetricsHelper<T>.ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                    groupTPRs[groupKey] = tpr;

                    // Compute FPR (False Positive Rate)
                    var fpr = InterpretabilityMetricsHelper<T>.ComputeFalsePositiveRate(groupPredictions, groupActualLabels);
                    groupFPRs[groupKey] = fpr;

                    // Compute Precision
                    var precision = InterpretabilityMetricsHelper<T>.ComputePrecision(groupPredictions, groupActualLabels);
                    groupPrecisions[groupKey] = precision;
                }
            }

            // Compute group fairness metrics
            T equalOpportunity = _numOps.Zero;
            T equalizedOdds = _numOps.Zero;
            T predictiveParity = _numOps.Zero;

            if (actualLabels != null && groupTPRs.Count >= 2)
            {
                // Equal Opportunity: max TPR difference
                var orderedTPRs = groupTPRs.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
                T minTPR = orderedTPRs.First();
                T maxTPR = orderedTPRs.Last();
                equalOpportunity = _numOps.Subtract(maxTPR, minTPR);

                // Equalized Odds: max of TPR and FPR differences
                var orderedFPRs = groupFPRs.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
                T minFPR = orderedFPRs.First();
                T maxFPR = orderedFPRs.Last();

                double tprDiff = Convert.ToDouble(_numOps.Subtract(maxTPR, minTPR));
                double fprDiff = Convert.ToDouble(_numOps.Subtract(maxFPR, minFPR));
                equalizedOdds = _numOps.FromDouble(Math.Max(tprDiff, fprDiff));

                // Predictive Parity: max precision difference
                var orderedPrecisions = groupPrecisions.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
                T minPrecision = orderedPrecisions.First();
                T maxPrecision = orderedPrecisions.Last();
                predictiveParity = _numOps.Subtract(maxPrecision, minPrecision);
            }

            var fairnessMetrics = new FairnessMetrics<T>(
                demographicParity: _numOps.Zero,  // Not primary focus in group evaluator
                equalOpportunity: equalOpportunity,
                equalizedOdds: equalizedOdds,
                predictiveParity: predictiveParity,
                disparateImpact: _numOps.One,     // Not primary focus in group evaluator
                statisticalParityDifference: _numOps.Zero)
            {
                SensitiveFeatureIndex = sensitiveFeatureIndex
            };

            // Add per-group performance metrics
            foreach (var group in groups)
            {
                string groupKey = group?.ToString() ?? "unknown";
                fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_Size"] = _numOps.FromDouble(groupSizes[groupKey]);

                if (actualLabels != null)
                {
                    fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_TPR"] = groupTPRs[groupKey];
                    fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_FPR"] = groupFPRs[groupKey];
                    fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_Precision"] = groupPrecisions[groupKey];
                }
            }

            return fairnessMetrics;
        }
    }
}
