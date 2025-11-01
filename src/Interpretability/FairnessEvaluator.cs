using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Provides comprehensive fairness evaluation including all major fairness metrics.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class FairnessEvaluator<T> : FairnessEvaluatorBase<T>
    {
        /// <summary>
        /// Initializes a new instance of the FairnessEvaluator class.
        /// </summary>
        public FairnessEvaluator() : base(isHigherFairnessBetter: false)
        {
        }

        /// <summary>
        /// Implements comprehensive fairness evaluation logic.
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
            var groupStats = ComputeGroupStatistics(predictions, sensitiveFeature, actualLabels, groups);

            // Calculate fairness metrics
            T demographicParity = ComputeDemographicParity(groupStats);
            T equalOpportunity = actualLabels != null ? ComputeEqualOpportunity(groupStats) : _numOps.Zero;
            T equalizedOdds = actualLabels != null ? ComputeEqualizedOdds(groupStats) : _numOps.Zero;
            T predictiveParity = actualLabels != null ? ComputePredictiveParity(groupStats) : _numOps.Zero;
            T disparateImpact = ComputeDisparateImpact(groupStats);
            T statisticalParityDifference = ComputeStatisticalParityDifference(groupStats);

            var fairnessMetrics = new FairnessMetrics<T>(
                demographicParity: demographicParity,
                equalOpportunity: equalOpportunity,
                equalizedOdds: equalizedOdds,
                predictiveParity: predictiveParity,
                disparateImpact: disparateImpact,
                statisticalParityDifference: statisticalParityDifference)
            {
                SensitiveFeatureIndex = sensitiveFeatureIndex
            };

            // Add per-group metrics to additional metrics
            foreach (var group in groups)
            {
                string groupKey = group?.ToString() ?? "unknown";
                var stats = groupStats[groupKey];
                fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_PositiveRate"] = stats.PositiveRate;
                fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_Size"] = _numOps.FromDouble(stats.Size);

                if (actualLabels != null)
                {
                    fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_TPR"] = stats.TruePositiveRate;
                    fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_FPR"] = stats.FalsePositiveRate;
                    fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_Precision"] = stats.Precision;
                }
            }

            return fairnessMetrics;
        }

        /// <summary>
        /// Computes statistics for each group in the sensitive feature.
        /// </summary>
        private Dictionary<string, GroupStatistics<T>> ComputeGroupStatistics(
            Vector<T> predictions,
            Vector<T> sensitiveFeature,
            Vector<T>? actualLabels,
            List<T> groups)
        {
            var groupStats = new Dictionary<string, GroupStatistics<T>>();

            foreach (var group in groups)
            {
                var groupIndices = InterpretabilityMetricsHelper<T>.GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = InterpretabilityMetricsHelper<T>.GetSubset(predictions, groupIndices);
                string groupKey = group?.ToString() ?? "unknown";

                var stats = new GroupStatistics<T>
                {
                    GroupValue = group,
                    Size = groupIndices.Count,
                    PositiveRate = InterpretabilityMetricsHelper<T>.ComputePositiveRate(groupPredictions)
                };

                if (actualLabels != null)
                {
                    var groupActualLabels = InterpretabilityMetricsHelper<T>.GetSubset(actualLabels, groupIndices);
                    stats.TruePositiveRate = InterpretabilityMetricsHelper<T>.ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                    stats.FalsePositiveRate = InterpretabilityMetricsHelper<T>.ComputeFalsePositiveRate(groupPredictions, groupActualLabels);
                    stats.Precision = InterpretabilityMetricsHelper<T>.ComputePrecision(groupPredictions, groupActualLabels);
                }

                groupStats[groupKey] = stats;
            }

            return groupStats;
        }

        /// <summary>
        /// Computes Demographic Parity (max difference in positive rates).
        /// </summary>
        private T ComputeDemographicParity(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var rates = groupStats.Values.Select(s => s.PositiveRate).ToList();
            var maxRate = rates.Max(r => Convert.ToDouble(r));
            var minRate = rates.Min(r => Convert.ToDouble(r));
            return _numOps.FromDouble(maxRate - minRate);
        }

        /// <summary>
        /// Computes Equal Opportunity (max difference in TPR).
        /// </summary>
        private T ComputeEqualOpportunity(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var tprs = groupStats.Values.Select(s => s.TruePositiveRate).ToList();
            var maxTPR = tprs.Max(r => Convert.ToDouble(r));
            var minTPR = tprs.Min(r => Convert.ToDouble(r));
            return _numOps.FromDouble(maxTPR - minTPR);
        }

        /// <summary>
        /// Computes Equalized Odds (max of TPR and FPR differences).
        /// </summary>
        private T ComputeEqualizedOdds(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var tprs = groupStats.Values.Select(s => s.TruePositiveRate).ToList();
            var fprs = groupStats.Values.Select(s => s.FalsePositiveRate).ToList();

            var maxTPR = tprs.Max(r => Convert.ToDouble(r));
            var minTPR = tprs.Min(r => Convert.ToDouble(r));
            var tprDiff = maxTPR - minTPR;

            var maxFPR = fprs.Max(r => Convert.ToDouble(r));
            var minFPR = fprs.Min(r => Convert.ToDouble(r));
            var fprDiff = maxFPR - minFPR;

            return _numOps.FromDouble(Math.Max(tprDiff, fprDiff));
        }

        /// <summary>
        /// Computes Predictive Parity (max difference in precision).
        /// </summary>
        private T ComputePredictiveParity(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var precisions = groupStats.Values.Select(s => s.Precision).ToList();
            var maxPrecision = precisions.Max(r => Convert.ToDouble(r));
            var minPrecision = precisions.Min(r => Convert.ToDouble(r));
            return _numOps.FromDouble(maxPrecision - minPrecision);
        }

        /// <summary>
        /// Computes Disparate Impact (ratio of min to max positive rate).
        /// </summary>
        private T ComputeDisparateImpact(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var rates = groupStats.Values.Select(s => s.PositiveRate).ToList();
            var maxRate = rates.Max(r => Convert.ToDouble(r));
            var minRate = rates.Min(r => Convert.ToDouble(r));

            if (maxRate == 0)
            {
                return _numOps.One;
            }

            return _numOps.FromDouble(minRate / maxRate);
        }

        /// <summary>
        /// Computes Statistical Parity Difference (max - min positive rate).
        /// </summary>
        private T ComputeStatisticalParityDifference(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var rates = groupStats.Values.Select(s => s.PositiveRate).ToList();
            var maxRate = rates.Max(r => Convert.ToDouble(r));
            var minRate = rates.Min(r => Convert.ToDouble(r));
            return _numOps.FromDouble(maxRate - minRate);
        }
    }
}
