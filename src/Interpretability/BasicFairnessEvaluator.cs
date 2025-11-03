using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Basic fairness evaluator that computes only fundamental fairness metrics.
    /// Includes demographic parity (statistical parity difference) and disparate impact.
    /// Does not require actual labels.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class BasicFairnessEvaluator<T> : FairnessEvaluatorBase<T>
    {
        /// <summary>
        /// Initializes a new instance of the BasicFairnessEvaluator class.
        /// </summary>
        public BasicFairnessEvaluator() : base(isHigherFairnessBetter: false)
        {
            // Lower fairness scores indicate better fairness (closer to 0 means less disparity)
        }

        /// <summary>
        /// Computes basic fairness metrics (demographic parity and disparate impact).
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

            // Compute positive rates for each group
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

            // Compute demographic parity (statistical parity difference)
            var orderedRates = groupPositiveRates.Values.OrderBy(r => Convert.ToDouble(r)).ToList();
            T minRate = orderedRates.First();
            T maxRate = orderedRates.Last();

            T demographicParity = _numOps.Subtract(maxRate, minRate);

            // Compute disparate impact
            T disparateImpact;
            if (_numOps.Equals(maxRate, _numOps.Zero))
            {
                disparateImpact = _numOps.One;
            }
            else
            {
                disparateImpact = _numOps.Divide(minRate, maxRate);
            }

            var fairnessMetrics = new FairnessMetrics<T>(
                demographicParity: demographicParity,
                equalOpportunity: _numOps.Zero,  // Not computed in basic evaluator
                equalizedOdds: _numOps.Zero,     // Not computed in basic evaluator
                predictiveParity: _numOps.Zero,  // Not computed in basic evaluator
                disparateImpact: disparateImpact,
                statisticalParityDifference: demographicParity)
            {
                SensitiveFeatureIndex = sensitiveFeatureIndex
            };

            // Add per-group basic metrics
            foreach (var group in groups)
            {
                string groupKey = group?.ToString() ?? "unknown";
                fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_PositiveRate"] = groupPositiveRates[groupKey];
                fairnessMetrics.AdditionalMetrics[$"Group_{groupKey}_Size"] = _numOps.FromDouble(groupSizes[groupKey]);
            }

            return fairnessMetrics;
        }
    }
}
