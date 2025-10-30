using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Provides methods for evaluating fairness metrics of machine learning models.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class FairnessEvaluator<T>
    {
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the FairnessEvaluator class.
        /// </summary>
        public FairnessEvaluator()
        {
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Evaluates fairness metrics for a model on the given dataset.
        /// </summary>
        /// <param name="model">The model to evaluate.</param>
        /// <param name="inputs">The input data.</param>
        /// <param name="sensitiveFeatureIndex">The index of the sensitive feature in the input data.</param>
        /// <param name="actualLabels">The actual labels (optional, required for some metrics).</param>
        /// <returns>A FairnessMetrics object containing computed metrics.</returns>
        /// <exception cref="ArgumentNullException">Thrown when model or inputs is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when sensitiveFeatureIndex is invalid.</exception>
        public async Task<FairnessMetrics<T>> EvaluateFairnessAsync(
            IFullModel<T, Matrix<T>, Vector<T>> model,
            Matrix<T> inputs,
            int sensitiveFeatureIndex,
            Vector<T>? actualLabels = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (inputs == null)
            {
                throw new ArgumentNullException(nameof(inputs));
            }

            if (sensitiveFeatureIndex < 0 || sensitiveFeatureIndex >= inputs.Columns)
            {
                throw new ArgumentOutOfRangeException(nameof(sensitiveFeatureIndex),
                    $"Sensitive feature index must be between 0 and {inputs.Columns - 1}.");
            }

            if (actualLabels != null && actualLabels.Length != inputs.Rows)
            {
                throw new ArgumentException($"Actual labels length ({actualLabels.Length}) must match input rows ({inputs.Rows}).", nameof(actualLabels));
            }

            // Get predictions from the model
            Vector<T> predictions = model.Predict(inputs);

            // Extract sensitive feature column
            Vector<T> sensitiveFeature = inputs.GetColumn(sensitiveFeatureIndex);

            // Identify unique groups
            var groups = GetUniqueGroups(sensitiveFeature);

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
                var stats = groupStats[group];
                fairnessMetrics.AdditionalMetrics[$"Group_{group}_PositiveRate"] = stats.PositiveRate;
                fairnessMetrics.AdditionalMetrics[$"Group_{group}_Size"] = _numOps.FromDouble(stats.Size);

                if (actualLabels != null)
                {
                    fairnessMetrics.AdditionalMetrics[$"Group_{group}_TPR"] = stats.TruePositiveRate;
                    fairnessMetrics.AdditionalMetrics[$"Group_{group}_FPR"] = stats.FalsePositiveRate;
                    fairnessMetrics.AdditionalMetrics[$"Group_{group}_Precision"] = stats.Precision;
                }
            }

            await Task.CompletedTask; // Satisfy async signature
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
                var groupIndices = GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = GetSubset(predictions, groupIndices);
                string groupKey = _numOps.ToDouble(group).ToString();

                var stats = new GroupStatistics<T>
                {
                    GroupValue = group,
                    Size = groupIndices.Count,
                    PositiveRate = ComputePositiveRate(groupPredictions)
                };

                if (actualLabels != null)
                {
                    var groupActualLabels = GetSubset(actualLabels, groupIndices);
                    stats.TruePositiveRate = ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                    stats.FalsePositiveRate = ComputeFalsePositiveRate(groupPredictions, groupActualLabels);
                    stats.Precision = ComputePrecision(groupPredictions, groupActualLabels);
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
            var maxRate = rates.Max(r => _numOps.ToDouble(r));
            var minRate = rates.Min(r => _numOps.ToDouble(r));
            return _numOps.FromDouble(maxRate - minRate);
        }

        /// <summary>
        /// Computes Equal Opportunity (max difference in TPR).
        /// </summary>
        private T ComputeEqualOpportunity(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var tprs = groupStats.Values.Select(s => s.TruePositiveRate).ToList();
            var maxTPR = tprs.Max(r => _numOps.ToDouble(r));
            var minTPR = tprs.Min(r => _numOps.ToDouble(r));
            return _numOps.FromDouble(maxTPR - minTPR);
        }

        /// <summary>
        /// Computes Equalized Odds (max of TPR and FPR differences).
        /// </summary>
        private T ComputeEqualizedOdds(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var tprs = groupStats.Values.Select(s => s.TruePositiveRate).ToList();
            var fprs = groupStats.Values.Select(s => s.FalsePositiveRate).ToList();

            var maxTPR = tprs.Max(r => _numOps.ToDouble(r));
            var minTPR = tprs.Min(r => _numOps.ToDouble(r));
            var tprDiff = maxTPR - minTPR;

            var maxFPR = fprs.Max(r => _numOps.ToDouble(r));
            var minFPR = fprs.Min(r => _numOps.ToDouble(r));
            var fprDiff = maxFPR - minFPR;

            return _numOps.FromDouble(Math.Max(tprDiff, fprDiff));
        }

        /// <summary>
        /// Computes Predictive Parity (max difference in precision).
        /// </summary>
        private T ComputePredictiveParity(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var precisions = groupStats.Values.Select(s => s.Precision).ToList();
            var maxPrecision = precisions.Max(r => _numOps.ToDouble(r));
            var minPrecision = precisions.Min(r => _numOps.ToDouble(r));
            return _numOps.FromDouble(maxPrecision - minPrecision);
        }

        /// <summary>
        /// Computes Disparate Impact (ratio of min to max positive rate).
        /// </summary>
        private T ComputeDisparateImpact(Dictionary<string, GroupStatistics<T>> groupStats)
        {
            var rates = groupStats.Values.Select(s => s.PositiveRate).ToList();
            var maxRate = rates.Max(r => _numOps.ToDouble(r));
            var minRate = rates.Min(r => _numOps.ToDouble(r));

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
            var maxRate = rates.Max(r => _numOps.ToDouble(r));
            var minRate = rates.Min(r => _numOps.ToDouble(r));
            return _numOps.FromDouble(maxRate - minRate);
        }

        /// <summary>
        /// Gets unique groups from the sensitive feature vector.
        /// </summary>
        private List<T> GetUniqueGroups(Vector<T> sensitiveFeature)
        {
            var groups = new HashSet<T>();
            for (int i = 0; i < sensitiveFeature.Length; i++)
            {
                groups.Add(sensitiveFeature[i]);
            }
            return groups.ToList();
        }

        /// <summary>
        /// Gets indices where the sensitive feature equals the specified group value.
        /// </summary>
        private List<int> GetGroupIndices(Vector<T> sensitiveFeature, T groupValue)
        {
            var indices = new List<int>();
            for (int i = 0; i < sensitiveFeature.Length; i++)
            {
                if (_numOps.Equals(sensitiveFeature[i], groupValue))
                {
                    indices.Add(i);
                }
            }
            return indices;
        }

        /// <summary>
        /// Gets a subset of a vector based on the specified indices.
        /// </summary>
        private Vector<T> GetSubset(Vector<T> vector, List<int> indices)
        {
            var subset = new Vector<T>(indices.Count);
            for (int i = 0; i < indices.Count; i++)
            {
                subset[i] = vector[indices[i]];
            }
            return subset;
        }

        /// <summary>
        /// Computes the positive prediction rate.
        /// </summary>
        private T ComputePositiveRate(Vector<T> predictions)
        {
            if (predictions.Length == 0)
            {
                return _numOps.Zero;
            }

            T positiveCount = _numOps.Zero;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (_numOps.ToDouble(predictions[i]) > 0.5)
                {
                    positiveCount = _numOps.Add(positiveCount, _numOps.One);
                }
            }

            return _numOps.Divide(positiveCount, _numOps.FromDouble(predictions.Length));
        }

        /// <summary>
        /// Computes the True Positive Rate.
        /// </summary>
        private T ComputeTruePositiveRate(Vector<T> predictions, Vector<T> actualLabels)
        {
            T truePositives = _numOps.Zero;
            T actualPositives = _numOps.Zero;

            for (int i = 0; i < predictions.Length; i++)
            {
                bool predicted = _numOps.ToDouble(predictions[i]) > 0.5;
                bool actual = _numOps.ToDouble(actualLabels[i]) > 0.5;

                if (actual)
                {
                    actualPositives = _numOps.Add(actualPositives, _numOps.One);
                    if (predicted)
                    {
                        truePositives = _numOps.Add(truePositives, _numOps.One);
                    }
                }
            }

            if (_numOps.Equals(actualPositives, _numOps.Zero))
            {
                return _numOps.Zero;
            }

            return _numOps.Divide(truePositives, actualPositives);
        }

        /// <summary>
        /// Computes the False Positive Rate.
        /// </summary>
        private T ComputeFalsePositiveRate(Vector<T> predictions, Vector<T> actualLabels)
        {
            T falsePositives = _numOps.Zero;
            T actualNegatives = _numOps.Zero;

            for (int i = 0; i < predictions.Length; i++)
            {
                bool predicted = _numOps.ToDouble(predictions[i]) > 0.5;
                bool actual = _numOps.ToDouble(actualLabels[i]) > 0.5;

                if (!actual)
                {
                    actualNegatives = _numOps.Add(actualNegatives, _numOps.One);
                    if (predicted)
                    {
                        falsePositives = _numOps.Add(falsePositives, _numOps.One);
                    }
                }
            }

            if (_numOps.Equals(actualNegatives, _numOps.Zero))
            {
                return _numOps.Zero;
            }

            return _numOps.Divide(falsePositives, actualNegatives);
        }

        /// <summary>
        /// Computes the Precision.
        /// </summary>
        private T ComputePrecision(Vector<T> predictions, Vector<T> actualLabels)
        {
            T truePositives = _numOps.Zero;
            T predictedPositives = _numOps.Zero;

            for (int i = 0; i < predictions.Length; i++)
            {
                bool predicted = _numOps.ToDouble(predictions[i]) > 0.5;
                bool actual = _numOps.ToDouble(actualLabels[i]) > 0.5;

                if (predicted)
                {
                    predictedPositives = _numOps.Add(predictedPositives, _numOps.One);
                    if (actual)
                    {
                        truePositives = _numOps.Add(truePositives, _numOps.One);
                    }
                }
            }

            if (_numOps.Equals(predictedPositives, _numOps.Zero))
            {
                return _numOps.Zero;
            }

            return _numOps.Divide(truePositives, predictedPositives);
        }
    }

    /// <summary>
    /// Represents statistics for a specific group in fairness evaluation.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    internal class GroupStatistics<T>
    {
        /// <summary>
        /// Gets or sets the group value.
        /// </summary>
        public T GroupValue { get; set; }

        /// <summary>
        /// Gets or sets the size of the group.
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Gets or sets the positive prediction rate.
        /// </summary>
        public T PositiveRate { get; set; }

        /// <summary>
        /// Gets or sets the True Positive Rate.
        /// </summary>
        public T TruePositiveRate { get; set; }

        /// <summary>
        /// Gets or sets the False Positive Rate.
        /// </summary>
        public T FalsePositiveRate { get; set; }

        /// <summary>
        /// Gets or sets the Precision.
        /// </summary>
        public T Precision { get; set; }

        /// <summary>
        /// Initializes a new instance of the GroupStatistics class.
        /// </summary>
        public GroupStatistics()
        {
            GroupValue = default(T);
            Size = 0;
        }
    }
}
