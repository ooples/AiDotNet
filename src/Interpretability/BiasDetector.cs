using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Provides methods for detecting bias in model predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class BiasDetector<T>
    {
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the BiasDetector class.
        /// </summary>
        public BiasDetector()
        {
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Detects bias in predictions by comparing outcomes across sensitive groups.
        /// </summary>
        /// <param name="predictions">The model predictions (binary: 0 or 1 for classification).</param>
        /// <param name="sensitiveFeature">The sensitive feature values (e.g., 0 for group A, 1 for group B).</param>
        /// <param name="actualLabels">The actual labels (optional, required for some metrics).</param>
        /// <returns>A BiasDetectionResult containing identified biases.</returns>
        /// <exception cref="ArgumentNullException">Thrown when predictions or sensitiveFeature is null.</exception>
        /// <exception cref="ArgumentException">Thrown when vectors have mismatched lengths.</exception>
        public BiasDetectionResult<T> DetectBias(Vector<T> predictions, Vector<T> sensitiveFeature, Vector<T>? actualLabels = null)
        {
            if (predictions == null)
            {
                throw new ArgumentNullException(nameof(predictions));
            }

            if (sensitiveFeature == null)
            {
                throw new ArgumentNullException(nameof(sensitiveFeature));
            }

            if (predictions.Length != sensitiveFeature.Length)
            {
                throw new ArgumentException($"Predictions length ({predictions.Length}) must match sensitiveFeature length ({sensitiveFeature.Length}).", nameof(predictions));
            }

            if (actualLabels != null && predictions.Length != actualLabels.Length)
            {
                throw new ArgumentException($"Predictions length ({predictions.Length}) must match actualLabels length ({actualLabels.Length}).", nameof(actualLabels));
            }

            var result = new BiasDetectionResult<T>();

            // Identify unique groups in the sensitive feature
            var groups = GetUniqueGroups(sensitiveFeature);

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
                var groupIndices = GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = GetSubset(predictions, groupIndices);
                var positiveRate = ComputePositiveRate(groupPredictions);
                string groupKey = _numOps.ToDouble(group).ToString();

                groupPositiveRates[groupKey] = positiveRate;
                groupSizes[groupKey] = groupIndices.Count;
            }

            result.GroupPositiveRates = groupPositiveRates;
            result.GroupSizes = groupSizes;

            // Compute disparate impact
            var orderedRates = groupPositiveRates.Values.OrderBy(r => _numOps.ToDouble(r)).ToList();
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
            double disparateImpactValue = _numOps.ToDouble(result.DisparateImpactRatio);
            double statisticalParityValue = _numOps.ToDouble(result.StatisticalParityDifference);

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
                var groupIndices = GetGroupIndices(sensitiveFeature, group);
                var groupPredictions = GetSubset(predictions, groupIndices);
                var groupActualLabels = GetSubset(actualLabels, groupIndices);
                string groupKey = _numOps.ToDouble(group).ToString();

                // Compute True Positive Rate (TPR)
                var tpr = ComputeTruePositiveRate(groupPredictions, groupActualLabels);
                groupTPRs[groupKey] = tpr;

                // Compute False Positive Rate (FPR)
                var fpr = ComputeFalsePositiveRate(groupPredictions, groupActualLabels);
                groupFPRs[groupKey] = fpr;

                // Compute Precision
                var precision = ComputePrecision(groupPredictions, groupActualLabels);
                groupPrecisions[groupKey] = precision;
            }

            result.GroupTruePositiveRates = groupTPRs;
            result.GroupFalsePositiveRates = groupFPRs;
            result.GroupPrecisions = groupPrecisions;

            // Compute equal opportunity difference (max TPR difference)
            if (groupTPRs.Count >= 2)
            {
                var orderedTPRs = groupTPRs.Values.OrderBy(r => _numOps.ToDouble(r)).ToList();
                T minTPR = orderedTPRs.First();
                T maxTPR = orderedTPRs.Last();
                result.EqualOpportunityDifference = _numOps.Subtract(maxTPR, minTPR);

                double eoValue = _numOps.ToDouble(result.EqualOpportunityDifference);
                if (Math.Abs(eoValue) > 0.1)
                {
                    result.HasBias = true;
                    result.Message += $" Equal opportunity difference = {eoValue:F3}.";
                }
            }
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
        /// Computes the positive prediction rate (proportion of positive predictions).
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
                // Assume binary predictions: 1 is positive, 0 is negative
                if (_numOps.ToDouble(predictions[i]) > 0.5)
                {
                    positiveCount = _numOps.Add(positiveCount, _numOps.One);
                }
            }

            return _numOps.Divide(positiveCount, _numOps.FromDouble(predictions.Length));
        }

        /// <summary>
        /// Computes the True Positive Rate (Sensitivity / Recall).
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
        /// Computes the Precision (Positive Predictive Value).
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
            Message = string.Empty;
            GroupPositiveRates = new Dictionary<string, T>();
            GroupSizes = new Dictionary<string, int>();
            GroupTruePositiveRates = new Dictionary<string, T>();
            GroupFalsePositiveRates = new Dictionary<string, T>();
            GroupPrecisions = new Dictionary<string, T>();
        }
    }
}
