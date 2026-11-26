using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Provides static utility methods for computing interpretability and fairness metrics.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a collection of reusable helper methods for fairness and bias analysis.
    ///
    /// These methods handle common tasks like:
    /// - Identifying unique groups in data (e.g., different age groups, genders)
    /// - Computing metrics like positive rates, true positive rates, etc.
    /// - Extracting subsets of data for specific groups
    ///
    /// By centralizing these methods here, we avoid code duplication and ensure consistent
    /// calculations across all bias detection and fairness evaluation tools.
    /// </para>
    /// </remarks>
    public static class InterpretabilityMetricsHelper<T>
    {
        private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

        /// <summary>
        /// Identifies all unique groups in the sensitive feature.
        /// </summary>
        /// <param name="sensitiveFeature">The sensitive feature vector (e.g., race, gender, age group).</param>
        /// <returns>A list of unique group values found in the sensitive feature.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method finds all the different categories in your sensitive feature.
        ///
        /// For example, if your sensitive feature is gender with values [1, 0, 1, 0, 1],
        /// this method would return a list containing [0, 1] (the two unique groups).
        ///
        /// This is the first step in fairness analysis - we need to know which groups exist
        /// before we can compare how they're treated.
        /// </para>
        /// </remarks>
        public static List<T> GetUniqueGroups(Vector<T> sensitiveFeature)
        {
            if (sensitiveFeature == null)
                throw new ArgumentNullException(nameof(sensitiveFeature));

            // Bulk set construction keeps first-occurrence order, as adding each element in turn did.
            return new HashSet<T>(sensitiveFeature).ToList();
        }

        /// <summary>
        /// Gets the indices of all samples belonging to a specific group.
        /// </summary>
        /// <param name="sensitiveFeature">The sensitive feature vector.</param>
        /// <param name="groupValue">The group value to search for.</param>
        /// <returns>A list of indices where the sensitive feature matches the group value.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method finds the positions of all members of a specific group.
        ///
        /// For example, if your sensitive feature is [1, 0, 1, 0, 1] and groupValue is 1,
        /// this method would return [0, 2, 4] (the positions where the value is 1).
        ///
        /// This allows us to isolate data for a specific group so we can analyze how
        /// the model treats that group separately.
        /// </para>
        /// </remarks>
        public static List<int> GetGroupIndices(Vector<T> sensitiveFeature, T groupValue)
        {
            if (sensitiveFeature == null)
                throw new ArgumentNullException(nameof(sensitiveFeature));

            var engine = AiDotNetEngine.Current;
            var values = sensitiveFeature.AsTensor();
            var differs = engine.TensorNotEquals(values, groupValue);
            var matches = engine.TensorSubtract(engine.TensorFullLike(values, _numOps.One), differs);
            return matches.ToVector().NonZeroIndices().ToList();
        }

        /// <summary>
        /// Extracts a subset of a vector based on specified indices.
        /// </summary>
        /// <param name="vector">The source vector.</param>
        /// <param name="indices">The indices to extract.</param>
        /// <returns>A new vector containing only the elements at the specified indices.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method creates a smaller vector containing only specific elements.
        ///
        /// For example, if you have a vector [10, 20, 30, 40, 50] and indices [0, 2, 4],
        /// this method would return a new vector [10, 30, 50].
        ///
        /// This is useful for extracting predictions or labels for a specific group after
        /// you've identified which indices belong to that group.
        /// </para>
        /// </remarks>
        public static Vector<T> GetSubset(Vector<T> vector, List<int> indices)
        {
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));
            if (indices == null)
                throw new ArgumentNullException(nameof(indices));

            if (indices.Count == 0)
                return new Vector<T>(0);

            var positions = new Tensor<int>(indices.ToArray(), new[] { indices.Count });
            return AiDotNetEngine.Current.TensorIndexSelect(vector.AsTensor(), positions, 0).ToVector();
        }

        /// <summary>
        /// Computes the positive prediction rate (proportion of positive predictions).
        /// </summary>
        /// <param name="predictions">The prediction vector (binary: 0 or 1).</param>
        /// <returns>The proportion of positive predictions (predictions equal to 1).</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method calculates what fraction of predictions are positive (1).
        ///
        /// For example, if predictions are [1, 0, 1, 1, 0], the positive rate would be 3/5 = 0.6
        /// (60% of predictions are positive).
        ///
        /// This is a key metric for fairness - if one group has a much higher positive rate than
        /// another, it might indicate bias in the model.
        /// </para>
        /// </remarks>
        public static T ComputePositiveRate(Vector<T> predictions)
        {
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));

            if (predictions.Length == 0)
                return _numOps.Zero;

            var engine = AiDotNetEngine.Current;
            var values = predictions.AsTensor();
            var positive = engine.TensorGreaterEqual(values, engine.TensorFullLike(values, _numOps.FromDouble(0.5)));
            return CountRatio(engine.TensorCountNonzero(positive), predictions.Length);
        }

        /// <summary>
        /// Computes the True Positive Rate (TPR) or Recall.
        /// </summary>
        /// <param name="predictions">The prediction vector (binary: 0 or 1).</param>
        /// <param name="actualLabels">The actual label vector (binary: 0 or 1).</param>
        /// <returns>The proportion of actual positives that were correctly predicted as positive.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method calculates how good the model is at identifying positive cases.
        ///
        /// TPR = (True Positives) / (True Positives + False Negatives)
        ///
        /// For example, if there are 10 actual positive cases and the model correctly identified 8 of them,
        /// the TPR would be 8/10 = 0.8 (80%).
        ///
        /// In fairness analysis, we want the TPR to be similar across all groups. If the model is better
        /// at identifying positive cases for one group than another, that's a form of bias.
        /// </para>
        /// </remarks>
        public static T ComputeTruePositiveRate(Vector<T> predictions, Vector<T> actualLabels)
        {
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));
            if (actualLabels == null)
                throw new ArgumentNullException(nameof(actualLabels));

            if (predictions.Length == 0 || actualLabels.Length == 0)
                return _numOps.Zero;

            var (actualPositive, _, predictedPositive) = ClassifyAtThreshold(predictions, actualLabels);
            var engine = AiDotNetEngine.Current;
            // TPR = true positives / actual positives.
            int numerator = engine.TensorCountNonzero(engine.TensorMultiply(actualPositive, predictedPositive));
            int denominator = engine.TensorCountNonzero(actualPositive);

            return CountRatio(numerator, denominator);
        }

        /// <summary>
        /// Computes the False Positive Rate (FPR).
        /// </summary>
        /// <param name="predictions">The prediction vector (binary: 0 or 1).</param>
        /// <param name="actualLabels">The actual label vector (binary: 0 or 1).</param>
        /// <returns>The proportion of actual negatives that were incorrectly predicted as positive.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method calculates how often the model incorrectly predicts positive.
        ///
        /// FPR = (False Positives) / (False Positives + True Negatives)
        ///
        /// For example, if there are 10 actual negative cases and the model incorrectly called 2 of them positive,
        /// the FPR would be 2/10 = 0.2 (20%).
        ///
        /// In fairness analysis, we want the FPR to be similar across all groups. If the model makes more
        /// false positive errors for one group than another, that's a form of bias.
        /// </para>
        /// </remarks>
        public static T ComputeFalsePositiveRate(Vector<T> predictions, Vector<T> actualLabels)
        {
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));
            if (actualLabels == null)
                throw new ArgumentNullException(nameof(actualLabels));

            if (predictions.Length == 0 || actualLabels.Length == 0)
                return _numOps.Zero;

            var (_, actualNegative, predictedPositive) = ClassifyAtThreshold(predictions, actualLabels);
            var engine = AiDotNetEngine.Current;
            // FPR = false positives / actual negatives.
            int numerator = engine.TensorCountNonzero(engine.TensorMultiply(actualNegative, predictedPositive));
            int denominator = engine.TensorCountNonzero(actualNegative);

            return CountRatio(numerator, denominator);
        }

        /// <summary>
        /// Computes the Precision (Positive Predictive Value).
        /// </summary>
        /// <param name="predictions">The prediction vector (binary: 0 or 1).</param>
        /// <param name="actualLabels">The actual label vector (binary: 0 or 1).</param>
        /// <returns>The proportion of positive predictions that were actually correct.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method calculates how accurate the positive predictions are.
        ///
        /// Precision = (True Positives) / (True Positives + False Positives)
        ///
        /// For example, if the model made 10 positive predictions and 8 of them were correct,
        /// the precision would be 8/10 = 0.8 (80%).
        ///
        /// In fairness analysis, we want precision to be similar across all groups. If positive
        /// predictions are more reliable for one group than another, that's a form of bias.
        /// </para>
        /// </remarks>
        public static T ComputePrecision(Vector<T> predictions, Vector<T> actualLabels)
        {
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));
            if (actualLabels == null)
                throw new ArgumentNullException(nameof(actualLabels));

            if (predictions.Length == 0 || actualLabels.Length == 0)
                return _numOps.Zero;

            var (actualPositive, _, predictedPositive) = ClassifyAtThreshold(predictions, actualLabels);
            var engine = AiDotNetEngine.Current;
            // Precision = true positives / predicted positives.
            int numerator = engine.TensorCountNonzero(engine.TensorMultiply(actualPositive, predictedPositive));
            int denominator = engine.TensorCountNonzero(predictedPositive);

            return CountRatio(numerator, denominator);
        }

        /// <summary>
        /// count / total as T (0 when total is 0). The counts stay in int: a sum of 0/1 masks in T stops at 2048
        /// for Half.
        /// </summary>
        private static T CountRatio(int count, int total)
        {
            return total == 0 ? _numOps.Zero : _numOps.FromDouble((double)count / total);
        }

        /// <summary>
        /// 0/1 masks at the 0.5 threshold: actual positive (label &gt;= 0.5), actual negative (label &lt; 0.5) and
        /// predicted positive (prediction &gt;= 0.5), over the first predictions.Length labels.
        /// </summary>
        private static (Tensor<T> ActualPositive, Tensor<T> ActualNegative, Tensor<T> PredictedPositive) ClassifyAtThreshold(
            Vector<T> predictions, Vector<T> actualLabels)
        {
            if (actualLabels.Length < predictions.Length)
                throw new ArgumentException(
                    $"Expected at least {predictions.Length} labels, one per prediction, but got {actualLabels.Length}.",
                    nameof(actualLabels));

            var engine = AiDotNetEngine.Current;
            T threshold = _numOps.FromDouble(0.5);
            var labels = (actualLabels.Length == predictions.Length
                ? actualLabels
                : actualLabels.GetSubVector(0, predictions.Length)).AsTensor();
            var thresholds = engine.TensorFullLike(labels, threshold);
            return (
                engine.TensorGreaterEqual(labels, thresholds),
                engine.TensorLessThan(labels, threshold),
                engine.TensorGreaterEqual(predictions.AsTensor(), thresholds));
        }
    }
}
