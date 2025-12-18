using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Base class for all fairness evaluators that measure equitable treatment in models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a foundation class that all fairness evaluators build upon.
    ///
    /// Think of a fairness evaluator like a comprehensive audit:
    /// - It examines your model's behavior across multiple dimensions of fairness
    /// - It measures various fairness metrics (demographic parity, equal opportunity, etc.)
    /// - It provides a complete picture of how equitably your model treats different groups
    ///
    /// Different fairness evaluators might focus on different combinations of metrics, but they all
    /// share common functionality. This base class provides that shared foundation.
    /// </para>
    /// </remarks>
    public abstract class FairnessEvaluatorBase<T> : IFairnessEvaluator<T>
    {
        /// <summary>
        /// Provides mathematical operations for the specific numeric type being used.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is a toolkit that helps perform math operations
        /// regardless of whether we're using integers, decimals, doubles, etc.
        ///
        /// It allows the evaluator to work with different numeric types without
        /// having to rewrite the math operations for each type.
        /// </para>
        /// </remarks>
        protected readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Indicates whether higher fairness scores represent better (more equitable) models.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This tells us whether bigger numbers mean fairer models.
        ///
        /// For fairness evaluation:
        /// - Some metrics work where higher is better (e.g., disparate impact ratio closer to 1)
        /// - Other metrics work where lower is better (e.g., demographic parity difference closer to 0)
        ///
        /// This property indicates the general direction for the evaluator's primary metric.
        /// </para>
        /// </remarks>
        protected readonly bool _isHigherFairnessBetter;

        /// <summary>
        /// Initializes a new instance of the FairnessEvaluatorBase class.
        /// </summary>
        /// <param name="isHigherFairnessBetter">Indicates whether higher fairness scores represent better (more equitable) models.</param>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This sets up the basic properties of the fairness evaluator.
        ///
        /// Parameters:
        /// - isHigherFairnessBetter: Tells the system whether bigger numbers mean fairer models
        ///   (depends on which fairness metric is used as the primary measure)
        /// </para>
        /// </remarks>
        protected FairnessEvaluatorBase(bool isHigherFairnessBetter)
        {
            _isHigherFairnessBetter = isHigherFairnessBetter;
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Evaluates fairness of a model by analyzing its predictions across different groups.
        /// </summary>
        /// <param name="model">The model to evaluate for fairness.</param>
        /// <param name="inputs">The input data containing features and sensitive attributes.</param>
        /// <param name="sensitiveFeatureIndex">Index of the column containing the sensitive feature.</param>
        /// <param name="actualLabels">Optional actual labels for computing accuracy-based fairness metrics.</param>
        /// <returns>A FairnessMetrics object containing comprehensive fairness measurements.</returns>
        /// <exception cref="ArgumentNullException">Thrown when model or inputs is null.</exception>
        /// <exception cref="ArgumentException">Thrown when sensitiveFeatureIndex is invalid or when actualLabels length doesn't match inputs.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method measures how fairly your model treats different groups.
        ///
        /// It works by:
        /// 1. Validating that all required data is provided and properly formatted
        /// 2. Calling the specific fairness evaluation logic implemented by derived classes
        /// 3. Returning comprehensive metrics about the model's fairness
        ///
        /// The method handles the common validation logic, while the specific fairness calculations
        /// are defined in each evaluator that extends this base class.
        /// </para>
        /// </remarks>
        public FairnessMetrics<T> EvaluateFairness(
            IFullModel<T, Matrix<T>, Vector<T>> model,
            Matrix<T> inputs,
            int sensitiveFeatureIndex,
            Vector<T>? actualLabels = null)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            if (inputs == null)
                throw new ArgumentNullException(nameof(inputs));

            if (sensitiveFeatureIndex < 0 || sensitiveFeatureIndex >= inputs.Columns)
                throw new ArgumentException($"Sensitive feature index {sensitiveFeatureIndex} is out of range. Matrix has {inputs.Columns} columns.", nameof(sensitiveFeatureIndex));

            if (actualLabels != null && actualLabels.Length != inputs.Rows)
                throw new ArgumentException($"Actual labels length ({actualLabels.Length}) must match the number of input rows ({inputs.Rows}).", nameof(actualLabels));

            return GetFairnessMetrics(model, inputs, sensitiveFeatureIndex, actualLabels);
        }

        /// <summary>
        /// Abstract method that must be implemented by derived classes to perform specific fairness evaluation logic.
        /// </summary>
        /// <param name="model">The model to evaluate for fairness.</param>
        /// <param name="inputs">The input data containing features and sensitive attributes.</param>
        /// <param name="sensitiveFeatureIndex">Index of the column containing the sensitive feature.</param>
        /// <param name="actualLabels">Optional actual labels for computing accuracy-based fairness metrics.</param>
        /// <returns>A FairnessMetrics object containing comprehensive fairness measurements.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is a placeholder method that each specific evaluator must fill in.
        ///
        /// Think of it like a template that says "here's where you put your specific fairness evaluation logic."
        /// Each evaluator that extends this base class will provide its own implementation of this method,
        /// defining exactly how it calculates various fairness metrics.
        ///
        /// For example:
        /// - A comprehensive evaluator might calculate all major fairness metrics
        /// - A specialized evaluator might focus on specific metrics like equal opportunity
        /// - A custom evaluator might implement domain-specific fairness measures
        /// </para>
        /// </remarks>
        protected abstract FairnessMetrics<T> GetFairnessMetrics(
            IFullModel<T, Matrix<T>, Vector<T>> model,
            Matrix<T> inputs,
            int sensitiveFeatureIndex,
            Vector<T>? actualLabels);

        /// <summary>
        /// Gets a value indicating whether higher fairness scores represent better (more equitable) models.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This property tells you whether bigger numbers mean fairer models.
        ///
        /// The interpretation depends on which fairness metric is used:
        /// - For some metrics (like disparate impact ratio), higher values mean more fairness
        /// - For other metrics (like demographic parity difference), lower values mean more fairness
        ///
        /// This property indicates the direction for the evaluator's primary or aggregate metric.
        /// </para>
        /// </remarks>
        public bool IsHigherFairnessBetter => _isHigherFairnessBetter;

        /// <summary>
        /// Determines whether a new fairness score represents better (more equitable) performance than the current best score.
        /// </summary>
        /// <param name="currentFairness">The current fairness score to evaluate.</param>
        /// <param name="bestFairness">The best (most equitable) fairness score found so far.</param>
        /// <returns>True if the current fairness score is better (more equitable) than the best fairness score; otherwise, false.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method compares two fairness scores and tells you which model is more equitable.
        ///
        /// It takes into account whether higher scores are better or lower scores are better:
        /// - If higher scores are better, it returns true when the new score is higher
        /// - If lower scores are better, it returns true when the new score is lower
        ///
        /// This is particularly useful when:
        /// - Selecting the most fair model from multiple options
        /// - Deciding whether model changes improved fairness
        /// - Tracking fairness improvements during model development
        /// </para>
        /// </remarks>
        public bool IsBetterFairnessScore(T currentFairness, T bestFairness)
        {
            return _isHigherFairnessBetter
                ? _numOps.GreaterThan(currentFairness, bestFairness)
                : _numOps.LessThan(currentFairness, bestFairness);
        }
    }
}
