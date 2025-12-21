using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Base class for all bias detectors that identify unfair treatment in model predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a foundation class that all bias detectors build upon.
    ///
    /// Think of a bias detector like an inspector checking for fairness:
    /// - It examines how your model makes predictions for different groups of people
    /// - It identifies when certain groups are being treated unfairly
    /// - It provides metrics that measure the severity of the bias
    ///
    /// Different bias detectors might look for different types of unfairness, but they all
    /// share common functionality. This base class provides that shared foundation.
    /// </para>
    /// </remarks>
    public abstract class BiasDetectorBase<T> : IBiasDetector<T>
    {
        /// <summary>
        /// Provides mathematical operations for the specific numeric type being used.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is a toolkit that helps perform math operations
        /// regardless of whether we're using integers, decimals, doubles, etc.
        ///
        /// It allows the detector to work with different numeric types without
        /// having to rewrite the math operations for each type.
        /// </para>
        /// </remarks>
        protected readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Indicates whether lower bias scores represent better (fairer) models.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This tells us whether smaller numbers mean fairer models.
        ///
        /// For bias detection:
        /// - Lower bias scores typically indicate fairer models (closer to equal treatment)
        /// - A bias score of 0 would indicate perfect fairness
        ///
        /// This helps the system know how to compare different models for fairness.
        /// </para>
        /// </remarks>
        protected readonly bool _isLowerBiasBetter;

        /// <summary>
        /// Initializes a new instance of the BiasDetectorBase class.
        /// </summary>
        /// <param name="isLowerBiasBetter">Indicates whether lower bias scores represent better (fairer) models.</param>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This sets up the basic properties of the bias detector.
        ///
        /// Parameters:
        /// - isLowerBiasBetter: Tells the system whether smaller numbers mean fairer models
        ///   (typically true for bias metrics, where 0 represents perfect fairness)
        /// </para>
        /// </remarks>
        protected BiasDetectorBase(bool isLowerBiasBetter)
        {
            _isLowerBiasBetter = isLowerBiasBetter;
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Detects bias in model predictions by analyzing predictions across different groups.
        /// </summary>
        /// <param name="predictions">The model's predictions.</param>
        /// <param name="sensitiveFeature">The sensitive feature (e.g., race, gender) used to identify groups.</param>
        /// <param name="actualLabels">Optional actual labels for computing additional bias metrics.</param>
        /// <returns>A result object containing bias detection metrics and analysis.</returns>
        /// <exception cref="ArgumentNullException">Thrown when predictions or sensitiveFeature is null.</exception>
        /// <exception cref="ArgumentException">Thrown when predictions and sensitiveFeature have different lengths.</exception>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method checks if your model treats different groups fairly.
        ///
        /// It works by:
        /// 1. Validating that all required data is provided and properly formatted
        /// 2. Calling the specific bias detection logic implemented by derived classes
        /// 3. Returning detailed results about any bias found
        ///
        /// The method handles the common validation logic, while the specific bias detection
        /// algorithm is defined in each detector that extends this base class.
        /// </para>
        /// </remarks>
        public BiasDetectionResult<T> DetectBias(Vector<T> predictions, Vector<T> sensitiveFeature, Vector<T>? actualLabels = null)
        {
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));

            if (sensitiveFeature == null)
                throw new ArgumentNullException(nameof(sensitiveFeature));

            if (predictions.Length != sensitiveFeature.Length)
                throw new ArgumentException($"Predictions and sensitive feature lengths must match. Predictions: {predictions.Length}, Sensitive feature: {sensitiveFeature.Length}");

            if (actualLabels != null && predictions.Length != actualLabels.Length)
                throw new ArgumentException(
                    $"Predictions and actual labels lengths must match. Predictions: {predictions.Length}, Actual labels: {actualLabels.Length}");

            return GetBiasDetectionResult(predictions, sensitiveFeature, actualLabels);
        }

        /// <summary>
        /// Abstract method that must be implemented by derived classes to perform specific bias detection logic.
        /// </summary>
        /// <param name="predictions">The model's predictions.</param>
        /// <param name="sensitiveFeature">The sensitive feature used to identify groups.</param>
        /// <param name="actualLabels">Optional actual labels for computing additional bias metrics.</param>
        /// <returns>A result object containing bias detection metrics and analysis.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This is a placeholder method that each specific detector must fill in.
        ///
        /// Think of it like a template that says "here's where you put your specific bias detection logic."
        /// Each detector that extends this base class will provide its own implementation of this method,
        /// defining exactly how it detects and measures bias.
        ///
        /// For example:
        /// - A disparate impact detector would check if positive outcomes are equally distributed
        /// - An equal opportunity detector would check if qualified individuals have equal chances
        /// - A demographic parity detector would check for balanced outcomes across groups
        /// </para>
        /// </remarks>
        protected abstract BiasDetectionResult<T> GetBiasDetectionResult(
            Vector<T> predictions,
            Vector<T> sensitiveFeature,
            Vector<T>? actualLabels);

        /// <summary>
        /// Gets a value indicating whether lower bias scores represent better (fairer) models.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This property tells you whether smaller numbers mean fairer models.
        ///
        /// For most bias metrics:
        /// - IsLowerBiasBetter is true (0 bias means perfect fairness)
        /// - Lower values indicate the model treats different groups more equally
        ///
        /// This helps you interpret the scores correctly when comparing different models.
        /// </para>
        /// </remarks>
        public bool IsLowerBiasBetter => _isLowerBiasBetter;

        /// <summary>
        /// Determines whether a new bias score represents better (fairer) performance than the current best score.
        /// </summary>
        /// <param name="currentBias">The current bias score to evaluate.</param>
        /// <param name="bestBias">The best (fairest) bias score found so far.</param>
        /// <returns>True if the current bias score is better (fairer) than the best bias score; otherwise, false.</returns>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> This method compares two bias scores and tells you which model is fairer.
        ///
        /// It takes into account whether higher scores are better or lower scores are better:
        /// - If lower scores are better (typical for bias), it returns true when the new score is lower
        /// - If higher scores are better (less common), it returns true when the new score is higher
        ///
        /// This is particularly useful when:
        /// - Selecting the fairest model from multiple options
        /// - Deciding whether model changes improved fairness
        /// - Tracking fairness improvements during model development
        /// </para>
        /// </remarks>
        public bool IsBetterBiasScore(T currentBias, T bestBias)
        {
            return _isLowerBiasBetter
                ? _numOps.LessThan(currentBias, bestBias)
                : _numOps.GreaterThan(currentBias, bestBias);
        }
    }
}
