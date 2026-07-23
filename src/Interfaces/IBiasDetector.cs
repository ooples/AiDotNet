using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines an interface for detecting bias in machine learning model predictions.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This interface helps identify unfair treatment in machine learning models.
    ///
    /// In machine learning, bias occurs when a model treats different groups of people unfairly.
    /// For example, a loan approval model might unfairly reject applications from certain demographic groups.
    ///
    /// This interface provides methods to:
    /// - Detect when a model is biased against certain groups
    /// - Measure how severe the bias is
    /// - Compare different models to find which one is most fair
    ///
    /// The bias score measures how unfair the model is. Important points:
    /// - Lower bias scores usually indicate fairer models
    /// - The score helps us identify which groups are being treated unfairly
    /// - We can use these measurements to improve our models and make them more equitable
    /// </remarks>
    public interface IBiasDetector<T>
    {
        /// <summary>
        /// Detects bias in model predictions by analyzing predictions across different groups.
        /// </summary>
        /// <param name="predictions">The model's predictions.</param>
        /// <param name="sensitiveFeature">The sensitive feature (e.g., race, gender) used to identify groups.</param>
        /// <param name="actualLabels">Optional actual labels for computing additional bias metrics.</param>
        /// <returns>A result object containing bias detection metrics and analysis.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This method checks if your model treats different groups fairly.
        ///
        /// The input parameters:
        /// - predictions: The outputs your model produced
        /// - sensitiveFeature: Information about which group each person belongs to (like age, gender, etc.)
        /// - actualLabels: The correct answers (optional, but provides more detailed analysis if available)
        ///
        /// The method analyzes whether your model gives similar results to all groups or if it
        /// favors some groups over others. For example, it might detect if a hiring model
        /// recommends hiring men more often than equally qualified women.
        ///
        /// The result includes:
        /// - Whether bias was detected (true/false)
        /// - How severe the bias is
        /// - Which groups are affected
        /// - Specific metrics measuring different types of unfairness
        /// </remarks>
        BiasDetectionResult<T> DetectBias(Vector<T> predictions, Vector<T> sensitiveFeature, Vector<T>? actualLabels = null);

        /// <summary>
        /// Indicates whether lower bias scores represent better (fairer) models.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This property tells you whether smaller numbers mean fairer models.
        ///
        /// For bias detection, this is typically true - lower bias scores mean the model is
        /// treating all groups more equally. A bias score of 0 would mean perfect fairness.
        ///
        /// This property returns:
        /// - true: if lower scores mean fairer models (typical for bias metrics)
        /// - false: if higher scores mean fairer models (less common)
        ///
        /// Knowing this helps you correctly interpret and compare bias scores across different models.
        /// </remarks>
        bool IsLowerBiasBetter { get; }

        /// <summary>
        /// Compares two bias scores and determines if the current score represents better (fairer) performance.
        /// </summary>
        /// <param name="currentBias">The bias score to evaluate.</param>
        /// <param name="bestBias">The best (fairest) bias score found so far.</param>
        /// <returns>True if the current bias score is better (fairer) than the best bias score; otherwise, false.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This method compares two bias scores and tells you which model is fairer.
        ///
        /// When comparing models, you need to know which one is more fair. This method handles
        /// the comparison logic for you, taking into account whether higher or lower scores indicate better fairness.
        ///
        /// The input parameters:
        /// - currentBias: The bias score of a model you want to evaluate
        /// - bestBias: The best (fairest) score you've seen so far
        ///
        /// The method returns:
        /// - true: if the current model is fairer than the best model so far
        /// - false: if the current model is less fair than or equal to the best model
        ///
        /// This is particularly useful when testing many different models to find the fairest one.
        /// </remarks>
        bool IsBetterBiasScore(T currentBias, T bestBias);
    }
}
