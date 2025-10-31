using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines an interface for evaluating fairness in machine learning models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
    /// <remarks>
    /// <b>For Beginners:</b> This interface helps measure how fair a machine learning model is.
    ///
    /// Fairness in machine learning means treating all groups of people equally. A fair model
    /// makes similar decisions for people with similar qualifications, regardless of sensitive
    /// attributes like race, gender, or age.
    ///
    /// This interface provides methods to:
    /// - Evaluate how fair a model is across different groups
    /// - Measure various aspects of fairness (demographic parity, equal opportunity, etc.)
    /// - Compare different models to find which one is most equitable
    ///
    /// The fairness score measures how equitable the model is. Important points:
    /// - Higher fairness scores can indicate more equitable treatment (depends on the metric)
    /// - Multiple fairness metrics exist because fairness can be defined in different ways
    /// - We can use these measurements to improve our models and ensure equal treatment
    /// </remarks>
    public interface IFairnessEvaluator<T>
    {
        /// <summary>
        /// Evaluates fairness of a model by analyzing its predictions across different groups.
        /// </summary>
        /// <param name="model">The model to evaluate for fairness.</param>
        /// <param name="inputs">The input data containing features and sensitive attributes.</param>
        /// <param name="sensitiveFeatureIndex">Index of the column containing the sensitive feature.</param>
        /// <param name="actualLabels">Optional actual labels for computing accuracy-based fairness metrics.</param>
        /// <returns>A FairnessMetrics object containing comprehensive fairness measurements.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This method measures how fairly your model treats different groups.
        ///
        /// The input parameters:
        /// - model: The machine learning model you want to evaluate
        /// - inputs: The data containing both regular features and sensitive attributes (like age, gender)
        /// - sensitiveFeatureIndex: Which column in your data represents the sensitive attribute
        /// - actualLabels: The correct answers (optional, but enables more detailed fairness metrics)
        ///
        /// The method analyzes whether your model gives fair outcomes to all groups. For example,
        /// it might check if a loan approval model approves loans at similar rates for all age groups
        /// when other qualifications are equal.
        ///
        /// The result includes multiple fairness metrics:
        /// - Demographic Parity: Are positive outcomes distributed equally?
        /// - Equal Opportunity: Do qualified individuals have equal chances?
        /// - Equalized Odds: Are both true and false positive rates similar?
        /// - Predictive Parity: Is prediction accuracy similar across groups?
        /// - And more...
        /// </remarks>
        FairnessMetrics<T> EvaluateFairness(IFullModel<T, Matrix<T>, Vector<T>> model, Matrix<T> inputs, int sensitiveFeatureIndex, Vector<T>? actualLabels = null);

        /// <summary>
        /// Indicates whether higher fairness scores represent better (more equitable) models.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This property tells you whether bigger numbers mean fairer models.
        ///
        /// Different fairness metrics work in opposite ways:
        /// - For some metrics like disparate impact ratio, higher values mean more fairness
        /// - For other metrics like demographic parity difference, lower values mean more fairness
        ///
        /// This property returns:
        /// - true: if higher scores mean fairer models
        /// - false: if lower scores mean fairer models
        ///
        /// Note: This property indicates the general direction for the evaluator's primary metric.
        /// Individual fairness metrics within FairnessMetrics may have their own interpretations.
        ///
        /// Knowing this helps you correctly interpret and compare overall fairness scores.
        /// </remarks>
        bool IsHigherFairnessBetter { get; }

        /// <summary>
        /// Compares two fairness scores and determines if the current score represents better (more equitable) performance.
        /// </summary>
        /// <param name="currentFairness">The fairness score to evaluate.</param>
        /// <param name="bestFairness">The best (most equitable) fairness score found so far.</param>
        /// <returns>True if the current fairness score is better than the best fairness score; otherwise, false.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This method compares two fairness scores and tells you which model is more equitable.
        ///
        /// When comparing models, you need to know which one treats people more fairly. This method
        /// handles the comparison logic for you, taking into account whether higher or lower scores
        /// indicate better fairness for the specific metric being used.
        ///
        /// The input parameters:
        /// - currentFairness: The fairness score of a model you want to evaluate
        /// - bestFairness: The best (most equitable) score you've seen so far
        ///
        /// The method returns:
        /// - true: if the current model is more fair than the best model so far
        /// - false: if the current model is less fair than or equal to the best model
        ///
        /// This is particularly useful when testing many different models to find the most equitable one.
        /// </remarks>
        bool IsBetterFairnessScore(T currentFairness, T bestFairness);
    }
}
