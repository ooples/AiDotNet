namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for evaluating machine learning models.
/// </summary>
/// <remarks>
/// This interface provides functionality to assess how well a machine learning model performs
/// by comparing its predictions against known correct answers.
/// 
/// <b>For Beginners:</b> This interface helps you measure how good your AI model is at making predictions.
/// 
/// What is model evaluation?
/// - Model evaluation is the process of checking how accurate your AI model is
/// - It compares the model's predictions with the actual correct answers
/// - It calculates various metrics (like accuracy, error rates) to measure performance
/// - It helps you understand if your model is good enough to use in real situations
/// 
/// Real-world analogy:
/// Think of model evaluation like grading a test. If a student (your model) takes a test with
/// questions (input data) and provides answers (predictions), the teacher (evaluator) compares
/// these answers to the answer key (actual values) and assigns a score. This score tells you
/// how well the student understands the material. Similarly, model evaluation tells you how
/// well your AI model understands the patterns in your data.
/// 
/// Why evaluate models?
/// - To know if your model is accurate enough for real use
/// - To compare different models and choose the best one
/// - To identify where your model makes mistakes
/// - To detect problems like overfitting (when a model works well on training data but
///   poorly on new data)
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IModelEvaluator<T, TInput, TOutput>
{
    /// <summary>
    /// Evaluates a machine learning model using the provided input data and returns detailed performance metrics.
    /// </summary>
    /// <remarks>
    /// This method takes input data containing the model, test data, and evaluation parameters,
    /// then returns comprehensive metrics about the model's performance.
    /// 
    /// <b>For Beginners:</b> This method tests your AI model and tells you how well it performs.
    /// 
    /// What happens during evaluation:
    /// 1. The model makes predictions using the test data
    /// 2. These predictions are compared to the known correct answers
    /// 3. Various performance metrics are calculated (like accuracy, error rates)
    /// 4. The results are returned in a structured format
    /// 
    /// Common evaluation metrics explained:
    /// - Accuracy: The percentage of predictions that are correct
    ///   (e.g., 90% accuracy means 9 out of 10 predictions are right)
    /// - Error: How far off the predictions are from the actual values
    ///   (smaller errors are better)
    /// - Precision: How many of the positive predictions are actually correct
    ///   (important when false positives are costly)
    /// - Recall: How many of the actual positives were correctly identified
    ///   (important when false negatives are costly)
    /// 
    /// When to use this method:
    /// - After training a model to see how well it performs
    /// - When comparing different models to choose the best one
    /// - When tuning model parameters to improve performance
    /// - Before deploying a model to ensure it meets quality standards
    /// </remarks>
    /// <param name="input">
    /// The input data for evaluation, containing:
    /// - The model to evaluate
    /// - Test data (inputs and known correct outputs)
    /// - Evaluation parameters and settings
    /// </param>
    /// <returns>
    /// Detailed evaluation results including:
    /// - Performance metrics (accuracy, error rates, etc.)
    /// - Comparison between predictions and actual values
    /// - Additional insights about model performance
    /// </returns>
    ModelEvaluationData<T, TInput, TOutput> EvaluateModel(ModelEvaluationInput<T, TInput, TOutput> input);
}