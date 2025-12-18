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

    /// <summary>
    /// Performs cross-validation on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="X">The input data.</param>
    /// <param name="y">The output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <param name="crossValidator">Optional custom cross-validator implementation. If not provided, a default will be used.</param>
    /// <returns>A CrossValidationResult containing the evaluation metrics for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method performs cross-validation to assess how well the model generalizes to unseen data. It splits the data into
    /// multiple subsets (folds), trains the model on a portion of the data, and evaluates it on the held-out portion. This process
    /// is repeated multiple times to get a robust estimate of the model's performance. The method allows for customization of the
    /// cross-validation process through options and even allows for a custom cross-validator implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This method tests how well your model performs on different subsets of your data.
    ///
    /// Cross-validation:
    /// - Splits your data into several parts (called folds)
    /// - Trains the model multiple times, each time using a different part as a test set
    /// - Helps understand how well the model will work on new, unseen data
    ///
    /// This is useful for:
    /// - Getting a more reliable estimate of model performance
    /// - Detecting overfitting (when a model works well on training data but poorly on new data)
    /// - Comparing different models to see which one generalizes better
    ///
    /// For example, in 5-fold cross-validation, your data is split into 5 parts. The model is trained 5 times,
    /// each time using 4 parts for training and 1 for testing. The results are then averaged to get an overall performance score.
    /// </para>
    /// </remarks>
    CrossValidationResult<T, TInput, TOutput> PerformCrossValidation(
        IFullModel<T, TInput, TOutput> model,
        TInput X,
        TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer,
        ICrossValidator<T, TInput, TOutput>? crossValidator = null);
}
