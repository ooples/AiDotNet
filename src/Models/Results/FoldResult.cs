namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of a single fold in cross-validation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A FoldResult contains all the performance metrics for one "fold" 
/// in cross-validation. Think of it like a report card for a single test of your model,
/// where the model was trained on one subset of your data and tested on another.
/// </para>
/// </remarks>
public class FoldResult<T>
{
    /// <summary>
    /// Gets the index of this fold in the cross-validation process.
    /// </summary>
    public int FoldIndex { get; }

    /// <summary>
    /// Gets the error statistics for the training data.
    /// </summary>
    public ErrorStats<T> TrainingErrors { get; }

    /// <summary>
    /// Gets the error statistics for the validation data.
    /// </summary>
    public ErrorStats<T> ValidationErrors { get; }

    /// <summary>
    /// Gets the prediction statistics for the validation data.
    /// </summary>
    public PredictionStats<T> ValidationPredictionStats { get; }

    /// <summary>
    /// Gets the actual values from the validation dataset.
    /// </summary>
    public Vector<T> ActualValues { get; }

    /// <summary>
    /// Gets the predicted values for the validation dataset.
    /// </summary>
    public Vector<T> PredictedValues { get; }

    /// <summary>
    /// Gets the feature importance scores for this fold.
    /// </summary>
    public Dictionary<string, T> FeatureImportance { get; }

    /// <summary>
    /// Gets the time taken to train the model for this fold.
    /// </summary>
    public TimeSpan TrainingTime { get; }

    /// <summary>
    /// Gets the time taken to evaluate the model for this fold.
    /// </summary>
    public TimeSpan EvaluationTime { get; }

    /// <summary>
    /// Creates a new instance of the FoldResult class.
    /// </summary>
    /// <param name="foldIndex">The index of this fold.</param>
    /// <param name="trainingActual">The actual values in the training dataset.</param>
    /// <param name="trainingPredicted">The predicted values for the training dataset.</param>
    /// <param name="validationActual">The actual values in the validation dataset.</param>
    /// <param name="validationPredicted">The predicted values for the validation dataset.</param>
    /// <param name="featureImportance">Optional dictionary of feature importance scores.</param>
    /// <param name="trainingTime">Time taken to train the model.</param>
    /// <param name="evaluationTime">Time taken to evaluate the model.</param>
    /// <param name="featureCount">The number of features used in the model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a complete report of how well your model
    /// performed on one fold of cross-validation. It calculates various error metrics and statistics
    /// that help you understand your model's strengths and weaknesses.
    /// </para>
    /// </remarks>
    public FoldResult(
        int foldIndex, 
        Vector<T> trainingActual, 
        Vector<T> trainingPredicted, 
        Vector<T> validationActual, 
        Vector<T> validationPredicted,
        Dictionary<string, T>? featureImportance = null,
        TimeSpan? trainingTime = null,
        TimeSpan? evaluationTime = null,
        int featureCount = 0,
        ModelType modelType = ModelType.None)
    {
        FoldIndex = foldIndex;
        ActualValues = validationActual;
        PredictedValues = validationPredicted;
        
        TrainingErrors = new ErrorStats<T>(new ErrorStatsInputs<T> 
        { 
            Actual = trainingActual, 
            Predicted = trainingPredicted,
            FeatureCount = featureCount
        }, modelType);
        
        ValidationErrors = new ErrorStats<T>(new ErrorStatsInputs<T> 
        { 
            Actual = validationActual, 
            Predicted = validationPredicted,
            FeatureCount = featureCount
        }, modelType);
        
        ValidationPredictionStats = new PredictionStats<T>(new PredictionStatsInputs<T>
        {
            Actual = validationActual,
            Predicted = validationPredicted,
            NumberOfParameters = featureCount,
            ConfidenceLevel = 0.95,
            LearningCurveSteps = 5
        }, modelType);
        
        FeatureImportance = featureImportance ?? [];
        TrainingTime = trainingTime ?? TimeSpan.Zero;
        EvaluationTime = evaluationTime ?? TimeSpan.Zero;
    }
}