namespace AiDotNet.Models;

/// <summary>
/// Represents a comprehensive collection of statistical measures and data for evaluating model performance on a dataset.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates various statistical measures and the actual data used to evaluate a model's performance on a 
/// specific dataset. It includes error statistics that quantify prediction errors, basic statistics for both actual and 
/// predicted values, prediction quality statistics, and the raw data including features, actual values, and predicted values. 
/// This comprehensive collection of information allows for thorough analysis of model performance on the dataset.
/// </para>
/// <para><b>For Beginners:</b> This class stores all the statistics and data needed to evaluate how well a model performs.
/// 
/// When evaluating a model's performance:
/// - You need to measure different aspects of accuracy and error
/// - You want to compare actual values with predicted values
/// - You need to keep track of the input data that produced the predictions
/// 
/// This class stores all that information, including:
/// - Various error measurements (how far predictions are from actual values)
/// - Basic statistics about both actual values and predictions
/// - Statistics about prediction quality (how well the model captures patterns)
/// - The actual input features, target values, and model predictions
/// 
/// Having all this information in one place makes it easier to analyze model performance,
/// create visualizations, and compare different models.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix<T> for regression, Tensor<T> for neural networks).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector<T> for regression, Tensor<T> for neural networks).</typeparam>
public class DataSetStats<T, TInput, TOutput>
{
    private ModelType _modelType = default!;
    private ErrorStats<T>? _errorStats;
    private PredictionStats<T>? _predictionStats;
    private BasicStats<T>? _actualBasicStats;
    private BasicStats<T>? _predictedBasicStats;

    /// <summary>
    /// Gets the model type used for determining which statistics to calculate.
    /// </summary>
    public ModelType ModelType => _modelType;

    /// <summary>
    /// Gets the error statistics for the model's predictions.
    /// </summary>
    /// <value>An ErrorStats&lt;T&gt; object containing various error metrics.</value>
    public ErrorStats<T> ErrorStats
    {
        get => _errorStats ??= ErrorStats<T>.Empty(_modelType);
        set => _errorStats = value;
    }

    /// <summary>
    /// Gets the basic descriptive statistics for the actual target values.
    /// </summary>
    /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the actual values.</value>
    public BasicStats<T> ActualBasicStats
    {
        get => _actualBasicStats ??= BasicStats<T>.Empty(_modelType);
        set => _actualBasicStats = value;
    }

    /// <summary>
    /// Gets the basic descriptive statistics for the predicted values.
    /// </summary>
    /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the predicted values.</value>
    public BasicStats<T> PredictedBasicStats
    {
        get => _predictedBasicStats ??= BasicStats<T>.Empty(_modelType);
        set => _predictedBasicStats = value;
    }

    /// <summary>
    /// Gets the prediction quality statistics for the model.
    /// </summary>
    /// <value>A PredictionStats&lt;T&gt; object containing various prediction quality metrics.</value>
    public PredictionStats<T> PredictionStats
    {
        get => _predictionStats ??= PredictionStats<T>.Empty(_modelType);
        set => _predictionStats = value;
    }

    /// <summary>
    /// Gets or sets the predicted values.
    /// </summary>
    /// <value>The predicted values of type TOutput.</value>
    public TOutput Predicted { get; set; }

    /// <summary>
    /// Gets or sets the actual target values.
    /// </summary>
    /// <value>The actual target values of type TOutput.</value>
    public TOutput Actual { get; set; }

    /// <summary>
    /// Gets or sets the input features.
    /// </summary>
    /// <value>The input features of type TInput.</value>
    public TInput Features { get; set; }

    /// <summary>
    /// Initializes a new instance of the DataSetStats class with default empty model data and ModelType.None.
    /// </summary>
    /// <remarks>
    /// This constructor uses ModelType.None which will only enable general metrics. 
    /// Use the constructor with ModelType parameter to enable model-specific metrics.
    /// Statistics objects are lazily initialized when first accessed.
    /// </remarks>
    public DataSetStats() : this(ModelType.None)
    {
    }

    /// <summary>
    /// Initializes a new instance of the DataSetStats class with default empty model data for the specified model type.
    /// </summary>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <remarks>
    /// This constructor sets up the model type but doesn't create statistics objects until they're accessed,
    /// reducing unnecessary allocations.
    /// </remarks>
    public DataSetStats(ModelType modelType)
    {
        _modelType = modelType;
        (Features, Actual, Predicted) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
    }

    /// <summary>
    /// Creates a new instance of DataSetStats with pre-calculated statistics.
    /// </summary>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="features">The input features.</param>
    /// <param name="actual">The actual target values.</param>
    /// <param name="predicted">The predicted values.</param>
    /// <param name="errorStatsInputs">Optional inputs for error statistics calculation.</param>
    /// <param name="predictionStatsInputs">Optional inputs for prediction statistics calculation.</param>
    /// <returns>A DataSetStats instance populated with calculated statistics.</returns>
    public static DataSetStats<T, TInput, TOutput> CreateWithCalculatedStats(
        ModelType modelType,
        TInput features,
        TOutput actual,
        TOutput predicted,
        ErrorStatsInputs<T>? errorStatsInputs = null,
        PredictionStatsInputs<T>? predictionStatsInputs = null)
    {
        var stats = new DataSetStats<T, TInput, TOutput>(modelType)
        {
            Features = features,
            Actual = actual,
            Predicted = predicted
        };

        var actualVector = ConversionsHelper.ConvertToVector<T, TOutput>(actual);
        var predictedVector = ConversionsHelper.ConvertToVector<T, TOutput>(predicted);

        if (!actualVector.IsEmpty)
        {
            stats._actualBasicStats = new BasicStats<T>(actualVector, modelType);
        }

        if (!predictedVector.IsEmpty)
        {
            stats._predictedBasicStats = new BasicStats<T>(predictedVector, modelType);
        }

        // Calculate error stats if inputs provided or can be derived
        if (errorStatsInputs != null)
        {
            stats._errorStats = new ErrorStats<T>(errorStatsInputs, modelType);
        }
        else if (!actualVector.IsEmpty && !predictedVector.IsEmpty)
        {
            var inputs = new ErrorStatsInputs<T>
            {
                Actual = actualVector,
                Predicted = predictedVector,
                FeatureCount = 0 // This should be provided or calculated from features
            };
            stats._errorStats = new ErrorStats<T>(inputs, modelType);
        }

        // Calculate prediction stats if inputs provided
        if (predictionStatsInputs != null)
        {
            stats._predictionStats = new PredictionStats<T>(predictionStatsInputs, modelType);
        }
        else if (!actualVector.IsEmpty && !predictedVector.IsEmpty)
        {
            var inputs = new PredictionStatsInputs<T>
            {
                Actual = actualVector,
                Predicted = predictedVector,
                NumberOfParameters = 0, // This should be provided
                PredictionType = PredictionType.Regression // This should be determined from modelType
            };
            stats._predictionStats = new PredictionStats<T>(inputs, modelType);
        }

        return stats;
    }

    /// <summary>
    /// Updates the model type and invalidates cached statistics objects.
    /// </summary>
    /// <param name="modelType">The new model type to use.</param>
    /// <remarks>
    /// This method allows changing the model type after initialization. It will clear any
    /// cached statistics objects, which will be recreated with the new model type when accessed.
    /// </remarks>
    public void UpdateModelType(ModelType modelType)
    {
        if (_modelType != modelType)
        {
            _modelType = modelType;
            _errorStats = null;
            _predictionStats = null;
            // Note: BasicStats doesn't depend on model type, so we keep those
        }
    }

    /// <summary>
    /// Determines if all statistics have been calculated.
    /// </summary>
    /// <returns>True if all statistics objects have been initialized; otherwise, false.</returns>
    public bool AreAllStatsCalculated()
    {
        return _errorStats != null &&
               _predictionStats != null &&
               _actualBasicStats != null &&
               _predictedBasicStats != null;
    }

    /// <summary>
    /// Forces calculation of all statistics objects.
    /// </summary>
    /// <remarks>
    /// This method ensures all statistics objects are initialized, creating empty ones if necessary.
    /// Use this when you need to ensure all properties are non-null.
    /// </remarks>
    public void EnsureAllStatsInitialized()
    {
        _ = ErrorStats;
        _ = PredictionStats;
        _ = ActualBasicStats;
        _ = PredictedBasicStats;
    }
}