

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
    /// <summary>
    /// Gets or sets the error statistics for the model's predictions.
    /// </summary>
    /// <value>An ErrorStats&lt;T&gt; object containing various error metrics.</value>
    /// <remarks>
    /// [Existing remarks for ErrorStats]
    /// </remarks>
    public ErrorStats<T> ErrorStats { get; set; } = ErrorStats<T>.Empty();

    /// <summary>
    /// Gets or sets the basic descriptive statistics for the actual target values.
    /// </summary>
    /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the actual values.</value>
    /// <remarks>
    /// [Existing remarks for ActualBasicStats]
    /// </remarks>
    public BasicStats<T> ActualBasicStats { get; set; } = BasicStats<T>.Empty();

    /// <summary>
    /// Gets or sets the basic descriptive statistics for the predicted values.
    /// </summary>
    /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the predicted values.</value>
    /// <remarks>
    /// [Existing remarks for PredictedBasicStats]
    /// </remarks>
    public BasicStats<T> PredictedBasicStats { get; set; } = BasicStats<T>.Empty();

    /// <summary>
    /// Gets or sets the prediction quality statistics for the model.
    /// </summary>
    /// <value>A PredictionStats&lt;T&gt; object containing various prediction quality metrics.</value>
    /// <remarks>
    /// [Existing remarks for PredictionStats]
    /// </remarks>
    public PredictionStats<T> PredictionStats { get; set; } = PredictionStats<T>.Empty();

    /// <summary>
    /// Gets or sets uncertainty quantification diagnostics for the dataset.
    /// </summary>
    /// <remarks>
    /// This is populated when uncertainty quantification is enabled and the evaluation flow requests UQ diagnostics.
    /// </remarks>
    public UncertaintyStats<T> UncertaintyStats { get; set; } = UncertaintyStats<T>.Empty();

    /// <summary>
    /// Gets or sets adversarial robustness diagnostics for the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is populated when adversarial robustness evaluation is enabled. It contains metrics such as
    /// clean accuracy, adversarial accuracy, certified accuracy, and attack success rates.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how well the model resists adversarial attacks.
    /// Adversarial attacks are specially crafted inputs designed to fool machine learning models.
    /// High adversarial accuracy means the model is robust against such attacks.
    /// </para>
    /// </remarks>
    public RobustnessStats<T> RobustnessStats { get; set; } = RobustnessStats<T>.Empty();

    /// <summary>
    /// Gets or sets the predicted values.
    /// </summary>
    /// <value>The predicted values of type TOutput.</value>
    /// <remarks>
    /// <para>
    /// This property contains the model's predictions for the dataset. The structure of this object depends on the type of model and prediction task.
    /// For regression tasks, it might be a vector where each element corresponds to an observation in the dataset. For more complex tasks like
    /// image classification or sequence prediction, it might be a tensor or other multi-dimensional structure.
    /// </para>
    /// [Existing remarks for Predicted]
    /// </remarks>
    public TOutput Predicted { get; set; }

    /// <summary>
    /// Gets or sets the actual target values.
    /// </summary>
    /// <value>The actual target values of type TOutput.</value>
    /// <remarks>
    /// <para>
    /// This property contains the actual target values for the dataset. The structure of this object matches that of the Predicted property
    /// and depends on the type of model and prediction task. These values represent the true outcomes that the model attempts to predict.
    /// </para>
    /// [Existing remarks for Actual]
    /// </remarks>
    public TOutput Actual { get; set; }

    /// <summary>
    /// Gets or sets the input features.
    /// </summary>
    /// <value>The input features of type TInput.</value>
    /// <remarks>
    /// <para>
    /// This property contains the input feature data for the dataset. The structure of this object depends on the type of model and input data.
    /// For traditional machine learning tasks, it might be a matrix where each row represents an observation and each column a feature.
    /// For more complex tasks like image or sequence processing, it might be a tensor or other multi-dimensional structure.
    /// </para>
    /// [Existing remarks for Features]
    /// </remarks>
    public TInput Features { get; set; }

    /// <summary>
    /// Initializes a new instance of the DataSetStats class with default empty model data.
    /// </summary>
    public DataSetStats()
    {
        (Features, Actual, Predicted) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
    }
}
