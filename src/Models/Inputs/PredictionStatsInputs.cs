namespace AiDotNet.Models.Inputs;


/// <summary>
/// Container for inputs to the PredictionStats class.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class encapsulates all the inputs required for calculating prediction statistics.
/// It ensures that all necessary parameters are provided in a structured way.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class is like a box that holds all the different pieces of 
/// information needed to calculate prediction statistics. Grouping them this way makes 
/// it easier to pass them around in your code.
/// </para>
/// </remarks>
public class PredictionStatsInputs<T>
{
    /// <summary>
    /// Vector<double> of actual values (ground truth).
    /// </summary>
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Vector<double> of predicted values from your model.
    /// </summary>
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Number of features or parameters in your model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is the number of input variables or features that your model uses to make predictions.
    /// For example, if you're predicting house prices based on size, location, and age, then
    /// the number of parameters would be 3.
    /// </remarks>
    public int NumberOfParameters { get; set; } = 1;

    /// <summary>
    /// The confidence level for statistical intervals (e.g., 0.95 for 95% confidence).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The confidence level tells you how certain you want to be about your intervals.
    /// A 95% confidence level (0.95) means you want to be 95% confident that the true
    /// value falls within the calculated interval.
    /// </remarks>
    public double ConfidenceLevel { get; set; } = 0.95;

    /// <summary>
    /// Number of steps to use when calculating the learning curve.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This controls how many points are calculated in the learning curve.
    /// A learning curve shows how your model's performance improves as it sees more training data.
    /// More steps give you a more detailed curve but take longer to calculate.
    /// </remarks>
    public int LearningCurveSteps { get; set; } = 5;

    /// <summary>
    /// The type of prediction (regression, binary classification, etc.).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This tells the system what kind of prediction your model is making:
    /// - Regression: predicting continuous values (like prices or temperatures)
    /// - Binary Classification: predicting one of two classes (like yes/no, spam/not spam)
    /// - Multiclass Classification: predicting one of several classes (like dog/cat/bird)
    /// </remarks>
    public PredictionType PredictionType { get; set; } = PredictionType.Regression;

    /// <summary>
    /// The number of time steps ahead that are being forecast (for time series models).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// For time series forecasting, this tells you how many steps into the future you're predicting.
    /// For example, if you're using daily data to predict 7 days ahead, the forecast horizon is 7.
    /// </remarks>
    public int ForecastHorizon { get; set; } = 1;

    /// <summary>
    /// The number of classes in a classification problem.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// For classification models, this is how many different categories you're trying to predict.
    /// For binary classification (yes/no), it's 2. For multiclass (e.g., dog/cat/bird), it's 3 or more.
    /// </remarks>
    public int NumberOfClasses { get; set; } = 2;

    /// <summary>
    /// Whether to calculate bootstrap intervals.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Bootstrap intervals use a resampling technique to estimate the variability in your predictions.
    /// They're useful but can be computationally expensive, so you can turn them off to save time.
    /// </remarks>
    public bool CalculateBootstrap { get; set; } = true;

    /// <summary>
    /// The number of bootstrap samples to use (if bootstrap intervals are calculated).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This controls how many random resamples are created when calculating bootstrap intervals.
    /// More samples give more accurate intervals but take longer to calculate.
    /// </remarks>
    public int BootstrapSamples { get; set; } = 1000;

    /// <summary>
    /// Whether to use parallel processing for computationally intensive calculations.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This setting determines whether the system will use multiple CPU cores to speed up calculations.
    /// It can make things faster, especially for large datasets, but uses more system resources.
    /// </remarks>
    public bool UseParallelProcessing { get; set; } = true;
}