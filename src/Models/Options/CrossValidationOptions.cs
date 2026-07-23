namespace AiDotNet.Models.Options;

/// <summary>
/// Represents the configuration options for cross-validation in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates various settings that control how cross-validation is performed. It allows users to customize 
/// the validation process, including the number of folds, type of validation, data shuffling, and which metrics to compute.
/// These options provide flexibility in how models are evaluated, enabling users to tailor the validation process to their 
/// specific needs and dataset characteristics.
/// </para>
/// <para><b>For Beginners:</b> This class is like a settings panel for cross-validation.
/// 
/// What it does:
/// - Lets you choose how many parts (folds) to split your data into
/// - Allows you to pick the type of cross-validation you want to use
/// - Gives you the option to shuffle your data randomly
/// - Lets you decide which measurements (metrics) to use when evaluating your model
/// 
/// It's like customizing the rules for a series of tests your model will go through, ensuring you get the most useful 
/// information about how well your model performs.
/// </para>
/// </remarks>
public class CrossValidationOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of folds to use in cross-validation.
    /// </summary>
    /// <remarks>
    /// This determines how many parts the data will be split into. The default is 5 folds.
    /// A higher number of folds generally provides a more thorough evaluation but takes longer to compute.
    /// </remarks>
    public int NumberOfFolds { get; set; } = 5;

    /// <summary>
    /// Gets or sets the type of cross-validation to perform.
    /// </summary>
    /// <remarks>
    /// This specifies the strategy for splitting the data. The default is K-Fold cross-validation.
    /// Different types are suitable for different kinds of data or evaluation needs.
    /// </remarks>
    public CrossValidationType ValidationType { get; set; } = CrossValidationType.KFold;

    /// <summary>
    /// Gets or sets the random seed for data shuffling.
    /// </summary>
    /// <remarks>
    /// If set, this ensures reproducibility of random operations. If null, a random seed will be used.
    /// Setting a specific seed allows you to get the same results across multiple runs.
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets whether to shuffle the data before splitting into folds.
    /// </summary>
    /// <remarks>
    /// Shuffling helps ensure that the order of the data doesn't affect the validation results.
    /// It's generally recommended to keep this true unless you have a specific reason not to shuffle.
    /// </remarks>
    public bool ShuffleData { get; set; } = true;

    /// <summary>
    /// Gets or sets the array of metrics to compute during cross-validation.
    /// </summary>
    /// <remarks>
    /// These metrics will be calculated for each fold and averaged across all folds.
    /// The default metrics are R-squared (R2), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
    /// You can customize this array to include any metrics relevant to your specific problem.
    /// </remarks>
    public MetricType[] MetricsToCompute { get; set; } = { MetricType.R2, MetricType.RMSE, MetricType.MAE };
}
