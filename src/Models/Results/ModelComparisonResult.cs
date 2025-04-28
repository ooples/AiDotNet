namespace AiDotNet.Models.Results;

/// <summary>
/// Contains the result of comparing a user-selected model with an automatically selected model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class ModelComparisonResult<T, TInput, TOutput>
{
    /// <summary>
    /// The model that was selected by the user.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? YourModel { get; set; }

    /// <summary>
    /// The model that would be automatically selected based on the data.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? AutoSelectedModel { get; set; }

    /// <summary>
    /// A ranked list of recommended models for the data.
    /// </summary>
    public List<ModelRecommendation<T, TInput, TOutput>> Recommendations { get; set; } = [];

    /// <summary>
    /// Detailed insights comparing the two models.
    /// </summary>
    public string ComparisonInsights { get; set; } = string.Empty;
}