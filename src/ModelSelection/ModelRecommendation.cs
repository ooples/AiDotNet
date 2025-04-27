namespace AiDotNet.ModelSelection;

/// <summary>
/// Represents a model recommendation with explanation and confidence score.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This class packages information about a recommended model,
/// including why it might be appropriate for your data and how confident the system is
/// about this recommendation. It also includes a convenient factory method to create
/// an instance of the recommended model.
/// </remarks>
public class ModelRecommendation<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the name of the recommended model.
    /// </summary>
    public string ModelName { get; }

    /// <summary>
    /// Gets an explanation of why this model is recommended.
    /// </summary>
    public string Explanation { get; }

    /// <summary>
    /// Gets a confidence score for this recommendation (0-100).
    /// </summary>
    public int ConfidenceScore { get; }

    /// <summary>
    /// Gets a factory function that creates an instance of the recommended model.
    /// </summary>
    public Func<IFullModel<T, TInput, TOutput>> ModelFactory { get; }

    /// <summary>
    /// Creates a new model recommendation.
    /// </summary>
    /// <param name="modelName">The name of the recommended model.</param>
    /// <param name="explanation">An explanation of why this model is recommended.</param>
    /// <param name="confidenceScore">A confidence score for this recommendation (0-100).</param>
    /// <param name="modelFactory">A factory function that creates an instance of the recommended model.</param>
    public ModelRecommendation(string modelName, string explanation, int confidenceScore, Func<IFullModel<T, TInput, TOutput>> modelFactory)
    {
        ModelName = modelName;
        Explanation = explanation;
        ConfidenceScore = confidenceScore;
        ModelFactory = modelFactory;
    }
}