global using AiDotNet.ModelSelection;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface defining methods for automatically selecting and recommending models
/// based on input and output data characteristics.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data structure type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data structure type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface defines methods that can analyze your data and suggest
/// appropriate machine learning models. It's like having an AI assistant that examines your data
/// and recommends which approach might work best for your specific problem.
/// 
/// Implementations of this interface can use different strategies to analyze data and make
/// recommendations. For example, they might consider the data's structure, the type of problem
/// you're trying to solve, and the size of your dataset when suggesting models.
/// </remarks>
public interface IModelSelector<T, TInput, TOutput>
{
    /// <summary>
    /// Analyzes input and output data to automatically select the most appropriate model.
    /// </summary>
    /// <param name="sampleX">A sample of the input data to analyze its structure.</param>
    /// <param name="sampleY">A sample of the output data to analyze its structure.</param>
    /// <returns>The selected model instance.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method examines your data and chooses the best type of
    /// model for your specific problem automatically. It helps you get started without
    /// needing to know exactly which model type to use.
    /// </remarks>
    IFullModel<T, TInput, TOutput> SelectModel(TInput sampleX, TOutput sampleY);

    /// <summary>
    /// Analyzes input and output data and provides a ranked list of recommended models with explanations.
    /// </summary>
    /// <param name="sampleX">A sample of the input data to analyze its structure.</param>
    /// <param name="sampleY">A sample of the output data to analyze its structure.</param>
    /// <returns>A ranked list of model recommendations with explanations.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of choosing a single model for you, this method
    /// gives you recommendations about which models might work well for your data,
    /// along with explanations about each option's strengths and weaknesses.
    /// </remarks>
    List<ModelRecommendation<T, TInput, TOutput>> GetModelRecommendations(TInput sampleX, TOutput sampleY);
}