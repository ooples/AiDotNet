namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a dataset containing inputs and corresponding outputs for supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A dataset is a collection of examples for training or evaluating
/// machine learning models. Each example has an input (features) and an output (label/target).</para>
///
/// <para><b>Example:</b>
/// In image classification:
/// - Input: Image pixels (e.g., 28×28 array of numbers)
/// - Output: Label (e.g., "cat", "dog", or one-hot vector)
///
/// In regression:
/// - Input: Features (e.g., house size, location, age)
/// - Output: Price (e.g., $250,000)
/// </para>
/// </remarks>
public interface IDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the number of examples in the dataset.
    /// </summary>
    int Count { get; }

    /// <summary>
    /// Gets the input for a specific example.
    /// </summary>
    /// <param name="index">The zero-based index of the example.</param>
    /// <returns>The input data for the specified example.</returns>
    TInput GetInput(int index);

    /// <summary>
    /// Gets the output for a specific example.
    /// </summary>
    /// <param name="index">The zero-based index of the example.</param>
    /// <returns>The output data for the specified example.</returns>
    TOutput GetOutput(int index);

    /// <summary>
    /// Gets all inputs in the dataset.
    /// </summary>
    /// <returns>Enumerable of all input examples.</returns>
    IEnumerable<TInput> GetInputs();

    /// <summary>
    /// Gets all outputs in the dataset.
    /// </summary>
    /// <returns>Enumerable of all output examples.</returns>
    IEnumerable<TOutput> GetOutputs();
}
