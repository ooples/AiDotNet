using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Padding;

/// <summary>
/// Interface for padding strategies that handle variable-length sequences in batches.
/// </summary>
public interface IPaddingStrategy
{
    /// <summary>
    /// Gets the name of the padding strategy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Pads a batch of vectors to a uniform length.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="vectors">Array of vectors with potentially different lengths</param>
    /// <param name="attentionMask">Output attention mask indicating which positions are padding (optional)</param>
    /// <returns>A matrix where each row is a padded vector</returns>
    Matrix<T> PadBatch<T>(Vector<T>[] vectors, out Matrix<T>? attentionMask);

    /// <summary>
    /// Unpads the output vectors based on the attention mask.
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    /// <param name="paddedMatrix">The padded output matrix</param>
    /// <param name="originalLengths">Array of original lengths before padding</param>
    /// <returns>Array of unpadded vectors</returns>
    Vector<T>[] UnpadBatch<T>(Matrix<T> paddedMatrix, int[] originalLengths);
}
