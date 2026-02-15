namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for sequence loss functions that operate on variable-length sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("SequenceLossFunction")]
public interface ISequenceLossFunction<T>
{
    /// <summary>
    /// Calculates the loss for sequence data.
    /// </summary>
    /// <param name="logProbs">The log probabilities for each time step and class.</param>
    /// <param name="targets">The target sequence indices.</param>
    /// <param name="inputLengths">The actual lengths of input sequences.</param>
    /// <param name="targetLengths">The actual lengths of target sequences.</param>
    /// <returns>The calculated loss value.</returns>
    T CalculateLoss(Tensor<T> logProbs, int[][] targets, int[] inputLengths, int[] targetLengths);

    /// <summary>
    /// Calculates the gradient of the loss with respect to the inputs.
    /// </summary>
    /// <param name="logProbs">The log probabilities for each time step and class.</param>
    /// <param name="targets">The target sequence indices.</param>
    /// <param name="inputLengths">The actual lengths of input sequences.</param>
    /// <param name="targetLengths">The actual lengths of target sequences.</param>
    /// <returns>The gradient tensor with same shape as inputs.</returns>
    Tensor<T> CalculateGradient(Tensor<T> logProbs, int[][] targets, int[] inputLengths, int[] targetLengths);
}
