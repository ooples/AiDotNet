namespace AiDotNet.Interpretability.Interfaces;

/// <summary>
/// Interface for transformer networks that support attention visualization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Transformers use "attention" to focus on different parts
/// of the input when making predictions. This interface provides methods to extract
/// these attention patterns for visualization and analysis.</para>
///
/// <para>For example, in a text classifier, attention might show which words the model
/// focused on most when making its classification decision.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("TransformerNetwork")]
public interface ITransformerNetwork<T, TInput, TOutput>
{
    /// <summary>
    /// Gets attention weights from all transformer layers.
    /// </summary>
    /// <param name="input">The input tensor (typically token embeddings).</param>
    /// <returns>A list of attention weight tensors, one per layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each attention tensor shows how much each position
    /// in the input attends to every other position. Shape is typically
    /// [num_heads, sequence_length, sequence_length].</para>
    ///
    /// <para>High attention values mean the model considered those position pairs
    /// to be related or important. Attention rollout aggregates these across layers
    /// to show overall importance.</para>
    /// </remarks>
    List<Tensor<T>> GetAttentionWeights(Tensor<T> input);
}
