using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Optional capability for servable models that support autoregressive text generation.
/// </summary>
/// <remarks>
/// <para>
/// Token-level generation requires a model whose forward pass maps a sequence of token IDs
/// to per-position vocabulary logits (the shape transformer language models produce). Models
/// served as tensor-to-tensor (<see cref="IServableModel{T}"/> backed by a neural network)
/// expose this capability; vector/matrix prediction models do not.
/// </para>
/// <para><b>For Beginners:</b> Ordinary serving runs one prediction and returns a vector.
/// Text generation is different: the model is called repeatedly, each call producing the
/// logits for the next token, which is then appended and fed back in. This interface exposes
/// that single "tokens in, logits out" forward pass so the continuous-batching engine can
/// drive the generation loop.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used by the model.</typeparam>
public interface IServableGenerativeModel<T>
{
    /// <summary>
    /// Gets whether this model supports token-level generation. When false,
    /// <see cref="Forward"/> must not be called.
    /// </summary>
    bool SupportsGeneration { get; }

    /// <summary>
    /// Runs a single token-level forward pass.
    /// </summary>
    /// <param name="inputTokenIds">
    /// Token IDs shaped <c>[batch = 1, sequenceLength]</c> (each element is a token ID cast to <typeparamref name="T"/>).
    /// </param>
    /// <returns>
    /// Logits shaped <c>[1, sequenceLength, vocabularySize]</c> or <c>[1, vocabularySize]</c>.
    /// </returns>
    Tensor<T> Forward(Tensor<T> inputTokenIds);
}
