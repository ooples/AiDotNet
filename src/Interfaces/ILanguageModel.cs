using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Marker interface for models with a language-model-shaped inference contract: token-tensor
/// input <c>[B, T]</c> → next-token-logits output <c>[B, T, V]</c>. Implemented by
/// <see cref="NeuralNetworks.Transformer{T}"/> and any custom transformer / decoder-only /
/// encoder-decoder model that follows the same shape.
/// </summary>
/// <remarks>
/// <para>
/// Introduced to replace the Tensor-shape heuristic in
/// <c>AiModelResultTransformerExtensions</c>: extension methods now gate on this interface
/// so custom transformer subclasses can plug in without needing to derive from the concrete
/// <see cref="NeuralNetworks.Transformer{T}"/> type. Reference impls (HF <c>generate</c>,
/// llama.cpp) all encode this contract via ad-hoc type checks; expressing it as an interface
/// is beyond industry standard.
/// </para>
/// </remarks>
public interface ILanguageModel<T>
{
    /// <summary>
    /// Runs one forward pass through the LM and returns next-token logits.
    /// </summary>
    /// <param name="tokenIds">Token ID tensor [B, T].</param>
    /// <returns>Logits tensor [B, T, V].</returns>
    Tensor<T> ForwardLogits(Tensor<T> tokenIds);
}
