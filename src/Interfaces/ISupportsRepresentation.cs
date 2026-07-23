using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// An opt-in capability for models that can expose a dense internal representation (embedding) of an
/// input — typically the penultimate-layer activations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The model's input type.</typeparam>
/// <remarks>
/// <para>
/// Implement this when a model has a meaningful latent representation that downstream logic can use to
/// judge how similar two inputs are <i>to the model</i>. Active-learning diversity uses it: measuring
/// redundancy in representation space is far stronger than in raw input space. Models that do not
/// implement it fall back to input features.
/// </para>
/// <para><b>For Beginners:</b> A neural network turns an input into progressively more abstract features
/// as it flows through layers. The activations just before the output layer are a compact "summary" of
/// the input as the model sees it. Exposing that summary lets other tools reason about which inputs the
/// model considers alike.</para>
/// </remarks>
public interface ISupportsRepresentation<T, in TInput>
{
    /// <summary>
    /// Returns the model's dense internal representation (embedding) of a single input.
    /// </summary>
    /// <param name="input">The input to embed.</param>
    /// <returns>A dense representation vector.</returns>
    Vector<T> GetRepresentation(TInput input);
}
