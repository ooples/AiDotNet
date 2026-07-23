using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Guidance;

/// <summary>
/// Interface for guidance methods that modify noise predictions during diffusion sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Guidance methods control how the diffusion model balances prompt adherence and image
/// quality during generation. Different methods offer different trade-offs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Guidance is like an invisible hand that steers the AI toward
/// your prompt. Standard CFG compares "with prompt" vs "without prompt" predictions.
/// Advanced methods like PAG and SAG use attention manipulation for better results.
/// </para>
/// </remarks>
public interface IGuidanceMethod<T>
{
    /// <summary>
    /// Gets the type of guidance this method implements.
    /// </summary>
    GuidanceType GuidanceType { get; }

    /// <summary>
    /// Applies guidance to combine conditional and unconditional noise predictions.
    /// </summary>
    /// <param name="unconditional">The unconditional noise prediction.</param>
    /// <param name="conditional">The conditional noise prediction.</param>
    /// <param name="scale">The guidance scale.</param>
    /// <param name="timestep">Current diffusion timestep (0 = final, 1 = initial).</param>
    /// <returns>The guided noise prediction.</returns>
    Tensor<T> Apply(Tensor<T> unconditional, Tensor<T> conditional, double scale, double timestep);
}
