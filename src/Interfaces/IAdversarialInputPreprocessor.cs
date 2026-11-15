using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// An input transformation applied before inference to remove adversarial perturbations.
/// </summary>
/// <remarks>
/// <para>
/// Implement this to plug a custom denoiser (for example a learned autoencoder, or a high-level representation
/// guided denoiser) into <see cref="AiDotNet.AdversarialRobustness.Defenses.AdversarialTraining{T, TInput, TOutput}"/>
/// through <see cref="AiDotNet.Models.Options.AdversarialDefenseOptions{T}.CustomPreprocessor"/>. A configured
/// preprocessor replaces the built-in <see cref="AiDotNet.Enums.AdversarialPreprocessingMethod"/>.
/// </para>
/// <para><b>For Beginners:</b> This is a "cleaning step" your inputs go through before the model sees them.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type.</typeparam>
public interface IAdversarialInputPreprocessor<T>
{
    /// <summary>
    /// Transforms one flattened input.
    /// </summary>
    /// <param name="input">The flattened input, nominally in [0, 1].</param>
    /// <returns>The transformed input, of the same length.</returns>
    Vector<T> Preprocess(Vector<T> input);
}
