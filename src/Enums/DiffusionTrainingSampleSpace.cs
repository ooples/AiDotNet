namespace AiDotNet.Enums;

/// <summary>
/// What a latent diffusion model's training sample is: an image its VAE encodes, or a latent already encoded.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A latent diffusion model learns to denoise a compressed "latent" version of an image,
/// not the pixels. If you train it on images, it compresses each one first; if you already compressed them
/// (to save time, say), tell it so and it uses them as they are.
/// </para>
/// </remarks>
public enum DiffusionTrainingSampleSpace
{
    /// <summary>
    /// The sample is an image <c>[batch, VAE.InputChannels, height, width]</c> (or unbatched). It is encoded to
    /// the scaled latent z = E(x) before noise is added, as latent diffusion trains (Rombach et al. 2022,
    /// section 3.3).
    /// </summary>
    Image,

    /// <summary>
    /// The sample is already a scaled latent <c>[batch, LatentChannels, height / factor, width / factor]</c> and
    /// is noised as it is.
    /// </summary>
    Latent
}
