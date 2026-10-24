namespace AiDotNet.Interfaces;

/// <summary>
/// A VAE that encodes video, [batch, channels, frames, height, width], compressing time as well as space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An image VAE shrinks each picture; a video VAE also merges neighbouring frames, so a clip of
/// 49 frames might become 13 latent frames. <see cref="GetLatentFrameCount"/> says how many latent frames a clip gives.
/// </para>
/// </remarks>
public interface IVideoVAEModel<T> : IVAEModel<T>
{
    /// <summary>Gets how many input frames one latent frame covers (after the first, for a causal VAE).</summary>
    int TemporalCompression { get; }

    /// <summary>Gets the number of latent frames a clip of <paramref name="frames"/> frames encodes to.</summary>
    /// <param name="frames">The clip's frame count.</param>
    /// <returns>The latent frame count.</returns>
    int GetLatentFrameCount(int frames);
}
