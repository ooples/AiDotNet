namespace AiDotNet.Audio.StableAudio;

/// <summary>
/// Specifies the size variant of the Stable Audio model.
/// </summary>
/// <remarks>
/// <para>
/// Stable Audio is a latent diffusion model by Stability AI for high-quality audio generation.
/// It uses a Diffusion Transformer (DiT) architecture instead of U-Net for improved quality
/// and supports variable-length audio generation.
/// </para>
/// <para><b>For Beginners:</b> Think of model sizes like different quality levels:
/// <list type="bullet">
/// <item><description><b>Small</b>: Fast generation, good for experimentation (300M parameters)</description></item>
/// <item><description><b>Base</b>: Balanced quality and speed (800M parameters)</description></item>
/// <item><description><b>Large</b>: Best quality, requires more resources (1.5B parameters)</description></item>
/// <item><description><b>Open</b>: Open-source variant with permissive license</description></item>
/// </list>
/// Start with Small for testing, use Large for production.
/// </para>
/// </remarks>
public enum StableAudioModelSize
{
    /// <summary>
    /// Small model variant (300M parameters).
    /// </summary>
    /// <remarks>
    /// - T5 encoder: 256 hidden dim
    /// - DiT: 512 hidden dim, 12 blocks
    /// - Fast inference, suitable for experimentation
    /// </remarks>
    Small = 0,

    /// <summary>
    /// Base model variant (800M parameters). Default choice.
    /// </summary>
    /// <remarks>
    /// - T5 encoder: 768 hidden dim
    /// - DiT: 1024 hidden dim, 24 blocks
    /// - Good balance of quality and speed
    /// </remarks>
    Base = 1,

    /// <summary>
    /// Large model variant (1.5B parameters).
    /// </summary>
    /// <remarks>
    /// - T5 encoder: 1024 hidden dim
    /// - DiT: 1536 hidden dim, 32 blocks
    /// - Highest quality, requires significant GPU memory
    /// </remarks>
    Large = 2,

    /// <summary>
    /// Stable Audio Open variant.
    /// </summary>
    /// <remarks>
    /// - Open-source model with permissive license
    /// - Optimized for music generation
    /// - Based on Base architecture
    /// </remarks>
    Open = 3,

    /// <summary>
    /// Stable Audio 2.0 variant.
    /// </summary>
    /// <remarks>
    /// - Improved architecture with better coherence
    /// - Extended duration support (up to 3 minutes)
    /// - Enhanced stereo output
    /// </remarks>
    V2 = 4
}
