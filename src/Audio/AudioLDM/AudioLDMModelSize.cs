namespace AiDotNet.Audio.AudioLDM;

/// <summary>
/// Specifies the size variant of the AudioLDM model.
/// </summary>
/// <remarks>
/// <para>
/// AudioLDM (Audio Latent Diffusion Model) comes in different sizes balancing quality
/// and computational requirements. All variants use latent diffusion in a compressed
/// audio representation space.
/// </para>
/// <para><b>For Beginners:</b> Think of model sizes like different quality levels:
/// <list type="bullet">
/// <item><description><b>Small</b>: Fast generation, good for experimentation (345M parameters)</description></item>
/// <item><description><b>Base</b>: Balanced quality and speed (740M parameters)</description></item>
/// <item><description><b>Large</b>: Best quality, requires more resources (1.5B parameters)</description></item>
/// </list>
/// Start with Small for testing, use Large for final production.
/// </para>
/// </remarks>
public enum AudioLDMModelSize
{
    /// <summary>
    /// Small model variant (345M parameters).
    /// </summary>
    /// <remarks>
    /// - CLAP encoder: 256 hidden dim
    /// - U-Net: 320 base channels, 4 attention heads
    /// - Fast inference, suitable for experimentation
    /// </remarks>
    Small = 0,

    /// <summary>
    /// Base model variant (740M parameters). Default choice.
    /// </summary>
    /// <remarks>
    /// - CLAP encoder: 512 hidden dim
    /// - U-Net: 512 base channels, 8 attention heads
    /// - Good balance of quality and speed
    /// </remarks>
    Base = 1,

    /// <summary>
    /// Large model variant (1.5B parameters).
    /// </summary>
    /// <remarks>
    /// - CLAP encoder: 768 hidden dim
    /// - U-Net: 768 base channels, 8 attention heads
    /// - Highest quality, requires significant GPU memory
    /// </remarks>
    Large = 2,

    /// <summary>
    /// AudioLDM-2 variant with improved architecture.
    /// </summary>
    /// <remarks>
    /// - Uses GPT-2 style text encoder
    /// - Improved CLAP conditioning
    /// - Better audio-text alignment
    /// </remarks>
    V2 = 3,

    /// <summary>
    /// Music-specialized variant.
    /// </summary>
    /// <remarks>
    /// - Fine-tuned on music datasets
    /// - Better instrument separation
    /// - Improved musical coherence
    /// </remarks>
    Music = 4
}
