namespace AiDotNet.Audio.MusicGen;

/// <summary>
/// Specifies the size variant of the MusicGen model.
/// </summary>
/// <remarks>
/// <para>
/// MusicGen comes in different sizes balancing quality and computational requirements.
/// Larger models produce higher quality music but require more memory and compute.
/// </para>
/// <para><b>For Beginners:</b> Think of model sizes like different quality levels:
/// <list type="bullet">
/// <item><description><b>Small</b>: Fast generation, good for experimentation (300M parameters)</description></item>
/// <item><description><b>Medium</b>: Balanced quality and speed (1.5B parameters)</description></item>
/// <item><description><b>Large</b>: Best quality, requires more resources (3.3B parameters)</description></item>
/// </list>
/// Start with Small for testing, use Large for final production.
/// </para>
/// </remarks>
public enum MusicGenModelSize
{
    /// <summary>
    /// Small model variant (300M parameters).
    /// </summary>
    /// <remarks>
    /// - Text encoder: 256 hidden dim
    /// - LM: 1024 hidden dim, 24 layers, 16 heads
    /// - Fast inference, suitable for real-time applications
    /// </remarks>
    Small = 0,

    /// <summary>
    /// Medium model variant (1.5B parameters). Default choice.
    /// </summary>
    /// <remarks>
    /// - Text encoder: 768 hidden dim
    /// - LM: 1536 hidden dim, 24 layers, 16 heads
    /// - Good balance of quality and speed
    /// </remarks>
    Medium = 1,

    /// <summary>
    /// Large model variant (3.3B parameters).
    /// </summary>
    /// <remarks>
    /// - Text encoder: 1024 hidden dim
    /// - LM: 2048 hidden dim, 48 layers, 16 heads
    /// - Highest quality, requires significant GPU memory
    /// </remarks>
    Large = 2,

    /// <summary>
    /// Melody model variant (1.5B parameters).
    /// </summary>
    /// <remarks>
    /// - Same architecture as Medium
    /// - Additionally conditioned on melody input
    /// - Can generate music that follows a given melody
    /// </remarks>
    Melody = 3,

    /// <summary>
    /// Stereo model variant (1.5B parameters).
    /// </summary>
    /// <remarks>
    /// - Same architecture as Medium
    /// - Generates stereo audio output
    /// - Uses additional codebook for left/right channel separation
    /// </remarks>
    Stereo = 4
}
