namespace AiDotNet.Audio.AudioGen;

/// <summary>
/// Result of audio generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> AudioGenResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class AudioGenResult<T>
{
    /// <summary>
    /// Gets or sets the generated audio waveform.
    /// </summary>
    public required T[] Audio { get; set; }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the actual duration in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the prompt used for generation.
    /// </summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the random seed used.
    /// </summary>
    public int SeedUsed { get; set; }

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}
