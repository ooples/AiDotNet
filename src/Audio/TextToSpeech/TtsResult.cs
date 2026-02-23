namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Result of text-to-speech synthesis.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> TtsResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class TtsResult<T>
{
    /// <summary>
    /// Gets or sets the generated audio waveform.
    /// </summary>
    public required T[] Audio { get; set; }

    /// <summary>
    /// Gets or sets the sample rate of the audio.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the duration in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; set; }
}
