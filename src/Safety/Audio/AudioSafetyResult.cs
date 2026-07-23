namespace AiDotNet.Safety.Audio;

/// <summary>
/// Detailed result from audio safety evaluation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AudioSafetyResult provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class AudioSafetyResult
{
    /// <summary>Whether the audio is safe overall.</summary>
    public bool IsSafe { get; init; }

    /// <summary>Deepfake probability score (0.0 = authentic, 1.0 = fake).</summary>
    public double DeepfakeScore { get; init; }

    /// <summary>Toxicity score (0.0 = safe, 1.0 = maximally toxic).</summary>
    public double ToxicityScore { get; init; }

    /// <summary>Whether a watermark was detected.</summary>
    public bool WatermarkDetected { get; init; }

    /// <summary>Detected sample rate of the audio.</summary>
    public int SampleRate { get; init; }
}
