using AiDotNet.Models.Options;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for speaker diarization.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SpeakerDiarizer model. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeakerDiarizerOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SpeakerDiarizerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SpeakerDiarizerOptions(SpeakerDiarizerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        WindowDurationSeconds = other.WindowDurationSeconds;
        HopDurationSeconds = other.HopDurationSeconds;
        EmbeddingDimension = other.EmbeddingDimension;
        ClusteringThreshold = other.ClusteringThreshold;
        MinTurnDuration = other.MinTurnDuration;
        MaxSpeakers = other.MaxSpeakers;
        EmbeddingModelPath = other.EmbeddingModelPath;
    }

    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the window duration in seconds.
    /// </summary>
    public double WindowDurationSeconds { get; set; } = 1.5;

    /// <summary>
    /// Gets or sets the hop duration in seconds.
    /// </summary>
    public double HopDurationSeconds { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the clustering threshold (cosine similarity).
    /// </summary>
    public double ClusteringThreshold { get; set; } = 0.65;

    /// <summary>
    /// Gets or sets the minimum turn duration in seconds.
    /// </summary>
    public double MinTurnDuration { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum number of speakers (null for auto).
    /// </summary>
    public int? MaxSpeakers { get; set; }

    /// <summary>
    /// Gets or sets the path to the embedding model.
    /// </summary>
    public string? EmbeddingModelPath { get; set; }
}
