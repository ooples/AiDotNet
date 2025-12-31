namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for speaker diarization.
/// </summary>
public class SpeakerDiarizerOptions
{
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
