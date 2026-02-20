namespace AiDotNet.Safety.Video;

/// <summary>
/// Detailed result from video safety evaluation with per-frame annotations.
/// </summary>
public class VideoSafetyResult
{
    /// <summary>Whether the video is safe overall.</summary>
    public bool IsSafe { get; init; }

    /// <summary>Number of frames analyzed.</summary>
    public int FramesAnalyzed { get; init; }

    /// <summary>Number of frames flagged as unsafe.</summary>
    public int UnsafeFrames { get; init; }

    /// <summary>Temporal consistency score (0.0 = inconsistent/deepfake, 1.0 = natural).</summary>
    public double TemporalConsistency { get; init; }

    /// <summary>Per-frame safety scores (index → score).</summary>
    public IReadOnlyList<double> FrameScores { get; init; } = Array.Empty<double>();
}
