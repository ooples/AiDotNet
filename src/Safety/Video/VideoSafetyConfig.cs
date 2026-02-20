namespace AiDotNet.Safety.Video;

/// <summary>
/// Configuration for video safety detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure video content moderation including
/// frame sampling rate, deepfake detection, and content classification settings.
/// </para>
/// </remarks>
public class VideoSafetyConfig
{
    /// <summary>Frame sampling rate (frames per second to analyze). Default: 1.0.</summary>
    public double? FrameSamplingRate { get; set; }

    /// <summary>Content moderation threshold (0.0-1.0). Default: 0.5.</summary>
    public double? ModerationThreshold { get; set; }

    /// <summary>Whether to use temporal consistency analysis for deepfake detection. Default: true.</summary>
    public bool? TemporalAnalysis { get; set; }

    /// <summary>Maximum frames to analyze per video. Default: 100.</summary>
    public int? MaxFrames { get; set; }

    internal double EffectiveFrameSamplingRate => FrameSamplingRate ?? 1.0;
    internal double EffectiveModerationThreshold => ModerationThreshold ?? 0.5;
    internal bool EffectiveTemporalAnalysis => TemporalAnalysis ?? true;
    internal int EffectiveMaxFrames => MaxFrames ?? 100;
}
