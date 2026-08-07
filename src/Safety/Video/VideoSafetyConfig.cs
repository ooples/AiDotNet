using System;

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
    /// <remarks>
    /// NO LONGER AFFECTS SAMPLING. MultimodalVideoModerator samples the taxonomy's FIXED budget
    /// (<see cref="AiDotNet.Safety.Video.MultimodalVideoModerator{T}.TaxonomyFrameBudget"/> frames
    /// plus a thumbnail) regardless of video length, which is the whole point of a budget rather
    /// than a rate -- a 20x longer video is not 20x more work. Nothing reads this value, so leaving
    /// it undeprecated would let a caller configure a rate and reasonably expect it to apply.
    /// </remarks>
    [Obsolete("Video sampling is a fixed per-video budget, not a rate; this value is ignored. "
        + "See MultimodalVideoModerator<T>.TaxonomyFrameBudget.")]
    public double? FrameSamplingRate { get; set; }

    /// <summary>Content moderation threshold (0.0-1.0). Default: 0.5.</summary>
    public double? ModerationThreshold { get; set; }

    /// <summary>Whether to use temporal consistency analysis for deepfake detection. Default: true.</summary>
    public bool? TemporalAnalysis { get; set; }

    /// <summary>Maximum frames to analyze per video. Default: 100.</summary>
    public int? MaxFrames { get; set; }

    internal double EffectiveModerationThreshold => ModerationThreshold ?? 0.5;
    internal bool EffectiveTemporalAnalysis => TemporalAnalysis ?? true;
    internal int EffectiveMaxFrames => MaxFrames ?? 100;
}
