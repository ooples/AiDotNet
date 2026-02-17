using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Safety;
using AiDotNet.Safety.Image;

namespace AiDotNet.Safety.Video;

/// <summary>
/// Video content moderator that samples frames and applies image safety classification.
/// </summary>
/// <remarks>
/// <para>
/// This module samples video frames at a configurable rate and applies image-level safety
/// checks (NSFW, violence) to each sampled frame. This is the standard approach used by
/// production video moderation systems: rather than processing every frame, a subset is
/// selected and each is independently classified.
/// </para>
/// <para>
/// <b>For Beginners:</b> Videos are just sequences of images (frames). This module picks
/// some of those frames at regular intervals and checks each one for harmful content.
/// If any sampled frame is flagged, the entire video is flagged.
/// </para>
/// <para>
/// <b>Sampling strategy:</b>
/// - At 1 FPS sampling on a 30 FPS video, only ~3% of frames are analyzed
/// - Higher sampling rates improve detection but increase processing time
/// - Key-frame extraction could further optimize (not yet implemented)
/// </para>
/// <para>
/// <b>References:</b>
/// - YouTube content moderation: frame-level classification pipeline (2024)
/// - Efficient video understanding via sampling strategies (CVPR 2024)
/// - Video content moderation at scale (Meta, 2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FrameSamplingVideoModerator<T> : VideoSafetyModuleBase<T>
{
    private readonly CLIPImageSafetyClassifier<T> _imageClassifier;
    private readonly double _samplingRate;

    /// <inheritdoc />
    public override string ModuleName => "FrameSamplingVideoModerator";

    /// <summary>
    /// Initializes a new frame-sampling video moderator.
    /// </summary>
    /// <param name="samplingRate">
    /// Number of frames to sample per second. Default: 1.0 (one frame per second).
    /// Higher values improve detection but increase processing time.
    /// </param>
    /// <param name="nsfwThreshold">NSFW detection threshold for the image classifier.</param>
    /// <param name="violenceThreshold">Violence detection threshold for the image classifier.</param>
    public FrameSamplingVideoModerator(
        double samplingRate = 1.0,
        double nsfwThreshold = 0.8,
        double violenceThreshold = 0.75)
        : base(30.0)
    {
        if (samplingRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(samplingRate),
                "Sampling rate must be positive.");
        }

        _samplingRate = samplingRate;
        _imageClassifier = new CLIPImageSafetyClassifier<T>(nsfwThreshold, violenceThreshold);
    }

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        var findings = new List<SafetyFinding>();

        if (frames.Count == 0 || frameRate <= 0)
        {
            return findings;
        }

        // Compute sampling interval: how many frames to skip between samples
        int frameInterval = Math.Max(1, (int)(frameRate / _samplingRate));
        int framesAnalyzed = 0;

        for (int i = 0; i < frames.Count; i += frameInterval)
        {
            var frameFindings = _imageClassifier.EvaluateImage(frames[i]);
            framesAnalyzed++;

            foreach (var finding in frameFindings)
            {
                double timestamp = i / frameRate;

                findings.Add(new SafetyFinding
                {
                    Category = finding.Category,
                    Severity = finding.Severity,
                    Confidence = finding.Confidence,
                    Description = $"Video frame at {timestamp:F1}s flagged: {finding.Description}",
                    RecommendedAction = finding.RecommendedAction,
                    SourceModule = ModuleName,
                    SpanStart = (int)(timestamp * 1000), // milliseconds
                    SpanEnd = (int)((timestamp + 1.0 / frameRate) * 1000)
                });
            }
        }

        return findings;
    }
}
