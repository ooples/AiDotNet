namespace AiDotNet.Enums;

/// <summary>
/// Specifies how frame-level features are aggregated into a single video-level representation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When processing a video, each frame produces its own features.
/// This enum controls how those per-frame features are combined into one representation
/// for the whole video.</para>
/// </remarks>
public enum TemporalAggregationType
{
    /// <summary>
    /// Uses a temporal transformer to learn attention-weighted aggregation across frames.
    /// This is the method used in the VideoCLIP paper (Xu et al., 2021).
    /// </summary>
    TemporalTransformer,

    /// <summary>
    /// Simple average (mean pooling) across all frame features.
    /// </summary>
    MeanPooling,

    /// <summary>
    /// Takes only the last frame's features as the video representation.
    /// </summary>
    LastFrame
}
