namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for SAM 2.1 (Segment Anything Model 2.1).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM 2.1 is an updated version of SAM 2 with refined training recipes
/// that improve segmentation accuracy. It uses the same Hiera backbone architecture as SAM 2
/// but with better-tuned checkpoints.
/// </para>
/// <para>
/// <b>Technical Details:</b> Same Hiera backbone as SAM 2 with improved training procedures.
/// Supports both image and video segmentation with memory attention for temporal consistency.
/// </para>
/// <para>
/// <b>Reference:</b> Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.
/// </para>
/// </remarks>
public enum SAM21ModelSize
{
    /// <summary>
    /// Tiny variant (39M params). Fastest inference, suitable for edge devices.
    /// </summary>
    Tiny,

    /// <summary>
    /// Small variant (46M params). Good balance of speed and accuracy.
    /// </summary>
    Small,

    /// <summary>
    /// Base Plus variant (81M params). Improved accuracy over Small.
    /// </summary>
    BasePlus,

    /// <summary>
    /// Large variant (224M params). Maximum accuracy — default for quality-critical applications.
    /// </summary>
    Large
}
