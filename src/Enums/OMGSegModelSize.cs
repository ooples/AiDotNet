namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for OMG-Seg.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OMG-Seg (One Model that is Great for all Segmentation) handles over
/// 10 different segmentation tasks with a single model and only 70M trainable parameters.
/// It uses task-specific queries to switch between tasks at inference time.
/// </para>
/// <para>
/// <b>Technical Details:</b> Uses a shared transformer backbone with task-specific query sets.
/// Supports image segmentation (semantic, instance, panoptic), video segmentation, open-vocabulary
/// segmentation, interactive segmentation, and more.
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "OMG-Seg: Is One Model Good Enough For All Segmentation?", CVPR 2024.
/// </para>
/// </remarks>
public enum OMGSegModelSize
{
    /// <summary>
    /// Base variant (~70M trainable params). Standard configuration.
    /// </summary>
    Base,

    /// <summary>
    /// Large variant (~120M trainable params). Higher capacity for complex multi-task scenarios.
    /// </summary>
    Large
}
