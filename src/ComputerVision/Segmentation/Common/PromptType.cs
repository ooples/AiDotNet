namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Type of segmentation prompt.
/// </summary>
public enum PromptType
{
    /// <summary>No prompt (automatic mode).</summary>
    None,
    /// <summary>Point clicks on foreground/background.</summary>
    Point,
    /// <summary>Bounding box around the target.</summary>
    Box,
    /// <summary>Rough mask or scribble.</summary>
    Mask,
    /// <summary>Natural language description.</summary>
    Text,
    /// <summary>Audio signal for multi-modal models.</summary>
    Audio,
    /// <summary>Reference image+mask for in-context learning.</summary>
    Reference
}
