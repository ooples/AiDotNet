namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for DEVA (Decoupled Video Segmentation).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DEVA separates task-specific image segmentation from temporal propagation,
/// enabling flexible video segmentation with any image segmentation backbone.
/// </para>
/// <para>
/// <b>Reference:</b> Cheng et al., "Tracking Anything with Decoupled Video Segmentation", ICCV 2023.
/// </para>
/// </remarks>
public enum DEVAModelSize
{
    /// <summary>Standard variant with ResNet-50 temporal module.</summary>
    Base,
    /// <summary>Large variant with Swin-L temporal module.</summary>
    Large
}
