namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for OneFormer.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OneFormer is a universal segmentation model that handles all three
/// segmentation tasks (semantic, instance, panoptic) with a single model trained only on
/// panoptic data. It uses text conditioning to switch between tasks at inference time.
/// </para>
/// <para>
/// <b>Technical Details:</b> Built on top of Mask2Former with a text encoder that conditions
/// the segmentation on a task description. Uses Swin or DiNAT (Dilated Neighborhood
/// Attention Transformer) backbones.
/// </para>
/// <para>
/// <b>Reference:</b> Jain et al., "OneFormer: One Transformer to Rule Universal Image
/// Segmentation", CVPR 2023.
/// </para>
/// </remarks>
public enum OneFormerModelSize
{
    /// <summary>
    /// Swin-L backbone (219M params). Strong Swin-based variant.
    /// </summary>
    SwinLarge,

    /// <summary>
    /// DiNAT-L backbone (223M params). Best accuracy using dilated attention.
    /// </summary>
    DiNATLarge
}
