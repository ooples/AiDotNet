namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for MedNeXt medical segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical
/// Image Segmentation", MICCAI 2023.
/// </para>
/// </remarks>
public enum MedNeXtModelSize
{
    /// <summary>Small variant. Efficient for routine tasks.</summary>
    Small,
    /// <summary>Base variant. Standard configuration.</summary>
    Base,
    /// <summary>Medium variant. Higher accuracy.</summary>
    Medium,
    /// <summary>Large variant. SOTA on CT/MRI benchmarks.</summary>
    Large
}
