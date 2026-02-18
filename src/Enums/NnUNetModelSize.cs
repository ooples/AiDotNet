namespace AiDotNet.Enums;

/// <summary>
/// Defines the architecture variants for nnU-Net v2 medical segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> nnU-Net automatically configures everything (architecture, preprocessing,
/// training, post-processing) per dataset. These variants control the base architecture.
/// </para>
/// <para>
/// <b>Reference:</b> Isensee et al., "nnU-Net: a self-configuring method for deep learning-based
/// biomedical image segmentation", Nature Methods 2021.
/// </para>
/// </remarks>
public enum NnUNetModelSize
{
    /// <summary>2D U-Net. For 2D slices or anisotropic data.</summary>
    UNet2D,
    /// <summary>3D full-resolution U-Net. For isotropic 3D volumes.</summary>
    UNet3DFull,
    /// <summary>3D low-resolution U-Net + cascade. For large 3D volumes.</summary>
    UNet3DCascade
}
