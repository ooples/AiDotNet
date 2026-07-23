namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for medical image segmentation models that handle 2D slices and 3D volumetric data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Medical image segmentation requires special handling for multi-modal imaging data (CT, MRI,
/// X-ray, ultrasound, pathology), 3D volumetric processing, and clinical safety requirements.
/// Models must handle varying image resolutions, anisotropic voxel spacing, and multi-class
/// organ/lesion segmentation with high accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> Medical segmentation helps doctors by automatically outlining organs,
/// tumors, and other structures in medical images.
///
/// Key differences from regular segmentation:
/// - Works with 3D volumes (CT/MRI scans are stacks of 2D slices)
/// - Handles multiple imaging modalities (CT, MRI T1, MRI T2, etc.)
/// - Requires very high accuracy — mistakes can affect patient care
/// - Often needs to handle varying image resolutions and orientations
///
/// Models implementing this interface:
/// - nnU-Net v2 (Nature Methods, self-configuring gold standard)
/// - TransUNet (transformer + U-Net hybrid)
/// - Swin-UNETR (hierarchical transformer for brain MRI)
/// - MedSAM / MedSAM 2 (SAM adapted for medical data)
/// - MedNeXt (ConvNeXt-based, MICCAI 2023)
/// - UniverSeg (few-shot, no fine-tuning needed)
/// - BiomedParse (foundation model, 9 imaging modalities)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MedicalSegmentation")]
public interface IMedicalSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the imaging modalities this model supports (e.g., CT, MRI_T1, MRI_T2, Xray).
    /// </summary>
    IReadOnlyList<string> SupportedModalities { get; }

    /// <summary>
    /// Gets whether this model supports 3D volumetric segmentation.
    /// </summary>
    bool Supports3D { get; }

    /// <summary>
    /// Gets whether this model supports 2D slice-by-slice segmentation.
    /// </summary>
    bool Supports2D { get; }

    /// <summary>
    /// Gets whether this model supports few-shot segmentation (generalizing from few examples).
    /// </summary>
    bool SupportsFewShot { get; }

    /// <summary>
    /// Segments a 2D medical image slice.
    /// </summary>
    /// <param name="slice">2D image tensor [C, H, W] where C is the number of input channels/modalities.</param>
    /// <returns>Medical segmentation result with per-pixel labels and organ/structure metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in a single 2D slice from a CT or MRI scan and get back
    /// labeled regions for each organ or structure the model can identify.
    /// </para>
    /// </remarks>
    MedicalSegmentationResult<T> SegmentSlice(Tensor<T> slice);

    /// <summary>
    /// Segments a 3D medical volume.
    /// </summary>
    /// <param name="volume">3D volume tensor [C, D, H, W] where D is depth (number of slices).</param>
    /// <returns>Volumetric segmentation result with per-voxel labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in a full 3D scan (like a CT or MRI volume) and get back
    /// a 3D labeled volume where every voxel (3D pixel) is assigned to an organ or structure.
    /// This uses the full 3D context for better accuracy than slice-by-slice processing.
    /// </para>
    /// </remarks>
    MedicalSegmentationResult<T> SegmentVolume(Tensor<T> volume);

    /// <summary>
    /// Segments using few-shot examples (for models that support it).
    /// </summary>
    /// <param name="queryImage">The image to segment [C, H, W].</param>
    /// <param name="supportImages">Example images [N, C, H, W].</param>
    /// <param name="supportMasks">Example masks [N, 1, H, W].</param>
    /// <returns>Segmentation result based on the provided examples.</returns>
    MedicalSegmentationResult<T> SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks);
}

/// <summary>
/// Result of medical image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MedicalSegmentationResult<T>
{
    /// <summary>
    /// Per-pixel/voxel class labels. Shape is [H, W] for 2D or [D, H, W] for 3D.
    /// </summary>
    public Tensor<T> Labels { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Per-pixel/voxel class probabilities. Shape is [numClasses, H, W] or [numClasses, D, H, W].
    /// </summary>
    public Tensor<T> Probabilities { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Per-class Dice scores measuring segmentation quality (if ground truth was provided).
    /// </summary>
    public Dictionary<int, double> DiceScores { get; set; } = [];

    /// <summary>
    /// Metadata about each segmented structure (class name, volume, surface area, etc.).
    /// </summary>
    public List<SegmentedStructure> Structures { get; set; } = [];
}

/// <summary>
/// Metadata about a single segmented medical structure.
/// </summary>
public class SegmentedStructure
{
    /// <summary>
    /// Class ID of this structure.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Human-readable name of this structure (e.g., "Left Kidney", "Liver").
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Volume of the structure in voxels (for 3D) or area in pixels (for 2D).
    /// </summary>
    public double VolumeOrArea { get; set; }

    /// <summary>
    /// Mean confidence score across all voxels/pixels in this structure.
    /// </summary>
    public double MeanConfidence { get; set; }
}
