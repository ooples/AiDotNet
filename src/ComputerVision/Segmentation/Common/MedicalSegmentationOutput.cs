namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Output for medical image segmentation with volumetric support and clinical metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Medical segmentation output includes organ/structure labels for
/// each pixel or voxel, along with clinical metrics like Dice scores and volume measurements
/// that doctors use to assess segmentation quality and make clinical decisions.
/// </para>
/// </remarks>
public class MedicalSegmentationOutput<T>
{
    /// <summary>
    /// Per-pixel/voxel class labels. Shape is [H, W] for 2D or [D, H, W] for 3D.
    /// </summary>
    public Tensor<T>? Labels { get; set; }

    /// <summary>
    /// Per-class probability maps. Shape is [numClasses, H, W] or [numClasses, D, H, W].
    /// </summary>
    public Tensor<T>? Probabilities { get; set; }

    /// <summary>
    /// Whether this result is from a 3D volumetric segmentation.
    /// </summary>
    public bool Is3D { get; set; }

    /// <summary>
    /// Voxel spacing in mm [x, y, z] (for 3D results).
    /// </summary>
    public double[]? VoxelSpacing { get; set; }

    /// <summary>
    /// Image orientation matrix (for DICOM compatibility).
    /// </summary>
    public double[]? OrientationMatrix { get; set; }

    /// <summary>
    /// Per-class Dice scores (if ground truth was provided during evaluation).
    /// </summary>
    public Dictionary<int, double> DiceScores { get; set; } = [];

    /// <summary>
    /// Per-class Hausdorff distances in mm (if ground truth was provided).
    /// </summary>
    public Dictionary<int, double> HausdorffDistances { get; set; } = [];

    /// <summary>
    /// Per-class surface Dice scores at specified tolerance (if ground truth was provided).
    /// </summary>
    public Dictionary<int, double> SurfaceDiceScores { get; set; } = [];

    /// <summary>
    /// Segmented anatomical structures with volume and metadata.
    /// </summary>
    public List<AnatomicalStructure> Structures { get; set; } = [];

    /// <summary>
    /// Imaging modality used (CT, MRI_T1, MRI_T2, etc.).
    /// </summary>
    public string? Modality { get; set; }

    /// <summary>
    /// Model confidence summary across all structures.
    /// </summary>
    public double OverallConfidence { get; set; }

    /// <summary>
    /// Inference time.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Uncertainty map [H, W] or [D, H, W] indicating model uncertainty per pixel/voxel.
    /// Higher values indicate less certain predictions.
    /// </summary>
    public Tensor<T>? UncertaintyMap { get; set; }
}

/// <summary>
/// Metadata for a single segmented anatomical structure.
/// </summary>
public class AnatomicalStructure
{
    /// <summary>Class ID.</summary>
    public int ClassId { get; set; }

    /// <summary>Structure name (e.g., "Liver", "Left Kidney", "Spleen").</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Volume in cubic millimeters (for 3D) or area in square millimeters (for 2D).</summary>
    public double VolumeOrAreaMm { get; set; }

    /// <summary>Volume in voxels (for 3D) or area in pixels (for 2D).</summary>
    public int VoxelCount { get; set; }

    /// <summary>Surface area in square millimeters (for 3D).</summary>
    public double? SurfaceAreaMm2 { get; set; }

    /// <summary>Mean confidence across all voxels in this structure.</summary>
    public double MeanConfidence { get; set; }

    /// <summary>Minimum confidence across all voxels (useful for detecting uncertain regions).</summary>
    public double MinConfidence { get; set; }

    /// <summary>Centroid of the structure in image coordinates (x, y) or (x, y, z).</summary>
    public double[] Centroid { get; set; } = [];

    /// <summary>Bounding box [x1, y1, x2, y2] or [x1, y1, z1, x2, y2, z2].</summary>
    public double[] BoundingBox { get; set; } = [];

    /// <summary>Dice score against ground truth (if provided).</summary>
    public double? DiceScore { get; set; }

    /// <summary>Hausdorff distance in mm against ground truth (if provided).</summary>
    public double? HausdorffDistance { get; set; }

    /// <summary>
    /// SNOMED CT code for this structure (for clinical interoperability).
    /// </summary>
    public string? SnomedCode { get; set; }
}
