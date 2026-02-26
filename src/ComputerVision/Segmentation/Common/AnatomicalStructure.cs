namespace AiDotNet.ComputerVision.Segmentation.Common;

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
