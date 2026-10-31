namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PointNet++ models.
/// </summary>
public class PointNetPlusPlusOptions : ModelOptions
{
    /// <summary>Number of output classes.</summary>
    public int NumClasses { get; set; } = 40;
    /// <summary>Number of features per input point (3 for xyz coordinates).</summary>
    public int InputFeatureDim { get; set; } = 3;
    /// <summary>Number of centroids chosen by farthest-point sampling at each set-abstraction level.</summary>
    public int[] SamplingRates { get; set; } = new[] { 512, 128, 32 };
    /// <summary>Ball-query radius of each set-abstraction level.</summary>
    public double[] SearchRadii { get; set; } = new[] { 0.2, 0.4, 0.8 };
    /// <summary>Number of neighbors grouped around each centroid at each set-abstraction level.</summary>
    public int[] NeighborSamples { get; set; } = new[] { 32, 64, 128 };
    /// <summary>Shared-MLP widths of each set-abstraction level (single-scale grouping).</summary>
    public int[][] MlpDimensions { get; set; } =
    [
        new[] { 64, 64, 128 },
        new[] { 128, 128, 256 },
        new[] { 256, 512, 1024 }
    ];
    /// <summary>Whether each set-abstraction level groups neighbors at several radii (multi-scale grouping) instead of one.</summary>
    public bool UseMultiScaleGrouping { get; set; } = false;
    /// <summary>Ball-query radii for each set-abstraction level when <see cref="UseMultiScaleGrouping"/> is on.</summary>
    public double[][]? MultiScaleRadii { get; set; }
    /// <summary>Shared-MLP widths for each radius of each set-abstraction level when <see cref="UseMultiScaleGrouping"/> is on.</summary>
    public int[][][]? MultiScaleMlpDimensions { get; set; }
    /// <summary>Neighbors sampled at each radius of each set-abstraction level when <see cref="UseMultiScaleGrouping"/> is on.</summary>
    public int[][]? MultiScaleNeighborSamples { get; set; }
    /// <summary>Hidden widths of the fully connected classifier head.</summary>
    public int[] ClassifierChannels { get; set; } = new[] { 512, 256 };
    /// <summary>Whether dropout is applied during training.</summary>
    public bool UseDropout { get; set; } = true;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.3;
    /// <summary>Learning rate of the optimizer.</summary>
    public double LearningRate { get; set; } = 1e-3;
}
