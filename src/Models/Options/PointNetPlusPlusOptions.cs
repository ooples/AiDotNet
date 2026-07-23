namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PointNet++ models.
/// </summary>
public class PointNetPlusPlusOptions : ModelOptions
{
    public int NumClasses { get; set; } = 40;
    public int InputFeatureDim { get; set; } = 3;
    public int[] SamplingRates { get; set; } = new[] { 512, 128, 32 };
    public double[] SearchRadii { get; set; } = new[] { 0.2, 0.4, 0.8 };
    public int[] NeighborSamples { get; set; } = new[] { 32, 64, 128 };
    public int[][] MlpDimensions { get; set; } =
    [
        new[] { 64, 64, 128 },
        new[] { 128, 128, 256 },
        new[] { 256, 512, 1024 }
    ];
    public bool UseMultiScaleGrouping { get; set; } = false;
    public double[][]? MultiScaleRadii { get; set; }
    public int[][][]? MultiScaleMlpDimensions { get; set; }
    public int[][]? MultiScaleNeighborSamples { get; set; }
    public int[] ClassifierChannels { get; set; } = new[] { 512, 256 };
    public bool UseDropout { get; set; } = true;
    public double DropoutRate { get; set; } = 0.3;
    public double LearningRate { get; set; } = 1e-3;
}
