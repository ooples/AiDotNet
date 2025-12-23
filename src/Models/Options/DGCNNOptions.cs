namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DGCNN models.
/// </summary>
public class DGCNNOptions : ModelOptions
{
    public int NumClasses { get; set; } = 40;
    public int InputFeatureDim { get; set; } = 3;
    public int KnnK { get; set; } = 20;
    public int[] EdgeConvChannels { get; set; } = new[] { 64, 64, 128, 256 };
    public int[] ClassifierChannels { get; set; } = new[] { 512, 256 };
    public bool UseDropout { get; set; } = true;
    public double DropoutRate { get; set; } = 0.5;
    public double LearningRate { get; set; } = 1e-3;
}
