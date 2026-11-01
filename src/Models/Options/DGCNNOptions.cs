namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DGCNN models.
/// </summary>
public class DGCNNOptions : ModelOptions
{
    /// <summary>Number of output classes.</summary>
    public int NumClasses { get; set; } = 40;
    /// <summary>Number of features per input point (3 for xyz coordinates).</summary>
    public int InputFeatureDim { get; set; } = 3;
    /// <summary>Number of nearest neighbours that define each point's local graph (k in Wang et al. 2019).</summary>
    public int KnnK { get; set; } = 20;
    /// <summary>Output channels of each EdgeConv layer, in order.</summary>
    public int[] EdgeConvChannels { get; set; } = new[] { 64, 64, 128, 256 };
    /// <summary>Hidden widths of the fully connected classifier head.</summary>
    public int[] ClassifierChannels { get; set; } = new[] { 512, 256 };
    /// <summary>Whether dropout is applied during training.</summary>
    public bool UseDropout { get; set; } = true;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.5;
    /// <summary>Learning rate of the optimizer.</summary>
    public double LearningRate { get; set; } = 1e-3;
}
