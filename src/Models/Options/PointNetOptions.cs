namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PointNet models.
/// </summary>
public class PointNetOptions : ModelOptions
{
    /// <summary>Number of output classes.</summary>
    public int NumClasses { get; set; } = 40;
    /// <summary>Number of features per input point (3 for xyz coordinates).</summary>
    public int InputFeatureDim { get; set; } = 3;
    /// <summary>Dimension of the input transform's square matrix (3 for xyz).</summary>
    public int InputTransformDim { get; set; } = 3;
    /// <summary>Whether to align the input points with a learned input-transform network.</summary>
    public bool UseInputTransform { get; set; } = true;
    /// <summary>Whether to align per-point features with a learned feature-transform network.</summary>
    public bool UseFeatureTransform { get; set; } = true;
    /// <summary>Widths of the shared MLP applied after the input transform.</summary>
    public int[] InputMlpChannels { get; set; } = new[] { 64, 64 };
    /// <summary>Widths of the shared MLP applied after the feature transform, before global max pooling.</summary>
    public int[] FeatureMlpChannels { get; set; } = new[] { 64, 128, 1024 };
    /// <summary>Hidden widths of the fully connected classifier head.</summary>
    public int[] ClassifierChannels { get; set; } = new[] { 512, 256 };
    /// <summary>Shared-MLP widths of the input-transform network.</summary>
    public int[] InputTransformMlpChannels { get; set; } = new[] { 64, 128, 1024 };
    /// <summary>Fully connected widths of the input-transform network (the 3x3 T-Net).</summary>
    public int[] InputTransformFcChannels { get; set; } = new[] { 512, 256 };
    /// <summary>Shared-MLP widths of the feature-transform network.</summary>
    public int[] FeatureTransformMlpChannels { get; set; } = new[] { 64, 128, 1024 };
    /// <summary>Fully connected widths of the feature-transform network (the 64x64 T-Net).</summary>
    public int[] FeatureTransformFcChannels { get; set; } = new[] { 512, 256 };
    /// <summary>Whether dropout is applied during training.</summary>
    public bool UseDropout { get; set; } = true;
    /// <summary>Dropout probability applied during training.</summary>
    public double DropoutRate { get; set; } = 0.3;
    /// <summary>Learning rate of the optimizer.</summary>
    public double LearningRate { get; set; } = 1e-3;
}
