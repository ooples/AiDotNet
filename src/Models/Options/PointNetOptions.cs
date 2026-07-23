namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for PointNet models.
/// </summary>
public class PointNetOptions : ModelOptions
{
    public int NumClasses { get; set; } = 40;
    public int InputFeatureDim { get; set; } = 3;
    public int InputTransformDim { get; set; } = 3;
    public bool UseInputTransform { get; set; } = true;
    public bool UseFeatureTransform { get; set; } = true;
    public int[] InputMlpChannels { get; set; } = new[] { 64, 64 };
    public int[] FeatureMlpChannels { get; set; } = new[] { 64, 128, 1024 };
    public int[] ClassifierChannels { get; set; } = new[] { 512, 256 };
    public int[] InputTransformMlpChannels { get; set; } = new[] { 64, 128, 1024 };
    public int[] InputTransformFcChannels { get; set; } = new[] { 512, 256 };
    public int[] FeatureTransformMlpChannels { get; set; } = new[] { 64, 128, 1024 };
    public int[] FeatureTransformFcChannels { get; set; } = new[] { 512, 256 };
    public bool UseDropout { get; set; } = true;
    public double DropoutRate { get; set; } = 0.3;
    public double LearningRate { get; set; } = 1e-3;
}
