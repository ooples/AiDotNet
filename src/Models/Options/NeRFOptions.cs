namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for NeRF models.
/// </summary>
public class NeRFOptions : ModelOptions
{
    public int PositionEncodingLevels { get; set; } = 10;
    public int DirectionEncodingLevels { get; set; } = 4;
    public int HiddenDim { get; set; } = 256;
    public int NumLayers { get; set; } = 8;
    public int ColorHiddenDim { get; set; } = 128;
    public int ColorNumLayers { get; set; } = 1;
    public bool UseHierarchicalSampling { get; set; } = true;
    public int RenderSamples { get; set; } = 64;
    public int HierarchicalSamples { get; set; } = 128;
    public double RenderNearBound { get; set; } = 2.0;
    public double RenderFarBound { get; set; } = 6.0;
    public double LearningRate { get; set; } = 5e-4;
}
