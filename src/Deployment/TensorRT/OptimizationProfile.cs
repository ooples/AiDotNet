namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Represents a TensorRT optimization profile for dynamic shapes.
/// </summary>
public class OptimizationProfile
{
    public string? InputName { get; set; }
    public int[]? MinShape { get; set; }
    public int[]? OptimalShape { get; set; }
    public int[]? MaxShape { get; set; }
}
