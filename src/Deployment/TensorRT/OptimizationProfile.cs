namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Represents a TensorRT optimization profile for dynamic shapes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> OptimizationProfile provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class OptimizationProfile
{
    public string? InputName { get; set; }
    public int[]? MinShape { get; set; }
    public int[]? OptimalShape { get; set; }
    public int[]? MaxShape { get; set; }
}
