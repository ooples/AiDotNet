namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Configuration for a single optimization profile (for dynamic shapes).
/// </summary>
public class OptimizationProfileConfig
{
    /// <summary>
    /// Gets or sets the input tensor name.
    /// </summary>
    public string InputName { get; set; } = "input";

    /// <summary>
    /// Gets or sets the minimum shape for this input.
    /// </summary>
    public int[] MinShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the optimal shape for this input (used for optimization).
    /// </summary>
    public int[] OptimalShape { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the maximum shape for this input.
    /// </summary>
    public int[] MaxShape { get; set; } = Array.Empty<int>();
}
