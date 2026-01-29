using AiDotNet.Models.Options;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Attention Allocation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class AttentionAllocationOptions<T> : PortfolioOptimizerOptions<T>
{
    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Input sequence length (lookback window).
    /// </summary>
    public int SequenceLength { get; set; } = 60;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;
}
