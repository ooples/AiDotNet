using AiDotNet.Models.Options;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Neural Black-Litterman.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class BlackLittermanNeuralOptions<T> : PortfolioOptimizerOptions<T>
{
    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    public double DropoutRate { get; set; } = 0.1;
}
