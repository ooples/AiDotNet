namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for portfolio optimization models.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PortfolioOptions<T>
{
    /// <summary>
    /// The number of assets in the portfolio.
    /// </summary>
    public int NumAssets { get; set; } = 10;

    /// <summary>
    /// Whether to allow short selling (negative weights).
    /// </summary>
    public bool AllowShortSelling { get; set; } = false;

    /// <summary>
    /// Maximum weight for a single asset (concentration limit).
    /// </summary>
    public double MaxWeight { get; set; } = 1.0;

    /// <summary>
    /// Validates the portfolio options.
    /// </summary>
    public void Validate()
    {
        if (NumAssets < 1)
            throw new ArgumentException("NumAssets must be at least 1.", nameof(NumAssets));
        if (MaxWeight <= 0 || MaxWeight > 1.0)
            throw new ArgumentException("MaxWeight must be between 0 and 1.", nameof(MaxWeight));
    }
}
