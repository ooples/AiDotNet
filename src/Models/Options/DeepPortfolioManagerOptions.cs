namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the DeepPortfolioManager model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These options control the deep portfolio manager,
/// which uses neural networks to directly predict optimal portfolio weights.
/// </para>
/// </remarks>
public class DeepPortfolioManagerOptions<T> : PortfolioOptimizerOptions<T>
{
    /// <summary>
    /// Whether to allow short selling (negative weights).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Short selling means betting on prices going down.
    /// If false, all portfolio weights must be non-negative.
    /// </para>
    /// </remarks>
    public bool AllowShortSelling { get; set; } = false;

    /// <summary>
    /// Maximum weight for a single asset (concentration limit).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This prevents putting too much money into one stock.
    /// 1.0 means up to 100% in one asset; 0.5 means max 50% per asset.
    /// </para>
    /// </remarks>
    public double MaxWeight { get; set; } = 1.0;

    /// <summary>
    /// Validates the options.
    /// </summary>
    public void Validate()
    {
        if (NumAssets < 1)
            throw new ArgumentException("NumAssets must be at least 1.", nameof(NumAssets));
        if (MaxWeight <= 0 || MaxWeight > 1.0)
            throw new ArgumentException("MaxWeight must be between 0 and 1.", nameof(MaxWeight));
    }
}
