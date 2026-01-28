using AiDotNet.Models.Options;

namespace AiDotNet.Models.Options;

/// <summary>
/// Base options for portfolio optimizers.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class PortfolioOptimizerOptions<T> : FinancialModelOptions<T>
{
    /// <summary>
    /// Number of assets in the portfolio.
    /// </summary>
    public int NumAssets { get; set; } = 10;
}
