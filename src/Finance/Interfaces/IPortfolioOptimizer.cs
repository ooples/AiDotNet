using AiDotNet.Tensors;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for portfolio optimization models that determine optimal asset allocations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Portfolio optimizers determine how to allocate capital across assets to achieve
/// investment objectives while managing risk.
/// </para>
/// <para>
/// <b>For Beginners:</b> Portfolio optimization answers: "How should I divide my money?"
///
/// <b>Key Concepts:</b>
/// - <b>Expected Return:</b> How much you expect to earn
/// - <b>Risk (Volatility):</b> How much prices bounce around
/// - <b>Diversification:</b> Don't put all eggs in one basket
///
/// <b>Common Objectives:</b>
/// - Maximize returns for given risk (Sharpe ratio)
/// - Minimize risk for given return (minimum variance)
/// - Risk parity (equal risk contribution)
/// - Maximum diversification
/// </para>
/// </remarks>
public interface IPortfolioOptimizer<T> : IFinancialModel<T>
{
    /// <summary>
    /// Gets the number of assets in the portfolio universe.
    /// </summary>
    int NumAssets { get; }

    /// <summary>
    /// Optimizes portfolio weights given expected returns and covariance.
    /// </summary>
    /// <param name="expectedReturns">Expected returns tensor of shape [assets].</param>
    /// <param name="covariance">Covariance matrix tensor of shape [assets, assets].</param>
    /// <returns>Optimal weights tensor of shape [assets].</returns>
    Tensor<T> OptimizeWeights(Tensor<T> expectedReturns, Tensor<T> covariance);

    /// <summary>
    /// Computes portfolio risk contribution for each asset.
    /// </summary>
    /// <param name="weights">Portfolio weights tensor of shape [assets].</param>
    /// <param name="covariance">Covariance matrix tensor of shape [assets, assets].</param>
    /// <returns>Risk contribution per asset, shape [assets].</returns>
    Tensor<T> ComputeRiskContribution(Tensor<T> weights, Tensor<T> covariance);

    /// <summary>
    /// Calculates expected portfolio return.
    /// </summary>
    /// <param name="weights">Portfolio weights tensor of shape [assets].</param>
    /// <param name="expectedReturns">Expected returns tensor of shape [assets].</param>
    /// <returns>Expected portfolio return.</returns>
    T CalculateExpectedReturn(Tensor<T> weights, Tensor<T> expectedReturns);

    /// <summary>
    /// Calculates portfolio volatility (standard deviation of returns).
    /// </summary>
    /// <param name="weights">Portfolio weights tensor of shape [assets].</param>
    /// <param name="covariance">Covariance matrix tensor of shape [assets, assets].</param>
    /// <returns>Portfolio volatility.</returns>
    T CalculateVolatility(Tensor<T> weights, Tensor<T> covariance);

    /// <summary>
    /// Calculates the Sharpe ratio of the portfolio.
    /// </summary>
    /// <param name="weights">Portfolio weights tensor.</param>
    /// <param name="expectedReturns">Expected returns tensor.</param>
    /// <param name="covariance">Covariance matrix tensor.</param>
    /// <param name="riskFreeRate">Risk-free rate (e.g., Treasury yield).</param>
    /// <returns>Sharpe ratio (excess return per unit of risk).</returns>
    T CalculateSharpeRatio(Tensor<T> weights, Tensor<T> expectedReturns, Tensor<T> covariance, T riskFreeRate);

    /// <summary>
    /// Gets portfolio-specific performance metrics.
    /// </summary>
    /// <returns>Dictionary containing portfolio optimization metrics.</returns>
    Dictionary<string, T> GetPortfolioMetrics();
}
