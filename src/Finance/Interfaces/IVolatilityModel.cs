using AiDotNet.Tensors;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Interface for volatility models that forecast price variability.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Volatility models predict how much prices will fluctuate, essential for
/// option pricing, risk management, and portfolio construction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Volatility measures how "bouncy" prices are.
///
/// <b>Why Volatility Matters:</b>
/// - Options pricing: Higher volatility = more expensive options
/// - Risk management: Volatile assets need more capital buffer
/// - Portfolio construction: Helps balance risk across assets
///
/// <b>Types of Volatility:</b>
/// - <b>Historical:</b> What happened in the past
/// - <b>Implied:</b> What the market expects (from option prices)
/// - <b>Realized:</b> What actually occurred over a period
/// - <b>Forecast:</b> Our prediction for the future
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("VolatilityModel")]
public interface IVolatilityModel<T> : IFinancialModel<T>
{
    /// <summary>
    /// Forecasts future volatility.
    /// </summary>
    /// <param name="historicalReturns">Historical returns tensor of shape [samples, assets] or [batch, sequence, assets].</param>
    /// <param name="horizon">Number of periods to forecast.</param>
    /// <returns>Volatility forecasts tensor of shape [horizon, assets].</returns>
    Tensor<T> ForecastVolatility(Tensor<T> historicalReturns, int horizon);

    /// <summary>
    /// Estimates the current volatility state.
    /// </summary>
    /// <param name="recentReturns">Recent returns tensor of shape [sequence, assets].</param>
    /// <returns>Current volatility estimate tensor of shape [assets].</returns>
    Tensor<T> EstimateCurrentVolatility(Tensor<T> recentReturns);

    /// <summary>
    /// Computes the correlation matrix from returns data.
    /// </summary>
    /// <param name="returns">Returns tensor of shape [samples, assets].</param>
    /// <returns>Correlation matrix tensor of shape [assets, assets].</returns>
    Tensor<T> ComputeCorrelationMatrix(Tensor<T> returns);

    /// <summary>
    /// Computes the covariance matrix from returns data.
    /// </summary>
    /// <param name="returns">Returns tensor of shape [samples, assets].</param>
    /// <returns>Covariance matrix tensor of shape [assets, assets].</returns>
    Tensor<T> ComputeCovarianceMatrix(Tensor<T> returns);

    /// <summary>
    /// Calculates realized volatility from high-frequency data.
    /// </summary>
    /// <param name="highFrequencyReturns">Intraday returns tensor of shape [periods, assets].</param>
    /// <returns>Realized volatility tensor of shape [assets].</returns>
    Tensor<T> CalculateRealizedVolatility(Tensor<T> highFrequencyReturns);

    /// <summary>
    /// Gets volatility-specific metrics.
    /// </summary>
    /// <returns>Dictionary containing volatility model metrics.</returns>
    Dictionary<string, T> GetVolatilityMetrics();
}
