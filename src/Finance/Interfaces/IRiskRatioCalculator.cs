using System.Collections.Generic;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// Computes risk-adjusted performance ratios (Sharpe, Sortino, Calmar) from a periodic return series.
/// </summary>
/// <remarks>
/// <para>
/// This is a customization point, not a trainable model. Risk models, backtests, and trading agents
/// depend on this interface to score a strategy's risk-adjusted return and default to the
/// <see cref="AiDotNet.Finance.Risk.RiskRatios{T}"/> implementation, but callers can substitute their
/// own conventions (e.g. a different annualization, a downside threshold other than zero).
/// </para>
/// <para><b>For Beginners:</b> These ratios all divide "how much you earned" by "how much risk you took";
/// higher is better. The default implementation uses the standard textbook formulas.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public interface IRiskRatioCalculator<T>
{
    /// <summary>Annualized Sharpe ratio (mean excess return / total volatility).</summary>
    T Sharpe(IReadOnlyList<T> returns, double riskFreePerPeriod = 0.0, int periodsPerYear = 252);

    /// <summary>Annualized Sortino ratio (mean excess return / downside deviation).</summary>
    T Sortino(IReadOnlyList<T> returns, double riskFreePerPeriod = 0.0, int periodsPerYear = 252);

    /// <summary>Calmar ratio (annualized return / maximum drawdown).</summary>
    T Calmar(IReadOnlyList<T> returns, int periodsPerYear = 252);
}
