using System;
using System.Collections.Generic;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Risk;

/// <summary>
/// Risk-adjusted performance ratios from a periodic return series: Sharpe, Sortino, and Calmar.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet exposes Sharpe on portfolio optimizers and trading agents, but Sortino (downside-only) and
/// Calmar (return-over-max-drawdown) were missing as standalone, series-based utilities. These let any
/// forecaster/strategy backtest be scored on risk-adjusted return without a portfolio or agent object.
/// </para>
/// <para><b>For Beginners:</b> Sharpe divides average return by total volatility; Sortino only counts
/// <i>downside</i> volatility (it doesn't punish upside swings); Calmar divides annualized return by the
/// worst peak-to-trough drawdown. Higher is better for all three.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
/// <remarks>
/// Implements <see cref="IRiskRatioCalculator{T}"/> so it can be injected as the default, swappable
/// risk-scoring strategy into risk models, backtests, and trading agents; the static methods remain
/// available for direct use.
/// </remarks>
public class RiskRatios<T> : IRiskRatioCalculator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Shared stateless default instance for injection as an <see cref="IRiskRatioCalculator{T}"/>.</summary>
    public static IRiskRatioCalculator<T> Default { get; } = new RiskRatios<T>();

    // Explicit IRiskRatioCalculator<T> members delegate to the static implementations.
    T IRiskRatioCalculator<T>.Sharpe(IReadOnlyList<T> returns, double riskFreePerPeriod, int periodsPerYear)
        => Sharpe(returns, riskFreePerPeriod, periodsPerYear);
    T IRiskRatioCalculator<T>.Sortino(IReadOnlyList<T> returns, double riskFreePerPeriod, int periodsPerYear)
        => Sortino(returns, riskFreePerPeriod, periodsPerYear);
    T IRiskRatioCalculator<T>.Calmar(IReadOnlyList<T> returns, int periodsPerYear)
        => Calmar(returns, periodsPerYear);

    /// <summary>Validates the shared external inputs of every ratio.</summary>
    private static void Validate(IReadOnlyList<T> returns, int periodsPerYear)
    {
        if (returns is null)
        {
            throw new ArgumentNullException(nameof(returns));
        }

        if (periodsPerYear <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(periodsPerYear), "periodsPerYear must be > 0.");
        }
    }

    /// <summary>Annualized Sharpe ratio: (mean(excess) / std(excess)) · √periodsPerYear.</summary>
    public static T Sharpe(IReadOnlyList<T> returns, double riskFreePerPeriod = 0.0, int periodsPerYear = 252)
    {
        Validate(returns, periodsPerYear);
        if (returns.Count < 2)
        {
            return NumOps.Zero;
        }

        var rf = NumOps.FromDouble(riskFreePerPeriod);
        var (mean, std) = MeanStd(returns, rf);
        if (!NumOps.GreaterThan(std, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        return NumOps.Multiply(NumOps.Divide(mean, std), NumOps.FromDouble(Math.Sqrt(periodsPerYear)));
    }

    /// <summary>Annualized Sortino ratio: (mean(excess) / downsideDeviation) · √periodsPerYear.</summary>
    public static T Sortino(IReadOnlyList<T> returns, double riskFreePerPeriod = 0.0, int periodsPerYear = 252)
    {
        Validate(returns, periodsPerYear);
        if (returns.Count < 2)
        {
            return NumOps.Zero;
        }

        var rf = NumOps.FromDouble(riskFreePerPeriod);
        var mean = NumOps.Zero;
        var downsideSq = NumOps.Zero;
        var n = NumOps.FromDouble(returns.Count);
        foreach (var r in returns)
        {
            var excess = NumOps.Subtract(r, rf);
            mean = NumOps.Add(mean, excess);
            if (!NumOps.GreaterThan(excess, NumOps.Zero))
            {
                downsideSq = NumOps.Add(downsideSq, NumOps.Multiply(excess, excess));
            }
        }

        mean = NumOps.Divide(mean, n);
        var downsideDev = NumOps.Sqrt(NumOps.Divide(downsideSq, n));
        if (!NumOps.GreaterThan(downsideDev, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        return NumOps.Multiply(NumOps.Divide(mean, downsideDev), NumOps.FromDouble(Math.Sqrt(periodsPerYear)));
    }

    /// <summary>
    /// Calmar ratio: annualized return / maximum drawdown, computed from a periodic return series by
    /// compounding an equity curve. Returns 0 when there is no drawdown.
    /// </summary>
    public static T Calmar(IReadOnlyList<T> returns, int periodsPerYear = 252)
    {
        Validate(returns, periodsPerYear);
        if (returns.Count < 2)
        {
            return NumOps.Zero;
        }

        // Compound an equity curve, tracking the running peak and the worst drawdown.
        var equity = 1.0;
        var peak = 1.0;
        var maxDrawdown = 0.0;
        foreach (var r in returns)
        {
            equity *= 1.0 + NumOps.ToDouble(r);
            if (equity > peak)
            {
                peak = equity;
            }

            var drawdown = peak > 0 ? (peak - equity) / peak : 0.0;
            if (drawdown > maxDrawdown)
            {
                maxDrawdown = drawdown;
            }
        }

        if (maxDrawdown <= 0.0)
        {
            return NumOps.Zero;
        }

        // A losing streak can drive the compounded equity to zero or negative
        // (any period return <= -100%). Math.Pow of a non-positive base by a
        // fractional exponent is NaN, so treat a wiped-out account as a total
        // (-100%) annualized loss rather than emitting NaN.
        var annualizedReturn = equity > 0.0
            ? Math.Pow(equity, (double)periodsPerYear / returns.Count) - 1.0
            : -1.0;
        return NumOps.FromDouble(annualizedReturn / maxDrawdown);
    }

    private static (T Mean, T Std) MeanStd(IReadOnlyList<T> returns, T riskFree)
    {
        var mean = NumOps.Zero;
        var n = NumOps.FromDouble(returns.Count);
        foreach (var r in returns)
        {
            mean = NumOps.Add(mean, NumOps.Subtract(r, riskFree));
        }

        mean = NumOps.Divide(mean, n);

        var sse = NumOps.Zero;
        foreach (var r in returns)
        {
            var d = NumOps.Subtract(NumOps.Subtract(r, riskFree), mean);
            sse = NumOps.Add(sse, NumOps.Multiply(d, d));
        }

        // Sample standard deviation (n-1) — unbiased for a return sample.
        var std = NumOps.Sqrt(NumOps.Divide(sse, NumOps.FromDouble(returns.Count - 1)));
        return (mean, std);
    }
}
