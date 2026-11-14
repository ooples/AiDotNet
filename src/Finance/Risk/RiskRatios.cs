using System;
using System.Collections.Generic;
using System.Linq;
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
        var engine = AiDotNetEngine.Current;
        var excess = engine.TensorSubtractScalar(ToTensor(returns), rf);
        var mean = MeanOf(excess, returns.Count);

        // Downside: the excesses that are not positive.
        var downside = engine.TensorWhere(
            engine.TensorGreaterThan(excess, NumOps.Zero), engine.TensorZerosLike(excess), excess);
        var downsideDev = RootMeanSquare(downside, returns.Count);
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

        // The equity curve is compounded wide: Half spaces values near 1 by 2^-10, so 1 + 1e-4 would round
        // back to 1 and every small return would vanish from the curve.
        var engine = AiDotNetEngine.Current;
        var wide = ToTensor(returns).Cast<double>();
        int count = returns.Count;
        double maxDrawdown;
        double annualizedReturn;
        if (engine.TensorCountNonzero(engine.TensorGreaterThan(wide, -1.0)) == count)
        {
            // Every growth factor is positive: compound in log space, so a long winning run cannot overflow
            // the curve (3^700 is past double) and turn the running peak and every drawdown into NaN.
            // drawdown = 1 - curve / peak = -expm1(log curve - log peak), with the peak starting from 1.
            var logCurve = engine.TensorCumSum(engine.TensorLog1p(wide), 0);
            var logPeak = engine.TensorMax(engine.TensorCumMax(logCurve, 0), 0.0);
            maxDrawdown = engine.TensorMaxValue(
                engine.TensorNegate(engine.TensorExpm1(engine.TensorSubtract(logCurve, logPeak))));
            annualizedReturn = Math.Exp(logCurve.AsSpan()[count - 1] * periodsPerYear / count) - 1.0;
        }
        else
        {
            // A return at or below -100% (or NaN) has no logarithm: compound the curve directly.
            var curve = engine.TensorCumProd(engine.TensorAddScalar(wide, 1.0), 0);
            var peak = engine.TensorMax(engine.TensorCumMax(curve, 0), 1.0);
            var drawdowns = engine.TensorWhere(
                engine.TensorGreaterThan(peak, 0.0),
                engine.TensorDivide(engine.TensorSubtract(peak, curve), peak),
                engine.TensorZerosLike(peak));
            maxDrawdown = engine.TensorMaxValue(drawdowns);
            var equity = curve.AsSpan()[count - 1];

            // A losing streak can drive the compounded equity to zero or negative
            // (any period return <= -100%). Math.Pow of a non-positive base by a
            // fractional exponent is NaN, so treat a wiped-out account as a total
            // (-100%) annualized loss rather than emitting NaN.
            annualizedReturn = equity > 0
                ? Math.Pow(equity, (double)periodsPerYear / count) - 1.0
                : -1.0;
        }

        // A NaN drawdown (a NaN return) has no ratio either, so it takes the same zero as "no drawdown".
        if (double.IsNaN(maxDrawdown) || maxDrawdown <= 0)
        {
            return NumOps.Zero;
        }

        return NumOps.FromDouble(annualizedReturn / maxDrawdown);
    }

    private static (T Mean, T Std) MeanStd(IReadOnlyList<T> returns, T riskFree)
    {
        var engine = AiDotNetEngine.Current;
        var excess = engine.TensorSubtractScalar(ToTensor(returns), riskFree);
        var mean = MeanOf(excess, returns.Count);

        // Sample standard deviation (n-1) — unbiased for a return sample.
        var std = RootMeanSquare(engine.TensorSubtractScalar(excess, mean), returns.Count - 1);
        return (mean, std);
    }

    /// <summary>Sum / count; a count past T's range (Half: 65504) divides by sqrt(count) twice instead of zeroing the mean.</summary>
    private static T MeanOf(Tensor<T> values, int count)
    {
        T sum = AiDotNetEngine.Current.TensorSum(values);
        T n = NumOps.FromDouble(count);
        if (!NumOps.IsInfinity(n))
        {
            return NumOps.Divide(sum, n);
        }

        T rootCount = NumOps.FromDouble(Math.Sqrt(count));
        return NumOps.Divide(NumOps.Divide(sum, rootCount), rootCount);
    }

    /// <summary>sqrt(sum(x^2) / divisor), with x pre-scaled by its largest magnitude.</summary>
    /// <remarks>
    /// Return deviations near 1e-4 square to 1e-8, below Half's smallest normal (6.1e-5), so an unscaled sum of
    /// squares rounds to a few subnormal steps or to zero; squares of deviations past 255 overflow Half instead.
    /// Scaling by the largest |x| keeps every square in [0, 1]. Zero when every value is zero (or NaN).
    /// </remarks>
    private static T RootMeanSquare(Tensor<T> values, int divisor)
    {
        var engine = AiDotNetEngine.Current;
        T scale = engine.TensorMaxValue(engine.TensorAbs(values));
        if (!NumOps.GreaterThan(scale, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        T unitSquares = engine.TensorSumOfSquares(engine.TensorDivideScalar(values, scale));
        return NumOps.Multiply(scale,
            NumOps.Divide(NumOps.Sqrt(unitSquares), NumOps.FromDouble(Math.Sqrt(divisor))));
    }

    private static Tensor<T> ToTensor(IReadOnlyList<T> returns)
        => new Tensor<T>(new[] { returns.Count }, new Vector<T>(returns.ToArray()));
}
