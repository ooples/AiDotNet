using System;
using System.Collections.Generic;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Evaluation;

/// <summary>
/// Risk-and-cost-aware performance of running one policy through a <see cref="PortfolioManagerEnvironment{T}"/>
/// once. These are the numbers an experiment is RANKED on when comparing candidate agents/rewards on an
/// untouched holdout — never training loss.
/// </summary>
public sealed record PortfolioBacktestResult(
    double FinalValue,
    double TotalReturn,
    double AnnualizedSharpe,
    double AnnualizedSortino,
    double MaxDrawdown,
    double Calmar,
    double AverageTurnover,
    int Steps);

/// <summary>
/// Runs a policy through a portfolio environment deterministically and scores it. A "policy" is just
/// <c>state → action</c>, so this evaluates a scripted baseline and a trained RL agent
/// (<c>s => agent.SelectAction(s, training:false)</c>) the same way — the honest-eval seam for the greenfield
/// trading-agent research: hold out a chronological tail, run each candidate, and compare on these metrics.
/// </summary>
public static class PortfolioBacktest
{
    /// <summary>Trading days per year, for annualizing the Sharpe ratio.</summary>
    private const double PeriodsPerYear = 252.0;

    /// <summary>
    /// Runs <paramref name="policy"/> through <paramref name="environment"/> for one full episode and returns
    /// its risk/cost-adjusted performance. The environment is reset first; its per-episode reward/peak state is
    /// cleared so a reused instance scores cleanly.
    /// </summary>
    public static PortfolioBacktestResult Run<T>(
        PortfolioManagerEnvironment<T> environment,
        Func<Vector<T>, Vector<T>> policy)
    {
        if (environment is null) throw new ArgumentNullException(nameof(environment));
        if (policy is null) throw new ArgumentNullException(nameof(policy));

        environment.ResetEpisodeState();
        var state = environment.Reset();

        // Seed with the pre-trade baseline (initial capital) so total return, Sharpe, and drawdown all include
        // the very first step's move rather than starting the series one step in.
        var values = new List<double> { environment.CurrentValue };
        double turnoverSum = 0;
        int steps = 0;
        bool done = false;

        while (!done)
        {
            var action = policy(state);
            var (nextState, _, isDone, info) = environment.Step(action);

            double value = info.TryGetValue("portfolioValue", out var pv) ? Convert.ToDouble(pv) : double.NaN;
            if (!double.IsNaN(value))
            {
                values.Add(value);
            }

            turnoverSum += environment.LastTurnover;
            steps++;
            state = nextState;
            done = isDone;
        }

        return Score(values, turnoverSum, steps);
    }

    /// <summary>Computes the performance metrics from a per-step portfolio-value series.</summary>
    private static PortfolioBacktestResult Score(IReadOnlyList<double> values, double turnoverSum, int steps)
    {
        if (values.Count < 2)
        {
            double single = values.Count == 1 ? values[0] : 0.0;
            return new PortfolioBacktestResult(single, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, steps);
        }

        double start = values[0];
        double final = values[^1];
        double totalReturn = start > 0 ? (final - start) / start : 0.0;

        // Per-step returns → annualized Sharpe + Sortino (Sortino penalizes only DOWNSIDE volatility).
        double sumR = 0, sumR2 = 0, sumDown2 = 0;
        int n = 0;
        for (int i = 1; i < values.Count; i++)
        {
            double prev = values[i - 1];
            if (prev <= 0)
            {
                continue;
            }

            double r = (values[i] - prev) / prev;
            sumR += r;
            sumR2 += r * r;
            if (r < 0)
            {
                sumDown2 += r * r;
            }
            n++;
        }

        double sharpe = 0.0, sortino = 0.0, mean = 0.0;
        if (n > 1)
        {
            mean = sumR / n;
            double variance = Math.Max(0.0, (sumR2 / n) - (mean * mean));
            double std = Math.Sqrt(variance);
            sharpe = std > 1e-12 ? (mean / std) * Math.Sqrt(PeriodsPerYear) : 0.0;
            double downsideDev = Math.Sqrt(sumDown2 / n);
            sortino = downsideDev > 1e-12 ? (mean / downsideDev) * Math.Sqrt(PeriodsPerYear) : 0.0;
        }

        // Max drawdown from the running peak.
        double peak = values[0], maxDd = 0.0;
        foreach (var v in values)
        {
            if (v > peak)
            {
                peak = v;
            }

            if (peak > 0)
            {
                maxDd = Math.Max(maxDd, (peak - v) / peak);
            }
        }

        // Calmar = annualized return / max drawdown (return per unit of worst-case loss). Annualize the
        // COMPOUNDED total return over the n observed periods geometrically, not the arithmetic per-step mean.
        double annualizedReturn;
        if (n > 0 && (1.0 + totalReturn) > 0.0)
        {
            annualizedReturn = Math.Pow(1.0 + totalReturn, PeriodsPerYear / n) - 1.0;
        }
        else
        {
            annualizedReturn = mean * PeriodsPerYear; // degenerate (total loss / no periods): fall back to linear.
        }

        double calmar = maxDd > 1e-12 ? annualizedReturn / maxDd : 0.0;

        double avgTurnover = steps > 0 ? turnoverSum / steps : 0.0;
        return new PortfolioBacktestResult(final, totalReturn, sharpe, sortino, maxDd, calmar, avgTurnover, steps);
    }
}

/// <summary>
/// No-skill baseline policies to rank a learned agent against. If a trained portfolio manager cannot beat these
/// on the untouched holdout, it has learned nothing worth trading (see the honest-eval discipline).
/// </summary>
public static class BaselinePolicies
{
    /// <summary>Equal-weight long book: allocate 1/N to each of the <paramref name="tradableCount"/> assets every
    /// step (the environment holds the mix, so this is effectively rebalanced buy-and-hold — the classic
    /// no-skill control).</summary>
    public static Func<Vector<T>, Vector<T>> EqualWeight<T>(int tradableCount)
    {
        if (tradableCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(tradableCount));
        }

        var w = MathHelper.GetNumericOperations<T>().FromDouble(1.0 / tradableCount);
        return _ =>
        {
            var a = new Tensor<T>(new[] { tradableCount });
            AiDotNetEngine.Current.TensorFill(a, w);
            return a.ToVector();
        };
    }

    /// <summary>Flat (all-cash) policy: never take a position. The zero-risk control.</summary>
    public static Func<Vector<T>, Vector<T>> Flat<T>(int tradableCount)
        => _ => new Vector<T>(tradableCount);

    /// <summary>
    /// Cross-sectional MOMENTUM baseline: go equal-weight long the tradable assets whose price rose over the
    /// observation window (recent winners), flat the rest. A far harder benchmark than equal-weight — momentum
    /// is a real, persistent effect — so beating it on the holdout is a much stronger signal of skill.
    /// <para>
    /// Decodes the environment's observation layout: the first <c>windowSize * totalColumns</c> entries are the
    /// per-step price/feature window (row-major by time then asset), so asset i's window-start and window-end
    /// prices are <c>obs[i]</c> and <c>obs[(windowSize-1)*totalColumns + i]</c>. <paramref name="totalColumns"/>
    /// is the env's NumAssets (tradable + feature columns); only the first <paramref name="tradableCount"/> are
    /// traded.
    /// </para>
    /// </summary>
    public static Func<Vector<T>, Vector<T>> Momentum<T>(int windowSize, int totalColumns, int tradableCount)
    {
        if (windowSize < 2) throw new ArgumentOutOfRangeException(nameof(windowSize), "Momentum needs a window of at least 2.");
        if (tradableCount <= 0 || tradableCount > totalColumns) throw new ArgumentOutOfRangeException(nameof(tradableCount));

        return state =>
        {
            // Window-start vs window-end price of each tradable asset; equal weight on the winners.
            var engine = AiDotNetEngine.Current;
            var ops = MathHelper.GetNumericOperations<T>();
            var prices = TradablePrices(state, windowSize, totalColumns, tradableCount);
            var first = engine.TensorSlice(prices, new[] { 0, 0 }, new[] { 1, tradableCount });
            var last = engine.TensorSlice(prices, new[] { windowSize - 1, 0 }, new[] { 1, tradableCount });
            var winners = engine.Reshape(engine.TensorGreaterThan(last, first), new[] { tradableCount });

            // The winners are counted in int (a 0/1 sum in Half stops at 2048) and each gets 1 / count, formed in
            // double and rounded once as master did.
            int count = (int)engine.TensorCountNonzero(winners);
            return count > 0
                ? engine.TensorMultiplyScalar(winners, ops.FromDouble(1.0 / count)).ToVector()
                : new Vector<T>(tradableCount);
        };
    }

    /// <summary>
    /// RISK-PARITY (inverse-volatility) baseline: allocate long weights proportional to 1/σ of each tradable
    /// asset's recent returns, so every name contributes roughly equal risk (calmer assets get more capital).
    /// A standard institutional benchmark that usually earns a HIGHER Sharpe than equal-weight, making it a
    /// tougher control. Decodes the same observation window as <see cref="Momentum{T}"/>.
    /// </summary>
    public static Func<Vector<T>, Vector<T>> RiskParity<T>(int windowSize, int totalColumns, int tradableCount)
    {
        if (windowSize < 3) throw new ArgumentOutOfRangeException(nameof(windowSize), "Risk-parity needs a window of at least 3.");
        if (tradableCount <= 0 || tradableCount > totalColumns) throw new ArgumentOutOfRangeException(nameof(tradableCount));

        return state =>
        {
            var engine = AiDotNetEngine.Current;
            var ops = MathHelper.GetNumericOperations<T>();
            var prices = TradablePrices(state, windowSize, totalColumns, tradableCount);
            int steps = windowSize - 1;
            var prev = engine.TensorSlice(prices, new[] { 0, 0 }, new[] { steps, tradableCount });
            var cur = engine.TensorSlice(prices, new[] { 1, 0 }, new[] { steps, tradableCount });

            // Per-step returns of each asset across the window, counting only steps with a positive previous price.
            var zeros = new Tensor<T>(new[] { steps, tradableCount });
            var valid = engine.TensorGreaterThan(prev, ops.Zero);
            var safePrev = engine.TensorWhere(valid, prev, engine.TensorAddScalar(zeros, ops.One));
            var r = engine.TensorWhere(valid, engine.TensorDivide(engine.TensorSubtract(cur, prev), safePrev), zeros);
            // Valid-return counts per asset are summed in int: a 0/1 sum in Half stops at 2048.
            var m = engine.ReduceSum(valid.Cast<int>(), new[] { 0 }, keepDims: false).Cast<T>();
            var sr = engine.ReduceSum(r, new[] { 0 }, keepDims: false);
            var sr2 = engine.ReduceSum(engine.TensorSquare(r), new[] { 0 }, keepDims: false);

            // Population variance of the valid returns; an asset without at least two valid returns gets zero
            // weight, otherwise a degenerate near-zero vol would hand it a runaway inverse-vol weight.
            var mSafe = engine.TensorClampMin(m, ops.One);
            var mu = engine.TensorDivide(sr, mSafe);
            var variance = engine.TensorClampMin(
                engine.TensorSubtract(engine.TensorDivide(sr2, mSafe), engine.TensorSquare(mu)), ops.Zero);
            // The 1e-6 floor is a Tolerance: in Half 1e-6 is subnormal and 1 / 1e-6 overflows to infinity, which
            // turned a zero-volatility asset into an inf / inf = NaN weight.
            var vol = engine.TensorAddScalar(engine.TensorSqrt(variance), FinanceNumerics.Tolerance<T>(1e-6));
            var invVol = engine.TensorWhere(
                engine.TensorGreaterThan(m, ops.One),
                engine.TensorReciprocal(vol),
                new Tensor<T>(new[] { tradableCount }));
            // Normalized by the largest inverse vol before the sum: each share is then at most 1 and the sum at most
            // tradableCount. In Half four flat assets each have 1 / 2^-14 = 16384, whose raw sum (65536) is past
            // 65504, so the weights came out 16384 / inf = 0 instead of 0.25 each.
            T largest = engine.TensorMaxValue(invVol);
            if (!ops.GreaterThan(largest, ops.Zero))
            {
                return new Vector<T>(tradableCount);
            }

            var share = engine.TensorDivideScalar(invVol, largest);
            return engine.TensorDivideScalar(share, engine.TensorSum(share)).ToVector();
        };
    }

    /// <summary>
    /// The observation's price window as <c>[windowSize, tradableCount]</c>: the first
    /// <c>windowSize * totalColumns</c> entries are row-major by time then asset.
    /// </summary>
    private static Tensor<T> TradablePrices<T>(Vector<T> state, int windowSize, int totalColumns, int tradableCount)
    {
        var engine = AiDotNetEngine.Current;
        var window = engine.Reshape(
            engine.TensorSlice(Tensor<T>.FromVector(state), new[] { 0 }, new[] { windowSize * totalColumns }),
            new[] { windowSize, totalColumns });
        return engine.TensorSlice(window, new[] { 0, 0 }, new[] { windowSize, tradableCount });
    }
}
