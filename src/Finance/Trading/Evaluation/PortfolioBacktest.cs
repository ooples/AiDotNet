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
    double MaxDrawdown,
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
        ArgumentNullException.ThrowIfNull(environment);
        ArgumentNullException.ThrowIfNull(policy);

        environment.ResetEpisodeState();
        var state = environment.Reset();

        var values = new List<double>();
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
            return new PortfolioBacktestResult(single, 0.0, 0.0, 0.0, 0.0, steps);
        }

        double start = values[0];
        double final = values[^1];
        double totalReturn = start > 0 ? (final - start) / start : 0.0;

        // Per-step returns → annualized Sharpe.
        double sumR = 0, sumR2 = 0;
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
            n++;
        }

        double sharpe = 0.0;
        if (n > 1)
        {
            double mean = sumR / n;
            double variance = Math.Max(0.0, (sumR2 / n) - (mean * mean));
            double std = Math.Sqrt(variance);
            sharpe = std > 1e-12 ? (mean / std) * Math.Sqrt(PeriodsPerYear) : 0.0;
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

        double avgTurnover = steps > 0 ? turnoverSum / steps : 0.0;
        return new PortfolioBacktestResult(final, totalReturn, sharpe, maxDd, avgTurnover, steps);
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

        var w = (T)Convert.ChangeType(1.0 / tradableCount, typeof(T), System.Globalization.CultureInfo.InvariantCulture);
        return _ =>
        {
            var a = new Vector<T>(tradableCount);
            for (int i = 0; i < tradableCount; i++)
            {
                a[i] = w;
            }

            return a;
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

        int lastRow = (windowSize - 1) * totalColumns;
        return state =>
        {
            var winners = new bool[tradableCount];
            int count = 0;
            for (int i = 0; i < tradableCount; i++)
            {
                double first = Convert.ToDouble(state[i]);
                double last = Convert.ToDouble(state[lastRow + i]);
                if (last > first)
                {
                    winners[i] = true;
                    count++;
                }
            }

            var action = new Vector<T>(tradableCount);
            if (count > 0)
            {
                var w = (T)Convert.ChangeType(1.0 / count, typeof(T), System.Globalization.CultureInfo.InvariantCulture);
                for (int i = 0; i < tradableCount; i++)
                {
                    if (winners[i])
                    {
                        action[i] = w;
                    }
                }
            }

            return action;
        };
    }
}
