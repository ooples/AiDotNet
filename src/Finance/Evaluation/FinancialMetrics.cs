using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// A financial-performance metric for a FORECASTING/financial model — the analogue of a cluster-validity index
/// for a clustering model. It scores how a model's predictions would have performed as a trading signal
/// (sign(prediction) taken against the realized return), so a forecaster is judged on money, not just error.
/// </summary>
/// <typeparam name="T">The model's numeric type (metrics compute in double internally).</typeparam>
public interface IFinancialMetric<T>
{
    /// <summary>Metric name (e.g. "StrategySharpe", "DirectionalAccuracy").</summary>
    string Name { get; }

    /// <summary>Higher is better for ranking (drawdown metrics negate so "higher is better" holds uniformly).</summary>
    bool HigherIsBetter { get; }

    /// <summary>Computes the metric from aligned predicted and realized (actual) return series.</summary>
    double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual);
}

/// <summary>The financial evaluation of a model: a name→value map of the metrics that were run.</summary>
public sealed class FinancialEvaluationResult
{
    public IReadOnlyDictionary<string, double> Metrics { get; }

    public FinancialEvaluationResult(IReadOnlyDictionary<string, double> metrics) => Metrics = metrics;

    public double this[string name] => Metrics.TryGetValue(name, out var v) ? v : double.NaN;
}

/// <summary>Shared math over a signal-following strategy: strategy return series = sign(prediction)·actual.</summary>
internal static class FinancialMath
{
    public const double PeriodsPerYear = 252.0;

    /// <summary>Strategy per-step returns: go long when the forecast is positive, short when negative.</summary>
    public static double[] StrategyReturns(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        int n = Math.Min(predicted.Count, actual.Count);
        var r = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sign = predicted[i] > 0 ? 1.0 : predicted[i] < 0 ? -1.0 : 0.0;
            r[i] = sign * actual[i];
        }

        return r;
    }

    public static (double mean, double std, double downsideStd) Moments(double[] r)
    {
        if (r.Length == 0) return (0, 0, 0);
        double s = 0, s2 = 0, d2 = 0;
        foreach (var x in r)
        {
            s += x;
            s2 += x * x;
            if (x < 0) d2 += x * x;
        }
        double mean = s / r.Length;
        double std = Math.Sqrt(Math.Max(0, s2 / r.Length - mean * mean));
        double downsideStd = Math.Sqrt(d2 / r.Length);
        return (mean, std, downsideStd);
    }

    /// <summary>Max drawdown of the compounded strategy equity curve, in [0, 1].</summary>
    public static double MaxDrawdown(double[] r)
    {
        double equity = 1.0, peak = 1.0, maxDd = 0.0;
        foreach (var x in r)
        {
            equity *= 1.0 + x;
            if (equity > peak) peak = equity;
            if (peak > 0) maxDd = Math.Max(maxDd, (peak - equity) / peak);
        }
        return maxDd;
    }
}

/// <summary>Fraction of steps where the forecast got the DIRECTION right (sign match) — the hit rate.</summary>
public sealed class DirectionalAccuracyMetric<T> : IFinancialMetric<T>
{
    public string Name => "DirectionalAccuracy";
    public bool HigherIsBetter => true;

    public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        int n = Math.Min(predicted.Count, actual.Count), correct = 0, counted = 0;
        for (int i = 0; i < n; i++)
        {
            if (actual[i] == 0) continue;
            counted++;
            if (Math.Sign(predicted[i]) == Math.Sign(actual[i])) correct++;
        }
        return counted > 0 ? (double)correct / counted : 0.0;
    }
}

/// <summary>Annualized Sharpe ratio of the signal-following strategy.</summary>
public sealed class StrategySharpeMetric<T> : IFinancialMetric<T>
{
    public string Name => "StrategySharpe";
    public bool HigherIsBetter => true;

    public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        var (mean, std, _) = FinancialMath.Moments(FinancialMath.StrategyReturns(predicted, actual));
        return std > 1e-12 ? mean / std * Math.Sqrt(FinancialMath.PeriodsPerYear) : 0.0;
    }
}

/// <summary>Annualized Sortino ratio (downside-only volatility) of the signal-following strategy.</summary>
public sealed class StrategySortinoMetric<T> : IFinancialMetric<T>
{
    public string Name => "StrategySortino";
    public bool HigherIsBetter => true;

    public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        var (mean, _, downside) = FinancialMath.Moments(FinancialMath.StrategyReturns(predicted, actual));
        return downside > 1e-12 ? mean / downside * Math.Sqrt(FinancialMath.PeriodsPerYear) : 0.0;
    }
}

/// <summary>Max drawdown of the strategy equity curve (NEGATED so higher is better for uniform ranking).</summary>
public sealed class MaxDrawdownMetric<T> : IFinancialMetric<T>
{
    public string Name => "StrategyMaxDrawdown";
    public bool HigherIsBetter => true; // reports -drawdown

    public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
        => -FinancialMath.MaxDrawdown(FinancialMath.StrategyReturns(predicted, actual));
}

/// <summary>Profit factor: gross profit / gross loss of the signal-following strategy.</summary>
public sealed class ProfitFactorMetric<T> : IFinancialMetric<T>
{
    public string Name => "ProfitFactor";
    public bool HigherIsBetter => true;

    public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        double up = 0, down = 0;
        foreach (var x in FinancialMath.StrategyReturns(predicted, actual))
        {
            if (x > 0) up += x;
            else down -= x;
        }
        return down > 1e-12 ? up / down : (up > 0 ? double.PositiveInfinity : 0.0);
    }
}

/// <summary>Information coefficient: Pearson correlation between predicted and realized returns.</summary>
public sealed class InformationCoefficientMetric<T> : IFinancialMetric<T>
{
    public string Name => "InformationCoefficient";
    public bool HigherIsBetter => true;

    public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        int n = Math.Min(predicted.Count, actual.Count);
        if (n < 2) return 0.0;
        double mp = 0, ma = 0;
        for (int i = 0; i < n; i++) { mp += predicted[i]; ma += actual[i]; }
        mp /= n; ma /= n;
        double cov = 0, vp = 0, va = 0;
        for (int i = 0; i < n; i++)
        {
            double dp = predicted[i] - mp, da = actual[i] - ma;
            cov += dp * da; vp += dp * dp; va += da * da;
        }
        double denom = Math.Sqrt(vp * va);
        return denom > 1e-12 ? cov / denom : 0.0;
    }
}

/// <summary>
/// Runs a set of financial metrics over a model's predicted-vs-realized returns. The default set —
/// directional accuracy, strategy Sharpe/Sortino, max drawdown, profit factor, information coefficient — runs
/// with no configuration (the analogue of the default cluster-validity indices); <see cref="AddMetric"/> extends
/// it, matching how <c>ConfigureClusterMetric</c> extends the default clustering indices.
/// </summary>
public sealed class FinancialEvaluator<T>
{
    private readonly List<IFinancialMetric<T>> _metrics =
    [
        new DirectionalAccuracyMetric<T>(),
        new StrategySharpeMetric<T>(),
        new StrategySortinoMetric<T>(),
        new MaxDrawdownMetric<T>(),
        new ProfitFactorMetric<T>(),
        new InformationCoefficientMetric<T>(),
    ];

    /// <summary>Adds a metric to the default set (does not replace it).</summary>
    public void AddMetric(IFinancialMetric<T> metric)
    {
        ArgumentNullException.ThrowIfNull(metric);
        _metrics.Add(metric);
    }

    public FinancialEvaluationResult Evaluate(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(actual);

        var results = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var m in _metrics)
        {
            double v;
            try { v = m.Compute(predicted, actual); }
            catch { v = double.NaN; }
            results[m.Name] = v;
        }

        return new FinancialEvaluationResult(results);
    }
}
