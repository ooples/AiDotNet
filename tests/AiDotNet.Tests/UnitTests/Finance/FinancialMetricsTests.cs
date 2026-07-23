using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Evaluation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the financial-metrics core: scoring a forecaster on money (signal-following strategy) rather than
/// error. A perfect direction forecast maxes the money metrics; an anti-forecast inverts them.
/// </summary>
public sealed class FinancialMetricsTests
{
    // A choppy but net-varying realized return series (so the metrics are finite / well-defined).
    private static readonly double[] Actual =
        { 0.02, -0.01, 0.03, -0.02, 0.015, -0.008, 0.025, -0.012, 0.018, -0.02, 0.03, -0.015 };

    private static double[] Perfect() => Actual.ToArray();                 // forecast matches direction exactly
    private static double[] Anti() => Actual.Select(a => -a).ToArray();    // forecast is exactly wrong

    [Fact]
    [Trait("category", "unit")]
    public void Directional_accuracy_is_1_for_a_perfect_forecast_and_0_for_the_anti_forecast()
    {
        var m = new DirectionalAccuracyMetric<double>();
        Assert.Equal(1.0, m.Compute(Perfect(), Actual), 10);
        Assert.Equal(0.0, m.Compute(Anti(), Actual), 10);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Strategy_sharpe_is_positive_for_a_perfect_forecast_and_negative_for_the_anti()
    {
        var sharpe = new StrategySharpeMetric<double>();
        var sortino = new StrategySortinoMetric<double>();

        Assert.True(sharpe.Compute(Perfect(), Actual) > 0);
        Assert.True(sharpe.Compute(Anti(), Actual) < 0);
        // A perfect forecast has NO losing trades → zero downside → Sortino is degenerate (0). The anti-forecast
        // loses every trade → full downside → a clearly negative Sortino.
        Assert.True(sortino.Compute(Anti(), Actual) < 0);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Information_coefficient_is_1_when_predictions_equal_actuals_and_minus_1_when_inverted()
    {
        var ic = new InformationCoefficientMetric<double>();
        Assert.Equal(1.0, ic.Compute(Perfect(), Actual), 6);
        Assert.Equal(-1.0, ic.Compute(Anti(), Actual), 6);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Max_drawdown_metric_is_non_positive_and_worse_for_the_anti_forecast()
    {
        var dd = new MaxDrawdownMetric<double>(); // reports -drawdown (higher is better)
        double perfect = dd.Compute(Perfect(), Actual);
        double anti = dd.Compute(Anti(), Actual);
        Assert.True(perfect <= 0 && anti <= 0);
        Assert.True(perfect >= anti, "the perfect forecast should have the smaller (less negative) drawdown");
    }

    [Fact]
    [Trait("category", "unit")]
    public void Evaluator_runs_the_full_default_metric_set()
    {
        var eval = new FinancialEvaluator<double>();
        var result = eval.Evaluate(Perfect(), Actual);

        foreach (var name in new[] { "DirectionalAccuracy", "StrategySharpe", "StrategySortino", "StrategyMaxDrawdown", "ProfitFactor", "InformationCoefficient" })
        {
            Assert.True(result.Metrics.ContainsKey(name), $"missing default metric {name}");
        }

        Assert.Equal(1.0, result["DirectionalAccuracy"], 10);
        Assert.True(result["StrategySharpe"] > 0);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Configured_metric_extends_the_default_set()
    {
        var eval = new FinancialEvaluator<double>();
        eval.AddMetric(new ProfitFactorMetric<double>()); // adding another instance still shows up (extends, not replaces)
        var result = eval.Evaluate(Perfect(), Actual);
        Assert.True(result.Metrics.Count >= 6);
    }
}
