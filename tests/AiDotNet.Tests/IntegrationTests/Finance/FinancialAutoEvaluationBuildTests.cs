using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Finance.Evaluation;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// Asserts that building a financial/forecasting model auto-evaluates it with financial-performance metrics
/// (the money-side analogue of clustering auto-evaluation), and that ConfigureFinancialMetric adds a custom
/// metric to that default set.
/// </summary>
/// <remarks>
/// Pins the observable facade behaviour: for any time-series/financial model, AiModelResult.FinancialEvaluation
/// is populated on the held-out partition with the default metric set (directional accuracy, strategy
/// Sharpe/Sortino, max drawdown, profit factor, information coefficient), those values are mirrored by name
/// into ConfiguredMetrics, and a configured custom financial metric appears alongside the defaults. It runs
/// purely by model type — no configuration is required to get the default set.
/// </remarks>
public class FinancialAutoEvaluationBuildTests
{
    // A stationary, mean-reverting series an AR model can actually fit, so the held-out predictions carry
    // real directional information for the strategy metrics.
    private static (Matrix<double> x, Vector<double> y) StationaryData(int n)
    {
        var y = new Vector<double>(n);
        var x = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++)
        {
            y[i] = Math.Sin(2.0 * Math.PI * i / 20.0);
            x[i, 0] = i;
        }

        return (x, y);
    }

    private static (ARModel<double> model, InMemoryDataLoader<double, Matrix<double>, Vector<double>> loader)
        BuildArSetup()
    {
        var (x, y) = StationaryData(120);
        var model = new ARModel<double>(new ARModelOptions<double> { AROrder = 3, MaxIterations = 100 });
        var loader = new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y);
        return (model, loader);
    }

    [Fact(Timeout = 120000)]
    public async Task Build_ForecastingModel_AutoEvaluatesWithDefaultFinancialMetrics()
    {
        var (model, loader) = BuildArSetup();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .BuildAsync();

        Assert.NotNull(result.FinancialEvaluation);
        // The default financial metrics run for any forecasting model, with no configuration.
        foreach (var name in new[] { "DirectionalAccuracy", "StrategySharpe", "StrategySortino", "StrategyMaxDrawdown", "ProfitFactor", "InformationCoefficient" })
        {
            Assert.True(result.FinancialEvaluation!.Metrics.ContainsKey(name), $"missing default financial metric {name}");
        }

        // Same values are mirrored, by name, into ConfiguredMetrics (finite ones).
        Assert.Contains("DirectionalAccuracy", result.ConfiguredMetrics.Keys);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureFinancialMetric_AddsCustomMetricToTheDefaultSet()
    {
        var (model, loader) = BuildArSetup();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .ConfigureDataLoader(loader)
            .ConfigureFinancialMetric(new TradeCountMetric<double>())
            .BuildAsync();

        Assert.NotNull(result.FinancialEvaluation);
        // The configured custom metric appears alongside the defaults, not replacing them.
        Assert.True(result.FinancialEvaluation!.Metrics.ContainsKey("TradeCount"), "custom metric missing from the evaluation");
        Assert.True(result.FinancialEvaluation.Metrics.ContainsKey("StrategySharpe"), "default metric was dropped");
    }

    /// <summary>A trivial custom financial metric (distinct name) proving ConfigureFinancialMetric extends the set.</summary>
    private sealed class TradeCountMetric<T> : IFinancialMetric<T>
    {
        public string Name => "TradeCount";
        public bool HigherIsBetter => true;

        public double Compute(IReadOnlyList<double> predicted, IReadOnlyList<double> actual)
        {
            int n = Math.Min(predicted.Count, actual.Count), trades = 0;
            for (int i = 0; i < n; i++)
            {
                if (predicted[i] != 0) trades++;
            }

            return trades;
        }
    }
}
