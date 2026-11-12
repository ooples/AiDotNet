#if NET5_0_OR_GREATER
using AiDotNet.Finance.Evaluation;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Finance.Portfolio;
using AiDotNet.Finance.Risk;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

public class FinanceNumericalRegressionTests
{
    [Fact]
    public void HalfGiftMetrics_KeepLargeCountsFloorsAndQuantileLevelsWide()
    {
        var metric = new GIFTEvalBenchmark<Half>();
        var ones = new Tensor<Half>(Enumerable.Repeat((Half)1, 70000).ToArray(), new[] { 70000 });
        var zeros = new Tensor<Half>(new[] { 70000 });
        var history = new Tensor<Half>(Enumerable.Range(0, 70001).Select(i => (Half)(i % 2)).ToArray(), new[] { 70001 });
        Assert.Equal(1.0, metric.ComputeMASE(ones, zeros, history));
        Assert.Equal(1.0, metric.ComputeMASE(ones, ones, new Tensor<Half>(new[] { (Half)1, (Half)1 }, new[] { 2 })));
        Assert.Equal(1.0, metric.ComputeCRPS(new Dictionary<double, Tensor<Half>> { [0.5] = zeros }, ones));
        var large = new Tensor<Half>(new[] { (Half)60000 }, new[] { 1 });
        var zero = new Tensor<Half>(new[] { 1 });
        Assert.Equal(0.00006, metric.ComputeQuantileLoss(zero, large, 1e-9), 12);
        double quantile = 1 - 1e-10;
        Assert.Equal((1 - quantile) * 60000, metric.ComputeQuantileLoss(large, zero, quantile), 12);
    }

    [Fact]
    public void HalfGiftBenchmark_AggregatesUnroundedLossesAndTargets()
    {
        var model = new Mock<ITimeSeriesFoundationModel<Half>>();
        model.Setup(m => m.Forecast(It.IsAny<Tensor<Half>>(), It.IsAny<double[]?>()))
            .Returns(new Tensor<Half>(new[] { 2 }));
        var series = new Tensor<Half>(new[] { (Half)0, (Half)1, (Half)60000, (Half)60000 }, new[] { 4 });
        var results = new GIFTEvalBenchmark<Half>().RunBenchmark(model.Object, new[] { series }, new[] { 2 }, new[] { 1 }, new[] { 0.5 });
        Assert.Equal(60000.0, results["MASE"]);
        Assert.Equal(1.0, results["CRPS"]);
        Assert.Equal(1.0, results["NumEvaluated"]);
        Assert.Equal(1.0, results["NumCRPSEvaluated"]);
    }

    [Fact]
    public void HalfEvaluationHarness_ReportsWideErrorsAndExactPointCounts()
    {
        const int count = 70000;
        var model = new Mock<ITimeSeriesFoundationModel<Half>>();
        model.SetupGet(m => m.SupportedTasks).Returns(new[] { TimeSeriesFoundationModelTask.Imputation });
        var predicted = new Tensor<Half>(Enumerable.Repeat((Half)60000, count).ToArray(), new[] { count });
        var target = new Tensor<Half>(Enumerable.Repeat((Half)30000, count).ToArray(), new[] { count });
        var zero = new Tensor<Half>(new[] { count });
        model.Setup(m => m.Forecast(It.IsAny<Tensor<Half>>(), It.IsAny<double[]?>())).Returns(predicted);
        model.Setup(m => m.Impute(It.IsAny<Tensor<Half>>(), It.IsAny<Tensor<Half>>())).Returns(predicted);
        var evaluator = new MultiTaskEvaluationHarness<Half>();
        var forecast = evaluator.EvaluateForecasting(model.Object, new[] { (target, target) });
        Assert.Equal(900000000.0, forecast["MSE"]);
        Assert.Equal(30000.0, forecast["MAE"]);
        Assert.Equal(30000.0, forecast["RMSE"]);
        Assert.Equal(count, forecast["NumPoints"]);
        var imputation = evaluator.EvaluateImputation(model.Object, new[] { target }, new[] { zero });
        Assert.Equal(900000000.0, imputation["MSE"]);
        Assert.Equal(30000.0, imputation["RMSE"]);
        Assert.Equal(count, imputation["NumMaskedPoints"]);
    }

    [Fact]
    public void HalfEvaluationHarness_ReportsWideAnomalyAndClassificationCounts()
    {
        const int count = 70000;
        var model = new Mock<ITimeSeriesFoundationModel<Half>>();
        model.SetupGet(m => m.SupportedTasks).Returns(new[] { TimeSeriesFoundationModelTask.AnomalyDetection, TimeSeriesFoundationModelTask.Classification });
        model.SetupGet(m => m.MaxContextLength).Returns(100000);
        model.SetupGet(m => m.MaxPredictionHorizon).Returns(100000);
        var zero = new Tensor<Half>(new[] { count });
        var labels = new Tensor<Half>(Enumerable.Repeat((Half)1, count).ToArray(), new[] { count });
        model.Setup(m => m.DetectAnomalies(It.IsAny<Tensor<Half>>(), It.IsAny<double?>())).Returns(zero);
        model.Setup(m => m.Classify(It.IsAny<Tensor<Half>>(), 2))
            .Returns(new Tensor<Half>(new[] { Half.NaN, (Half)1 }, new[] { 2 }));
        var evaluator = new MultiTaskEvaluationHarness<Half>();
        var anomalies = evaluator.EvaluateAnomalyDetection(model.Object, new[] { zero }, new[] { labels }, -1e-10);
        Assert.Equal(count, anomalies["TruePositives"]);
        Assert.Equal(1.0, anomalies["F1"]);
        var sample = new Tensor<Half>(new[] { 1 });
        var classification = evaluator.EvaluateClassification(model.Object,
            Enumerable.Repeat(sample, count).ToArray(), Enumerable.Repeat(1, count).ToArray(), 2);
        Assert.Equal(1.0, classification["Accuracy"]);
        Assert.Equal(count, classification["Correct"]);
        Assert.Equal(count, classification["Total"]);
        var metadata = evaluator.RunFullEvaluation(model.Object)["ModelInfo"];
        Assert.Equal(100000.0, metadata["MaxContextLength"]);
        Assert.Equal(100000.0, metadata["MaxPredictionHorizon"]);
    }

    [Fact]
    public void HalfBootstrapStatistics_KeepMomentsAndCountsWide()
    {
        Assert.Equal((Half)1, BootstrapConfidenceInterval<Half>.MeanStatistic(
            new Vector<Half>(Enumerable.Repeat((Half)1, 70000).ToArray())));
        Assert.Equal((Half)(3 / Math.Sqrt(2)), BootstrapConfidenceInterval<Half>.SharpeStatistic(
            new Vector<Half>(new[] { (Half)10000, (Half)20000 })));
        Assert.Equal((Half)0, BootstrapConfidenceInterval<Half>.SharpeStatistic(new Vector<Half>(new[] { (Half)1, (Half)1 })));
    }

    [Fact]
    public void HalfBootstrapPercentile_DoesNotOverflowBetweenFiniteEndpoints()
    {
        var percentile = typeof(BootstrapConfidenceInterval<Half>).GetMethod("Percentile",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        var endpoints = new Tensor<Half>(new[] { (Half)(-60000), (Half)60000 }, new[] { 2 });
        Assert.Equal((Half)0, Assert.IsType<Half>(percentile.Invoke(null, new object[] { endpoints, 50.0 })));
    }

    [Fact]
    public void HalfCvar_KeepsTailAndDualReductionsWide()
    {
        var objective = new CVaRPortfolioObjective<Half>(0.5);
        Assert.Equal((Half)60000, objective.ConditionalValueAtRisk(new Vector<Half>(Enumerable.Repeat((Half)60000, 4).ToArray())));
        Assert.Equal((Half)60000, objective.DualObjective(new Vector<Half>(Enumerable.Repeat((Half)30000, 4).ToArray()), (Half)0));
        var fractional = new CVaRPortfolioObjective<Half>(0.75);
        Assert.Equal((Half)3.8, fractional.ConditionalValueAtRisk(new Vector<Half>(new[] { (Half)0, (Half)1, (Half)2, (Half)3, (Half)4 })));
    }

    [Fact]
    public void CvarWeights_HandleLargeCountsAndExtremeFiniteTemperatures()
    {
        var half = new CVaRPortfolioObjective<Half>();
        var weights = half.Weights(new Vector<Half>(Enumerable.Repeat((Half)1, 70000).ToArray()), 1);
        Assert.All(weights.ToArray(), value => Assert.Equal((Half)(1.0 / 70000), value));
        var decimalWeights = new CVaRPortfolioObjective<decimal>().Weights(new Vector<decimal>(new[] { 1m, 2m }), 1e308);
        Assert.Equal(new[] { 0.5m, 0.5m }, decimalWeights.ToArray());
        var wide = new CVaRPortfolioObjective<double>();
        var extreme = wide.Weights(new Vector<double>(new[] { -1e308, 1e308 }), 1e308);
        Assert.Equal(1 / (1 + Math.Exp(-2)), extreme[1], 12);
        Assert.Equal(new[] { 0.0, 1.0 }, wide.Weights(new Vector<double>(new[] { 1.0, 2.0 }), 1e-320).ToArray());
        Assert.Equal(new[] { 0.5, 0.5 }, wide.Weights(new Vector<double>(new[] { 0.0, 0.0 }), 1e-320).ToArray());
    }

    [Fact]
    public void HalfTurnoverAndCost_AvoidIntermediateOverflowAndUnderflow()
    {
        var objective = new CVaRPortfolioObjective<Half>();
        var previous = new Vector<Half>(new[] { (Half)(-60000) });
        var next = new Vector<Half>(new[] { (Half)60000 });
        Assert.Equal((Half)60000, objective.Turnover(previous, next));
        Assert.Equal((Half)0.0006, objective.TransactionCost(previous, next, 0.0001));
        Assert.Equal((Half)0.0012, objective.TransactionCost(
            new Vector<Half>(new[] { (Half)(-60000), (Half)(-60000) }),
            new Vector<Half>(new[] { (Half)60000, (Half)60000 }), 0.0001));
    }

    [Fact]
    public void HalfSharpe_PreservesMomentsAllocationsAndFiniteConstantLoss()
    {
        var objective = new SharpeRatioPortfolioObjective<Half>();
        var returns = new Vector<Half>(new[] { (Half)10000, (Half)20000 });
        var moments = objective.Moments(returns);
        Assert.Equal((Half)15000, moments.Mean);
        Assert.Equal((Half)5000, moments.Volatility);
        Assert.Equal((Half)3, objective.SharpeRatio(returns));
        Assert.Equal((Half)Math.Log(1e-12), objective.Loss(new Vector<Half>(new[] { (Half)1, (Half)1 })));
        Assert.Equal(new[] { (Half)0.5, (Half)0.5 }, objective.Allocate(new Vector<Half>(new[] { (Half)60000, (Half)60000 })).ToArray());
    }

    [Fact]
    public void HalfGraphStatistics_PreserveFiniteVolatilityAndPerfectDependence()
    {
        var builder = new AssetGraphBuilder<Half>(2);
        var volatility = builder.VolatilitySeries(new Tensor<Half>(new[] { (Half)60000, (Half)(-60000) }, new[] { 2, 1 }));
        Assert.Equal((Half)60000, volatility.GetFlat(0));
        var values = new Vector<Half>(Enumerable.Range(0, 257).Select(i => (Half)(i % 11)).ToArray());
        Assert.Equal((Half)1, builder.DistanceCorrelation(values, values));
        Assert.Equal((Half)0, builder.DistanceCorrelation(values, new Vector<Half>(257)));
        var panel = new Tensor<Half>(new[] { 257, 2 });
        for (int row = 0; row < 257; row++) panel[row, 0] = panel[row, 1] = values[row];
        Assert.All(builder.DistanceCorrelationMatrix(panel).ToArray(), value => Assert.Equal((Half)1, value));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void EmptyAssetPanel_ReturnsEmptyCorrelationMatrix(int steps)
        => Assert.Equal(new[] { 0, 0 }, new AssetGraphBuilder<double>().DistanceCorrelationMatrix(new Tensor<double>(new[] { steps, 0 })).Shape);

    [Fact]
    public void GraphVarianceDiagonal_DoesNotReadOffDiagonalNaNs()
    {
        var panel = new Tensor<double>(new[] { 0.0, 0, double.NaN, 1, 1, double.NaN, 2, 2, double.NaN }, new[] { 3, 3 });
        var correlations = new AssetGraphBuilder<double>().DistanceCorrelationMatrix(panel);
        Assert.Equal(1.0, correlations[0, 1], 12);
        Assert.Equal(1.0, correlations[1, 0], 12);
    }

    [Fact]
    public void SignedAreas_UseWideGeometryAndExactZeroDiagonal()
    {
        var half = new PathSignatureTransform<Half>();
        var x = new Vector<Half>(new[] { (Half)0, (Half)60000, (Half)0 });
        var y = new Vector<Half>(new[] { (Half)0, (Half)60000, (Half)1 });
        Assert.Equal((Half)30000, half.SignedArea(x, y));
        var areas = half.CrossSignedAreas(new Tensor<Half>(new[] { (Half)0, (Half)0, (Half)60000, (Half)60000, (Half)0, (Half)1 }, new[] { 3, 2 }));
        Assert.Equal(new[] { (Half)0, (Half)30000, (Half)(-30000), (Half)0 }, areas.ToArray());
        var large = new PathSignatureTransform<float>().CrossSignedAreas(new Tensor<float>(new[] { 0f, 1e20f, 2e20f }, new[] { 3, 1 }));
        Assert.Equal(0f, large.GetFlat(0));
    }

    [Fact]
    public void Signature_ExclusivePrefixDoesNotCancelBeforeLargeIncrement()
    {
        double increment = Math.Pow(2, 54);
        var path = new Tensor<double>(new[] { 0.0, 0, 2.5, 0, increment, 1 }, new[] { 3, 2 });
        var signature = new PathSignatureTransform<double>().Signature(path);
        Assert.Equal(increment * 0.5 + 2, signature[3]);
    }

    [Fact]
    public void InformationCoefficient_KeepsGeometryRanksAndCountsWide()
    {
        Half[] values = { (Half)(-60000), (Half)60000 };
        Assert.Equal((Half)1, InformationCoefficient<Half>.Pearson(values, values));
        var tied = Enumerable.Range(0, 65504).Select(i => (Half)(i / 32752)).ToArray();
        Assert.Equal((Half)1, InformationCoefficient<Half>.Spearman(tied, tied));
        var ratio = InformationCoefficient<Half>.InformationRatio(new[] { (Half)10000, (Half)20000 });
        Assert.Equal((Half)15000, ratio.MeanIc);
        Assert.Equal((Half)Math.Sqrt(50000000), ratio.StdIc);
        Assert.Equal((Half)3, ratio.Icir);
        Assert.True(double.IsNaN(InformationCoefficient<double>.Spearman(new[] { 0.0, double.NaN, 2 }, new[] { 0.0, 1, 2 })));
    }

    [Fact]
    public void HalfCalmar_CompoundsSmallReturnsBeforeRounding()
    {
        var returns = Enumerable.Repeat((Half)0.0001, 10001).ToArray();
        returns[returns.Length - 1] = (Half)(-0.001);
        double equity = 1, peak = 1, drawdown = 0;
        foreach (Half value in returns)
        {
            equity *= 1 + (double)value;
            peak = Math.Max(peak, equity);
            drawdown = Math.Max(drawdown, (peak - equity) / peak);
        }
        double expected = (Math.Pow(equity, 252.0 / returns.Length) - 1) / drawdown;
        Assert.Equal((Half)expected, RiskRatios<Half>.Calmar(returns));
    }
}
#endif
