#if NET5_0_OR_GREATER
using AiDotNet.Finance.Evaluation;
using AiDotNet.Finance.Portfolio;
using AiDotNet.Finance.Risk;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

public class FinanceNumericalRegressionTests
{
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
