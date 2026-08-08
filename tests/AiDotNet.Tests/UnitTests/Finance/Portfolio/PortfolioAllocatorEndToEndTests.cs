using System;
using AiDotNet.Finance.Portfolio;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance.Portfolio;

/// <summary>
/// End-to-end checks that the two portfolio allocators' NETWORK HEADS agree with their allocation
/// layers.
/// </summary>
/// <remarks>
/// <para>
/// These exist because component-level tests could not catch the mismatch they cover. Both models
/// originally shared a layer factory whose output head ended in a softmax, and both allocation layers
/// were verified in isolation by calling them with hand-built scores. The assembled models were
/// nonetheless wrong:
/// </para>
/// <list type="bullet">
/// <item>SIT applied a tempered softmax on top of a softmax head, so its temperature acted on values
/// already squashed into (0, 1) and summing to 1.</item>
/// <item>GAT's sum-normalizing allocation could never emit an exact zero, because a softmax head
/// cannot produce one — losing the sparsity the paper rejected softmax to obtain.</item>
/// </list>
/// <para>
/// The lesson these encode: a component test that supplies its own inputs cannot verify what the
/// network actually hands the component.
/// </para>
/// </remarks>
public class PortfolioAllocatorEndToEndTests
{
    private const int Assets = 12;

    private static Tensor<double> Panel(int steps, int assets, int seed)
    {
        var t = new Tensor<double>(new[] { steps, assets });
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = (rng.NextDouble() * 0.04) - 0.02;
        return t;
    }

    private static double Sum(Vector<double> v)
    {
        double s = 0.0;
        for (int i = 0; i < v.Length; i++) s += v[i];
        return s;
    }

    // ------------------------------------------------------------ SIT

    [Fact]
    public void SitNetworkHeadEmitsRawScoresNotADistribution()
    {
        // The head must be LINEAR: SIT's objective applies softmax(scores / tau) itself. If the head
        // also softmaxed, its output would already sum to 1 and tau would be operating on a
        // distribution — making the paper's reported optimum near tau = 1.3 meaningless.
        var options = new SignatureInformedTransformerOptions<double> { NumAssets = Assets, LookbackWindow = 30 };
        var model = new SignatureInformedTransformer<double>(options);

        var scores = model.Predict(Panel(30, Assets, 3)).ToVector();

        double sum = 0.0;
        bool anyNegative = false;
        for (int i = 0; i < Math.Min(scores.Length, Assets); i++)
        {
            sum += scores[i];
            if (scores[i] < 0.0) anyNegative = true;
        }

        // A softmax head would give exactly 1.0 here and no negative entries. Either signal alone is
        // enough to prove the head is not normalizing.
        bool looksLikeADistribution = Math.Abs(sum - 1.0) < 1e-6 && !anyNegative;
        Assert.False(looksLikeADistribution,
            $"SIT's head appears to already normalize (sum={sum:R}, anyNegative={anyNegative}); " +
            "the tempered softmax in the objective would then be applied twice.");
    }

    [Fact]
    public void SitProducesAValidLongOnlyFullyInvestedAllocation()
    {
        var options = new SignatureInformedTransformerOptions<double> { NumAssets = Assets, LookbackWindow = 30 };
        var model = new SignatureInformedTransformer<double>(options);

        var w = model.OptimizePortfolio(Panel(30, Assets, 5));

        Assert.Equal(Assets, w.Length);
        for (int i = 0; i < w.Length; i++)
        {
            Assert.False(double.IsNaN(w[i]) || double.IsInfinity(w[i]), $"weight {i} is non-finite.");
            Assert.True(w[i] > 0.0, $"weight {i} = {w[i]} is not long-only.");
        }
        Assert.Equal(1.0, Sum(w), 8);
    }

    [Fact]
    public void SitTemperatureChangesTheRealizedConcentration()
    {
        // With a linear head the temperature has its intended effect end to end: a small tau
        // concentrates the allocation. Under a double softmax the scores would already be flattened
        // into a distribution and this spread would shrink dramatically.
        var input = Panel(30, Assets, 11);

        var concentrated = new SignatureInformedTransformer<double>(
            new SignatureInformedTransformerOptions<double>
            { NumAssets = Assets, LookbackWindow = 30, Temperature = 0.05 });
        var diffuse = new SignatureInformedTransformer<double>(
            new SignatureInformedTransformerOptions<double>
            { NumAssets = Assets, LookbackWindow = 30, Temperature = 50.0 });
        diffuse.UpdateParameters(concentrated.GetParameters());   // same weights, only tau differs

        var wc = concentrated.OptimizePortfolio(input);
        var wd = diffuse.OptimizePortfolio(input);

        double spreadC = Max(wc) - Min(wc);
        double spreadD = Max(wd) - Min(wd);

        Assert.True(spreadC > spreadD,
            $"A lower temperature must concentrate the allocation; got spread {spreadC:R} vs {spreadD:R}.");
    }

    // ------------------------------------------------------------ GAT

    [Fact]
    public void GatProducesASparseAllocationEndToEnd()
    {
        // THE property the paper chose its allocation layer for. The head is ReLU so scores contain
        // exact zeros, and sum-normalization turns those into exact zero weights, dropping firms
        // entirely. A softmax head cannot emit a zero, so every asset would be held — the "many tiny
        // holdings" the paper rejected softmax to avoid.
        var options = new GraphAttentionPortfolioOptions<double> { NumAssets = Assets, CorrelationWindow = 40 };
        var model = new GraphAttentionPortfolio<double>(options);

        var w = model.OptimizePortfolio(Panel(40, Assets, 7));

        int active = model.Objective.ActivePositions(w);
        Assert.True(active < Assets,
            $"GAT must be able to exclude assets, but all {Assets} received capital. " +
            "A softmax output head would produce exactly this symptom.");
        Assert.True(active >= 1, "At least one position must be held.");
    }

    [Fact]
    public void GatAllocationIsStillLongOnlyAndFullyInvested()
    {
        var options = new GraphAttentionPortfolioOptions<double> { NumAssets = Assets, CorrelationWindow = 40 };
        var model = new GraphAttentionPortfolio<double>(options);

        var w = model.OptimizePortfolio(Panel(40, Assets, 9));

        for (int i = 0; i < w.Length; i++)
        {
            Assert.False(double.IsNaN(w[i]) || double.IsInfinity(w[i]), $"weight {i} is non-finite.");
            Assert.True(w[i] >= 0.0, $"weight {i} = {w[i]} is short.");
        }
        Assert.Equal(1.0, Sum(w), 8);
    }

    [Fact]
    public void GatBuildsAPlanarGraphFromRealReturns()
    {
        // The graph pipeline end to end: returns -> volatility -> distance correlation -> TMFG.
        var options = new GraphAttentionPortfolioOptions<double>
        { NumAssets = Assets, CorrelationWindow = 60, VolatilityLookback = 10 };
        var model = new GraphAttentionPortfolio<double>(options);

        var edges = model.BuildGraph(Panel(60, Assets, 13));
        Assert.Equal(3 * (Assets - 2), edges.Count);

        var adjacency = model.BuildAdjacency(Panel(60, Assets, 13));
        Assert.Equal(new[] { Assets, Assets }, adjacency.Shape.ToArray());
        for (int i = 0; i < Assets; i++)
            Assert.Equal(1.0, adjacency[(i * Assets) + i], 10);   // self-loops present
    }

    [Fact]
    public void GatRealizedSharpeAndLossAreConsistent()
    {
        var options = new GraphAttentionPortfolioOptions<double> { NumAssets = Assets, CorrelationWindow = 40 };
        var model = new GraphAttentionPortfolio<double>(options);

        var returns = Panel(40, Assets, 21);
        var w = model.OptimizePortfolio(returns);

        double sharpe = model.RealizedSharpe(w, returns);
        double loss = model.PortfolioLoss(w, returns);

        Assert.False(double.IsNaN(sharpe) || double.IsInfinity(sharpe), $"Sharpe is non-finite: {sharpe}");
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"Loss is non-finite: {loss}");

        // loss = -ln(Sharpe) whenever the Sharpe ratio is positive.
        if (sharpe > 0.0) Assert.Equal(-Math.Log(sharpe), loss, 6);
    }

    private static double Max(Vector<double> v)
    {
        double m = double.NegativeInfinity;
        for (int i = 0; i < v.Length; i++) m = Math.Max(m, v[i]);
        return m;
    }

    private static double Min(Vector<double> v)
    {
        double m = double.PositiveInfinity;
        for (int i = 0; i < v.Length; i++) m = Math.Min(m, v[i]);
        return m;
    }
}
