using System;
using AiDotNet.Finance.Portfolio;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance.Portfolio;

/// <summary>
/// Verifies SIT's three components against the paper (Yoontae Hwang and Stefan Zohren,
/// "Signature-Informed Transformer for Asset Allocation", arXiv:2510.03129).
/// </summary>
/// <remarks>
/// Each test targets a way the method gets flattened in practice: truncating the signature to level 1
/// (which erases lead-lag entirely), symmetrizing the cross-features (lead-lag is ANTI-symmetric),
/// swapping CVaR for a mean or variance, letting the bias gate go negative, or letting weights go
/// short or not sum to one.
/// </remarks>
public class SignatureInformedTransformerComponentTests
{
    private static Tensor<double> Path(int steps, int dim, Func<int, int, double> value)
    {
        var t = new Tensor<double>(new[] { steps, dim });
        for (int k = 0; k < steps; k++)
            for (int d = 0; d < dim; d++)
                t[(k * dim) + d] = value(k, d);
        return t;
    }

    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) v[i] = values[i];
        return v;
    }

    // ------------------------------------------------------------ path signatures

    [Fact]
    public void TruncationLevelIsTwoBecauseLeadLagLivesInTheSecondOrderTerms()
    {
        Assert.Equal(2, PathSignatureTransform<double>.PaperTruncationLevel);
        Assert.Equal(2, new PathSignatureTransform<double>().Level);

        // Level 1 gives only the d increments; level 2 adds the d*d interaction block. That growth is
        // the reason the paper truncates at 2 rather than going higher.
        Assert.Equal(3, new PathSignatureTransform<double>(1).FeatureCount(3));
        Assert.Equal(3 + 9, new PathSignatureTransform<double>(2).FeatureCount(3));
    }

    [Fact]
    public void LevelsAboveTwoAreRejectedRatherThanSilentlyTruncated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PathSignatureTransform<double>(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PathSignatureTransform<double>(0));
    }

    [Fact]
    public void FirstOrderTermsAreTheTotalIncrement()
    {
        // Level-1 signature of a path is just its endpoint minus its start, so it is blind to the
        // route taken. Two paths with the same endpoints share these terms exactly — which is why
        // level 1 alone cannot express lead-lag.
        var sig = new PathSignatureTransform<double>();

        var direct = Path(2, 2, (k, d) => k == 0 ? 0.0 : (d == 0 ? 3.0 : 4.0));
        var winding = Path(4, 2, (k, d) => d == 0 ? new[] { 0.0, 5.0, 1.0, 3.0 }[k] : new[] { 0.0, -2.0, 6.0, 4.0 }[k]);

        var a = sig.Signature(direct);
        var b = sig.Signature(winding);

        Assert.Equal(3.0, a[0], 10);
        Assert.Equal(4.0, a[1], 10);
        Assert.Equal(a[0], b[0], 10);
        Assert.Equal(a[1], b[1], 10);

        // But their second-order terms differ, because the routes enclose different areas.
        bool secondOrderDiffers = false;
        for (int i = 2; i < a.Length; i++)
            if (Math.Abs(a[i] - b[i]) > 1e-9) { secondOrderDiffers = true; break; }
        Assert.True(secondOrderDiffers,
            "Level-2 terms must distinguish paths that share endpoints; otherwise the signature adds nothing.");
    }

    [Fact]
    public void SecondOrderTermsSatisfyTheShuffleIdentity()
    {
        // S_ij + S_ji == dX_i * dX_j exactly. This is the algebraic identity a signature must obey,
        // and it is what the half-increment correction in the discretization exists to preserve:
        // dropping that correction leaves the antisymmetric part right but the symmetric part wrong.
        var sig = new PathSignatureTransform<double>();
        var path = Path(6, 2, (k, d) => d == 0 ? Math.Sin(k * 0.7) * 2.0 : Math.Cos(k * 0.4) * 1.5);

        var s = sig.Signature(path);
        double dx = s[0], dy = s[1];
        const int dim = 2;
        double sxy = s[dim + (0 * dim) + 1];
        double syx = s[dim + (1 * dim) + 0];

        Assert.Equal(dx * dy, sxy + syx, 8);

        // And the diagonal terms are exactly half the squared increment.
        Assert.Equal(0.5 * dx * dx, s[dim + 0], 8);
        Assert.Equal(0.5 * dy * dy, s[dim + (1 * dim) + 1], 8);
    }

    [Fact]
    public void SignedAreaIsAntisymmetricWhichIsWhatMakesItLeadLag()
    {
        // Area(x,y) = -Area(y,x). A correlation is symmetric and therefore cannot say which asset
        // moved first; the signed area can, and symmetrizing it would throw that away.
        var sig = new PathSignatureTransform<double>();
        var x = Vec(0.0, 1.0, 2.0, 3.0);
        var y = Vec(0.0, 0.0, 1.0, 2.0);   // lags x by one step

        double forward = sig.SignedArea(x, y);
        double backward = sig.SignedArea(y, x);

        Assert.Equal(-forward, backward, 10);
        Assert.True(Math.Abs(forward) > 1e-9, "A genuine lag must produce a non-zero signed area.");
    }

    [Fact]
    public void PerfectlySynchronizedPathsEncloseZeroArea()
    {
        // Two assets moving in exact lockstep trace a straight line in the plane, so they enclose no
        // area and correctly register NO lead-lag — however strongly correlated they are. A measure
        // that reported a large value here would be measuring correlation, not lead-lag.
        var sig = new PathSignatureTransform<double>();
        var x = Vec(0.0, 1.0, 2.0, 3.0, 4.0);
        var y = Vec(0.0, 2.0, 4.0, 6.0, 8.0);   // exactly 2x, zero phase difference

        Assert.Equal(0.0, sig.SignedArea(x, y), 10);
    }

    [Fact]
    public void SignedAreaIsInvariantToAConstantOffset()
    {
        // The area must describe the SHAPE of the joint path, not where it sits. Using absolute
        // coordinates would report a spurious area for a pair that merely trades at high prices.
        var sig = new PathSignatureTransform<double>();
        var x = Vec(0.0, 1.0, 3.0, 2.0);
        var y = Vec(0.0, 2.0, 1.0, 4.0);
        var xShift = Vec(100.0, 101.0, 103.0, 102.0);
        var yShift = Vec(-50.0, -48.0, -49.0, -46.0);

        Assert.Equal(sig.SignedArea(x, y), sig.SignedArea(xShift, yShift), 8);
    }

    [Fact]
    public void CrossAreaMatrixIsAntisymmetricWithAZeroDiagonal()
    {
        var sig = new PathSignatureTransform<double>();
        var panel = Path(6, 3, (k, a) => Math.Sin((k * 0.6) - (a * 0.9)) * (1.0 + a));

        var areas = sig.CrossSignedAreas(panel);
        Assert.Equal(new[] { 3, 3 }, areas.Shape.ToArray());

        for (int j = 0; j < 3; j++)
        {
            // An asset cannot lead itself.
            Assert.Equal(0.0, areas[(j * 3) + j], 12);
            for (int l = 0; l < 3; l++)
                Assert.Equal(areas[(j * 3) + l], -areas[(l * 3) + j], 10);
        }
    }

    [Fact]
    public void SignatureOfASingleSampleIsZeroRatherThanUndefined()
    {
        var sig = new PathSignatureTransform<double>();
        var single = Path(1, 2, (k, d) => 5.0);
        var s = sig.Signature(single);
        for (int i = 0; i < s.Length; i++) Assert.Equal(0.0, s[i], 12);
    }

    [Fact]
    public void SignatureRejectsAWrongRankPath()
    {
        var sig = new PathSignatureTransform<double>();
        Assert.Throws<ArgumentException>(() => sig.Signature(new Tensor<double>(new[] { 4 })));
        Assert.Throws<ArgumentException>(() => sig.CrossSignedAreas(new Tensor<double>(new[] { 2, 2, 2 })));
    }

    // ------------------------------------------------------------ CVaR objective

    [Fact]
    public void LossIsTheNegativePortfolioReturn()
    {
        // L = -(w^T r). Feeding returns un-negated would put the CVaR tail on the PROFITABLE side and
        // train the model to avoid gains.
        var obj = new CVaRPortfolioObjective<double>();
        var w = Vec(0.5, 0.5);
        var r = Vec(0.10, -0.02);

        Assert.Equal(-0.04, obj.Loss(w, r), 10);
    }

    [Fact]
    public void CVaRAveragesTheTailWhileVaRReportsItsBoundary()
    {
        // The distinction the whole objective rests on: VaR is a threshold, CVaR is the mean beyond
        // it. A CVaR that returned the threshold would be indifferent to how bad the worst case is.
        // alpha = 0.6 over 5 samples gives a 2-observation tail, so the averaging is visible.
        // (At alpha = 0.8 the tail is a single observation and CVaR collapses onto the worst loss,
        // which is correct but shows nothing about averaging.)
        var obj = new CVaRPortfolioObjective<double>(0.6);
        var losses = new[] { -0.05, -0.01, 0.0, 0.02, 0.30 };   // one severe loss

        double var60 = obj.ValueAtRisk(losses);
        double cvar60 = obj.ConditionalValueAtRisk(losses);

        Assert.Equal(0.0, var60, 10);             // the 60th-percentile loss: the boundary
        Assert.Equal(0.16, cvar60, 10);           // mean of the worst 40%: {0.02, 0.30}
        Assert.True(cvar60 >= var60, "CVaR can never be below VaR.");

        // The tail is sized by the tail FRACTION, so a tighter alpha narrows it onto the worst loss.
        Assert.Equal(0.30, new CVaRPortfolioObjective<double>(0.8).ConditionalValueAtRisk(losses), 10);
    }

    [Fact]
    public void CVaRIsSensitiveToTailSeverityWhereVaRIsNot()
    {
        // Make the worst outcome far worse and leave everything else alone: VaR does not move, CVaR
        // does. Any objective that fails this is not measuring tail risk.
        var obj = new CVaRPortfolioObjective<double>(0.8);
        var mild = new[] { -0.05, -0.01, 0.0, 0.02, 0.10 };
        var severe = new[] { -0.05, -0.01, 0.0, 0.02, 5.00 };

        Assert.Equal(obj.ValueAtRisk(mild), obj.ValueAtRisk(severe), 10);
        Assert.True(obj.ConditionalValueAtRisk(severe) > obj.ConditionalValueAtRisk(mild) + 0.5,
            "CVaR must react to a far worse tail.");
    }

    [Fact]
    public void TheRockafellarUryasevDualIsMinimizedAtVaRAndMatchesCVaR()
    {
        // This dual is what makes CVaR trainable by gradient descent — it is convex in nu and its
        // hinge is differentiable almost everywhere, whereas sorting for a quantile is not. The
        // minimum value must equal the CVaR computed directly, or the two formulations disagree.
        var obj = new CVaRPortfolioObjective<double>(0.8);
        var losses = new[] { -0.05, -0.01, 0.0, 0.02, 0.30 };

        double nuStar = obj.OptimalNu(losses);
        double atOptimum = obj.DualObjective(losses, nuStar);

        Assert.Equal(obj.ConditionalValueAtRisk(losses), atOptimum, 8);

        // No other nu does better.
        for (double nu = -0.2; nu <= 0.5; nu += 0.01)
            Assert.True(obj.DualObjective(losses, nu) >= atOptimum - 1e-9,
                $"nu={nu} beat the claimed optimum {nuStar}.");
    }

    [Fact]
    public void WeightsAreLongOnlyAndFullyInvestedByConstruction()
    {
        // softmax ENFORCES the constraints rather than encouraging them: no projection step or penalty
        // term is needed, and the constraints cannot be violated mid-training.
        var obj = new CVaRPortfolioObjective<double>();
        var scores = Vec(2.0, -3.0, 0.5, 10.0, -7.0);

        var w = obj.Weights(scores, temperature: 1.3);

        double sum = 0.0;
        for (int i = 0; i < w.Length; i++)
        {
            Assert.True(w[i] > 0.0, $"weight {i} = {w[i]} is not long-only.");
            sum += w[i];
        }
        Assert.Equal(1.0, sum, 10);
    }

    [Fact]
    public void TemperatureControlsConcentration()
    {
        // Lower tau concentrates capital, higher tau spreads it. The paper reports an interior optimum
        // near 1.3 — neither extreme wins, so the knob has to actually do something.
        var obj = new CVaRPortfolioObjective<double>();
        var scores = Vec(3.0, 1.0, 0.0);

        var concentrated = obj.Weights(scores, 0.2);
        var diffuse = obj.Weights(scores, 20.0);

        Assert.True(concentrated[0] > diffuse[0],
            "A lower temperature must put more weight on the top-scoring asset.");
        Assert.True(diffuse[0] - diffuse[2] < concentrated[0] - concentrated[2],
            "A higher temperature must flatten the allocation.");
    }

    [Fact]
    public void WeightsStayFiniteAtExtremeScoresAndTinyTemperature()
    {
        // Without subtracting the max before exponentiating, a small tau overflows exp to infinity and
        // every weight becomes NaN.
        var obj = new CVaRPortfolioObjective<double>();
        var scores = Vec(1000.0, -1000.0, 0.0);

        var w = obj.Weights(scores, 0.01);
        double sum = 0.0;
        for (int i = 0; i < w.Length; i++)
        {
            Assert.False(double.IsNaN(w[i]) || double.IsInfinity(w[i]), $"weight {i} is non-finite.");
            sum += w[i];
        }
        Assert.Equal(1.0, sum, 8);
    }

    [Fact]
    public void ObjectiveRejectsAnInvalidAlphaOrTemperature()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CVaRPortfolioObjective<double>(0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CVaRPortfolioObjective<double>(1.0));

        var obj = new CVaRPortfolioObjective<double>();
        Assert.Throws<ArgumentOutOfRangeException>(() => obj.Weights(Vec(1.0, 2.0), 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => obj.Weights(Vec(1.0, 2.0), double.NaN));
    }

    [Fact]
    public void TurnoverIsHalvedBecauseARebalanceBothSellsAndBuys()
    {
        var obj = new CVaRPortfolioObjective<double>();
        var before = Vec(0.5, 0.5);
        var after = Vec(0.7, 0.3);

        // 0.2 moved from one holding to the other: one-way turnover is 0.2, not 0.4.
        Assert.Equal(0.2, obj.Turnover(before, after), 10);
        Assert.Equal(0.2 * 0.0010, obj.TransactionCost(before, after, 10.0), 12);
        Assert.Equal(0.0, obj.TransactionCost(before, after, 0.0), 12);
    }

    // ------------------------------------------------------------ signature-augmented attention

    [Fact]
    public void GammaIsSoftplusGatedSoTheBiasCanNeverInvert()
    {
        // A negative gate would flip every lead-lag relationship, making the model attend AWAY from
        // assets the evidence says it should attend to. Softplus lets gamma approach zero (ignore the
        // bias) but never cross it.
        var attention = new SignatureAugmentedAttention<double>();

        Assert.Equal(Math.Log(2.0), attention.Gamma, 10);   // softplus(0)

        attention.GammaLogit = -50.0;
        Assert.True(attention.Gamma > 0.0, "gamma must stay strictly positive.");
        Assert.True(attention.Gamma < 1e-10, "a very negative logit should nearly switch the bias off.");

        attention.GammaLogit = 100.0;
        Assert.False(double.IsInfinity(attention.Gamma), "softplus must not overflow for a large logit.");
        Assert.Equal(100.0, attention.Gamma, 6);            // asymptotically linear
    }

    [Fact]
    public void BiasIsTheQueryConditionedInnerProduct()
    {
        // b_{h,j,l} = <q_j^dyn, beta_{j,l}> per head. A static bias read straight off the
        // cross-signature could not let the current market state decide how much a given lead-lag
        // relationship matters right now.
        const int assets = 2, heads = 1, dBeta = 2;
        var q = new Tensor<double>(new[] { assets, heads, dBeta });
        q[0] = 1.0; q[1] = 2.0;      // asset 0
        q[2] = 3.0; q[3] = 4.0;      // asset 1

        var beta = new Tensor<double>(new[] { assets, assets, heads, dBeta });
        // beta[0,1] = (10, 20) -> b[0,1] = 1*10 + 2*20 = 50
        beta[(((0 * assets) + 1) * heads * dBeta) + 0] = 10.0;
        beta[(((0 * assets) + 1) * heads * dBeta) + 1] = 20.0;

        var bias = new SignatureAugmentedAttention<double>().ComputeBias(q, beta);

        Assert.Equal(new[] { heads, assets, assets }, bias.Shape.ToArray());
        Assert.Equal(50.0, bias[(0 * assets * assets) + (0 * assets) + 1], 10);
        Assert.Equal(0.0, bias[(0 * assets * assets) + (0 * assets) + 0], 10);
    }

    [Fact]
    public void BiasIsNotSymmetrized()
    {
        // Lead-lag is directional. If b[j,l] were forced to equal b[l,j] the signal would collapse to
        // something a correlation already expresses.
        const int assets = 2, heads = 1, dBeta = 1;
        var q = new Tensor<double>(new[] { assets, heads, dBeta });
        q[0] = 1.0; q[1] = 1.0;

        var beta = new Tensor<double>(new[] { assets, assets, heads, dBeta });
        beta[((0 * assets) + 1)] = 5.0;      // j=0 -> l=1
        beta[((1 * assets) + 0)] = -5.0;     // j=1 -> l=0, opposite sign

        var bias = new SignatureAugmentedAttention<double>().ComputeBias(q, beta);

        Assert.Equal(5.0, bias[(0 * assets) + 1], 10);
        Assert.Equal(-5.0, bias[(1 * assets) + 0], 10);
    }

    [Fact]
    public void ApplyBiasScalesByGammaAndLeavesLogitsIntactWhenTheGateIsOff()
    {
        var attention = new SignatureAugmentedAttention<double>();
        var logits = new Tensor<double>(new[] { 1, 2, 2 });
        for (int i = 0; i < 4; i++) logits[i] = i;
        var bias = new Tensor<double>(new[] { 1, 2, 2 });
        for (int i = 0; i < 4; i++) bias[i] = 1.0;

        attention.GammaLogit = 0.0;
        var gated = attention.ApplyBias(logits, bias);
        for (int i = 0; i < 4; i++) Assert.Equal(logits[i] + Math.Log(2.0), gated[i], 10);

        // Gate driven to ~0: the logits must pass through essentially untouched, so the model can
        // learn to disregard the signature evidence.
        attention.GammaLogit = -60.0;
        var ungated = attention.ApplyBias(logits, bias);
        for (int i = 0; i < 4; i++) Assert.Equal(logits[i], ungated[i], 10);
    }

    [Fact]
    public void LogitsAreScaledByInverseSqrtOfKeyDimension()
    {
        // Without 1/sqrt(dK) the dot products grow with dK and saturate the softmax, killing gradients.
        var attention = new SignatureAugmentedAttention<double>();
        const int dk = 4;
        var q = new Tensor<double>(new[] { 1, dk });
        var k = new Tensor<double>(new[] { 1, dk });
        for (int d = 0; d < dk; d++) { q[d] = 1.0; k[d] = 1.0; }

        // Raw dot product is dk = 4; scaled it must be 4 / sqrt(4) = 2.
        var logits = attention.ScaledDotProductLogits(q, k);
        Assert.Equal(2.0, logits[0], 10);
    }

    [Fact]
    public void AttentionRejectsMismatchedShapes()
    {
        var attention = new SignatureAugmentedAttention<double>();
        var q = new Tensor<double>(new[] { 2, 1, 2 });
        var badBeta = new Tensor<double>(new[] { 2, 2, 1, 3 });   // dBeta disagrees
        Assert.Throws<ArgumentException>(() => attention.ComputeBias(q, badBeta));

        Assert.Throws<ArgumentException>(() => attention.ScaledDotProductLogits(
            new Tensor<double>(new[] { 2, 3 }), new Tensor<double>(new[] { 2, 4 })));

        Assert.Throws<ArgumentException>(() => attention.ApplyBias(
            new Tensor<double>(new[] { 2, 2 }), new Tensor<double>(new[] { 3, 3 })));
    }
}
