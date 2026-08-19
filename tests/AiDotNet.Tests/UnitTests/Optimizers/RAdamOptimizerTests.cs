using System;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins <see cref="RAdamOptimizer{T, TInput, TOutput}"/> to Algorithm 2 of "On the Variance of the Adaptive
/// Learning Rate and Beyond" (Liu et al., ICLR 2020, arXiv:1908.03265).
/// </summary>
/// <remarks>
/// <para>
/// RAdam is easy to get wrong in a way no smoke test catches: drop the un-rectified branch and you are left
/// with something that still trains, still converges, and is simply Adam again — losing the one property the
/// paper exists to provide. So these tests target the branch structure directly rather than asserting that
/// loss goes down.
/// </para>
/// <para>
/// The reference below is an independent transcription of Algorithm 2, written from the paper rather than from
/// the implementation under test, so agreement between the two is evidence rather than tautology.
/// </para>
/// </remarks>
public class RAdamOptimizerTests
{
    private const double Lr = 0.01;
    private const double Beta1 = 0.9;
    private const double Beta2 = 0.999;
    private const double Eps = 1e-8;

    /// <summary>
    /// Independent scalar transcription of Algorithm 2, applied to one parameter for <paramref name="steps"/>
    /// iterations under a constant gradient.
    /// </summary>
    private static double ReferenceRAdam(double theta, double grad, int steps,
                                         double lr = Lr, double beta1 = Beta1, double beta2 = Beta2, double eps = Eps)
    {
        double m = 0.0, v = 0.0;
        double rhoInf = 2.0 / (1.0 - beta2) - 1.0;

        for (int t = 1; t <= steps; t++)
        {
            m = beta1 * m + (1.0 - beta1) * grad;
            v = beta2 * v + (1.0 - beta2) * grad * grad;

            double bc1 = 1.0 - Math.Pow(beta1, t);
            double bc2 = 1.0 - Math.Pow(beta2, t);
            double mHat = m / bc1;
            double rhoT = rhoInf - 2.0 * t * Math.Pow(beta2, t) / bc2;

            if (rhoT > 4.0)
            {
                double rt = Math.Sqrt(((rhoT - 4.0) * (rhoT - 2.0) * rhoInf) /
                                      ((rhoInf - 4.0) * (rhoInf - 2.0) * rhoT));
                double lt = 1.0 / (Math.Sqrt(v / bc2) + eps);
                theta -= lr * rt * mHat * lt;
            }
            else
            {
                theta -= lr * mHat;
            }
        }

        return theta;
    }

    private static double RhoAt(int t, double beta2 = Beta2)
        => (2.0 / (1.0 - beta2) - 1.0) - 2.0 * t * Math.Pow(beta2, t) / (1.0 - Math.Pow(beta2, t));

    private static RAdamOptimizer<double, Matrix<double>, Vector<double>> CreateOptimizer(
        double lr = Lr, bool adaptiveLr = false)
        => new(null, new RAdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = lr,
            Beta1 = Beta1,
            Beta2 = Beta2,
            Epsilon = Eps,
            UseAdaptiveLearningRate = adaptiveLr,
        });

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(12)]
    public void TapeStep_MatchesAlgorithm2AcrossBothBranches(int steps)
    {
        var optimizer = CreateOptimizer();
        var parameter = new Tensor<double>(new[] { 3 });
        new[] { 0.5, -0.25, 3.0 }.CopyTo(parameter.AsWritableSpan());
        var gradient = new Tensor<double>(new[] { 3 });
        new[] { 0.1, -0.4, 2.0 }.CopyTo(gradient.AsWritableSpan());
        var gradients = new Dictionary<Tensor<double>, Tensor<double>> { [parameter] = gradient };

        for (int step = 0; step < steps; step++)
        {
            optimizer.Step(new TapeStepContext<double>(new[] { parameter }, gradients, 0.0));
        }

        Assert.Equal(ReferenceRAdam(0.5, 0.1, steps), parameter.AsSpan()[0], 10);
        Assert.Equal(ReferenceRAdam(-0.25, -0.4, steps), parameter.AsSpan()[1], 10);
        Assert.Equal(ReferenceRAdam(3.0, 2.0, steps), parameter.AsSpan()[2], 10);
    }

    /// <summary>
    /// At the paper's default beta2 = 0.999 the SMA estimate rho_t crosses the threshold of 4 between t = 4 and
    /// t = 5. This is the fact the next two tests rest on, so it is asserted directly rather than assumed.
    /// </summary>
    [Fact]
    public void RhoCrossesTheRectificationThresholdBetweenStepsFourAndFive()
    {
        Assert.True(RhoAt(1) <= 4.0, $"rho_1 = {RhoAt(1)}");
        Assert.True(RhoAt(4) <= 4.0, $"rho_4 = {RhoAt(4)}");
        Assert.True(RhoAt(5) > 4.0, $"rho_5 = {RhoAt(5)}");
    }

    /// <summary>
    /// The un-rectified branch is a plain bias-corrected momentum step, so the FIRST step must be
    /// lr * g exactly — linear in the gradient.
    /// </summary>
    /// <remarks>
    /// This is the sharpest available separation from Adam. Adam's first step is lr * mHat / (sqrt(vHat) + eps),
    /// and at t = 1 that ratio is sign(g) regardless of magnitude — so Adam moves by ~lr whether the gradient is
    /// 1 or 1000. RAdam moves by lr * g. Feeding a large gradient makes the two differ by three orders of
    /// magnitude, which no tolerance can paper over.
    /// </remarks>
    [Fact]
    public void FirstStepIsPlainMomentum_ScalingLinearlyWithTheGradient()
    {
        var small = CreateOptimizer();
        var large = CreateOptimizer();

        var afterSmall = small.UpdateParameters(
            new Vector<double>(new[] { 0.0 }), new Vector<double>(new[] { 1.0 }));
        var afterLarge = large.UpdateParameters(
            new Vector<double>(new[] { 0.0 }), new Vector<double>(new[] { 1000.0 }));

        // Step 1: m = 0.1*g, mHat = m/(1-0.9) = g, so the update is exactly lr*g.
        Assert.Equal(-Lr * 1.0, afterSmall[0], 12);
        Assert.Equal(-Lr * 1000.0, afterLarge[0], 9);

        // And therefore 1000x apart — an Adam-shaped step would put them within a factor of ~1.
        Assert.Equal(1000.0, afterLarge[0] / afterSmall[0], 6);
    }

    /// <summary>
    /// Step 5 is the first rectified step, so it must be bounded by the adaptive term rather than growing with
    /// the gradient the way steps 1-4 do.
    /// </summary>
    [Fact]
    public void FifthStepIsRectified_AndNoLongerScalesWithGradientMagnitude()
    {
        var small = CreateOptimizer();
        var large = CreateOptimizer();

        Vector<double> pSmall = new(new[] { 0.0 }), pLarge = new(new[] { 0.0 });
        Vector<double> gSmall = new(new[] { 1.0 }), gLarge = new(new[] { 1000.0 });

        double lastSmallStep = 0.0, lastLargeStep = 0.0;
        for (int t = 1; t <= 5; t++)
        {
            var nextSmall = small.UpdateParameters(pSmall, gSmall);
            var nextLarge = large.UpdateParameters(pLarge, gLarge);
            lastSmallStep = nextSmall[0] - pSmall[0];
            lastLargeStep = nextLarge[0] - pLarge[0];
            pSmall = nextSmall;
            pLarge = nextLarge;
        }

        // On the rectified branch the gradient magnitude cancels out of mHat/sqrt(vHat), so a 1000x larger
        // gradient produces a step of essentially the same size — the opposite of step 1's behaviour.
        Assert.Equal(lastSmallStep, lastLargeStep, 6);
    }

    /// <summary>
    /// Element-wise agreement with the independent transcription of Algorithm 2 across steps that span BOTH
    /// branches (1-4 un-rectified, 5-12 rectified).
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(12)]
    public void MatchesAlgorithm2_AcrossBothBranches(int steps)
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 0.5, -0.25, 3.0 });
        var gradient = new Vector<double>(new[] { 0.1, -0.4, 2.0 });

        for (int t = 1; t <= steps; t++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        Assert.Equal(ReferenceRAdam(0.5, 0.1, steps), parameters[0], 10);
        Assert.Equal(ReferenceRAdam(-0.25, -0.4, steps), parameters[1], 10);
        Assert.Equal(ReferenceRAdam(3.0, 2.0, steps), parameters[2], 10);
    }

    /// <summary>
    /// The fused kernel must be told RAdam, not Adam. Reporting Adam would silently discard the rectification
    /// and the warmup branch on the compiled path only — the fused/eager divergence class of bug this PR exists
    /// to close.
    /// </summary>
    [Fact]
    public void FusedSpecReportsRAdamWithThePaperHyperparameters()
    {
        var optimizer = CreateOptimizer();

        Assert.True(((IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.RAdam, config.Type);
        Assert.Equal((float)Beta1, config.Beta1, 5);
        Assert.Equal((float)Beta2, config.Beta2, 5);
        Assert.Equal((float)Eps, config.Epsilon, 10);
        Assert.Equal(0f, config.WeightDecay);
    }

    /// <summary>
    /// An outer adaptive learning rate changes lr after the plan has baked it in, so the spec must decline —
    /// matching the Adam/AdaMax/AMSGrad specs.
    /// </summary>
    [Fact]
    public void FusedSpecDeclines_WhenAnOuterAdaptiveLearningRateIsConfigured()
    {
        var optimizer = CreateOptimizer(adaptiveLr: true);

        Assert.False(((IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out _),
            "RAdam fused despite an adaptive learning rate the compiled plan cannot follow.");
    }

    /// <summary>
    /// The step counter drives the rectification, so a round-trip that loses it would silently restart training
    /// in the un-rectified warmup phase.
    /// </summary>
    [Fact]
    public void SerializationRoundTripPreservesTheStepCounter()
    {
        var original = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 1.0 });
        var gradient = new Vector<double>(new[] { 0.3 });

        for (int t = 1; t <= 6; t++)
        {
            parameters = original.UpdateParameters(parameters, gradient);
        }

        var restored = CreateOptimizer();
        restored.Deserialize(original.Serialize());

        // Step 7 lands on the rectified branch for both only if the restored optimizer resumed at t = 6.
        var fromOriginal = original.UpdateParameters(parameters, gradient);
        var fromRestored = restored.UpdateParameters(parameters, gradient);

        Assert.Equal(fromOriginal[0], fromRestored[0], 12);
        Assert.Equal(ReferenceRAdam(1.0, 0.3, 7), fromOriginal[0], 10);
    }
}
