using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Video.FrameInterpolation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.FrameInterpolation;

/// <summary>
/// Verifies FIGAN's two specification-dense components against the paper
/// (van Amersfoort et al., arXiv:1711.06045): the multi-scale residual flow with occlusion-aware
/// synthesis, and the multi-scale perceptual objective.
/// </summary>
/// <remarks>
/// Each test targets a specific way a reimplementation goes wrong — MSE instead of L1, equal scale
/// weights, an unscaled flow upsample, a fixed 0.5 blend — rather than merely exercising the code.
/// </remarks>
public class FiganComponentTests
{
    // ------------------------------------------------------------------ loss

    [Fact]
    public void ContentLossIsL1NotMse()
    {
        // Errors of 1 and 3: L1 mean = 2.0, MSE = 5.0. The distinction is the whole point of Eq. 13,
        // and MSE is the default a reimplementation slips into.
        var predicted = new Vector<double>(new[] { 1.0, 3.0 });
        var target = new Vector<double>(new[] { 0.0, 0.0 });

        Assert.Equal(2.0, FiganLoss<double>.Content(predicted, target), 10);
    }

    [Fact]
    public void ContentLossAddsTheVggTermSquaredAndWeighted()
    {
        // Identical pixels so L1 = 0, isolating the perceptual term. Feature deltas of 2 and 4 give a
        // mean square of 10, times lambda_VGG = 0.001 -> 0.01.
        var same = new Vector<double>(new[] { 1.0, 1.0 });
        var predictedFeatures = new Vector<double>(new[] { 2.0, 4.0 });
        var targetFeatures = new Vector<double>(new[] { 0.0, 0.0 });

        double loss = FiganLoss<double>.Content(same, same, predictedFeatures, targetFeatures);
        Assert.Equal(0.01, loss, 10);
        Assert.Equal(0.001, FiganLoss<double>.VggWeight, 12);
    }

    [Fact]
    public void TotalWeightsTheFinestScaleAboveTheCoarserOnes()
    {
        // Eq. 12: finest at 1.0, the two coarser at 0.5, refine at 1.0, GAN at 1e-4.
        // 1.0 + 0.5*(2.0 + 4.0) + 0.25 + 1e-4*100 = 1.0 + 3.0 + 0.25 + 0.01 = 4.26
        double total = FiganLoss<double>.Total(
            finestScale: 1.0,
            coarserScales: new[] { 2.0, 4.0 },
            refinedSynthesis: 0.25,
            adversarial: 100.0);

        Assert.Equal(4.26, total, 10);
    }

    [Fact]
    public void AdversarialWeightIsThreeOrdersBelowTheContentTerms()
    {
        // A co-equal adversarial term destabilises training; the paper deliberately keeps it tiny.
        Assert.Equal(0.0001, FiganLoss<double>.AdversarialWeight, 12);
        Assert.True(FiganLoss<double>.AdversarialWeight < FiganLoss<double>.VggWeight,
            "The GAN weight must be smaller than even the VGG weight.");
    }

    [Fact]
    public void GeneratorAdversarialStaysFiniteWhenTheDiscriminatorIsCertain()
    {
        // log(1 - D) diverges at D = 1. Clamping keeps a confident discriminator from poisoning the
        // total loss with negative infinity.
        double loss = FiganLoss<double>.GeneratorAdversarial(1.0);
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"Non-finite: {loss}");
    }

    [Fact]
    public void DiscriminatorObjectiveRewardsCorrectDiscrimination()
    {
        // It MAXIMISES log D(real) + log(1 - D(fake)), so confident-and-correct must score higher
        // than confident-and-wrong.
        double correct = FiganLoss<double>.DiscriminatorObjective(0.99, 0.01);
        double wrong = FiganLoss<double>.DiscriminatorObjective(0.01, 0.99);
        Assert.True(correct > wrong, $"Correct ({correct:F4}) should exceed wrong ({wrong:F4}).");
    }

    // ------------------------------------------------------------------ multi-scale flow

    private static Tensor<double> Frame(int h, int w, Func<int, int, double> value)
    {
        var t = new Tensor<double>(new[] { h, w, 1 });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) t[(y * w) + x] = value(y, x);
        return t;
    }

    private static Tensor<double> Flow(int h, int w, double dx, double dy, double occlusion)
    {
        var t = new Tensor<double>(new[] { h, w, FiganMultiScaleFlow<double>.FlowChannels });
        for (int i = 0; i < h * w; i++)
        {
            t[(i * 3) + 0] = dx;
            t[(i * 3) + 1] = dy;
            t[(i * 3) + 2] = occlusion;
        }
        return t;
    }

    [Fact]
    public void GammaCarriesThreeChannelsFlowPlusOcclusion()
    {
        // Table 1 gives every flow module N_o = 3, and the residual module N_i = 9 = 3 + 3 + 3. Reading
        // Gamma as pure 2-channel flow leaves those widths unexplainable and W with no source.
        Assert.Equal(3, FiganMultiScaleFlow<double>.FlowChannels);
    }

    [Fact]
    public void DownsampleAveragesRatherThanSubsamples()
    {
        // A 2x2 block of 0,1,2,3 must average to 1.5. Subsampling would return 0 and alias the motion
        // the coarse level is supposed to summarise.
        var frame = Frame(2, 2, (y, x) => (y * 2) + x);
        var coarse = new FiganMultiScaleFlow<double>().Downsample(frame, 1);

        Assert.Equal(1, coarse.Shape[0]);
        Assert.Equal(1, coarse.Shape[1]);
        Assert.Equal(1.5, coarse[0], 10);
    }

    [Fact]
    public void UpsampleScalesDisplacementsButNotTheOcclusionWeight()
    {
        // One pixel of motion at a coarse level is TWO at the next. Forgetting this silently halves
        // motion at every level — the classic coarse-to-fine flow bug. W is a weight, so it must NOT
        // be scaled.
        var coarse = Flow(1, 1, dx: 1.0, dy: 2.0, occlusion: 0.7);
        var fine = new FiganMultiScaleFlow<double>().UpsampleFlow(coarse, 2, 2);

        Assert.Equal(2.0, fine[0], 10);   // dx doubled
        Assert.Equal(4.0, fine[1], 10);   // dy doubled
        Assert.Equal(0.7, fine[2], 10);   // W unchanged
    }

    [Fact]
    public void WarpShiftsTheFrameInOppositeDirectionsForTheTwoInputs()
    {
        // I_0(-Delta) and I_1(Delta): the two frames move TOWARDS each other. A ramp makes the
        // direction observable.
        var ramp = Frame(1, 4, (y, x) => x);
        var flow = Flow(1, 4, dx: 1.0, dy: 0.0, occlusion: 0.0);
        var module = new FiganMultiScaleFlow<double>();

        var forward = module.Warp(ramp, flow, direction: -1);
        var backward = module.Warp(ramp, flow, direction: +1);

        // Sampling at x-1 vs x+1 gives strictly lower vs higher values away from the borders.
        Assert.True(forward[2] < ramp[2], "direction -1 should sample from earlier in the ramp.");
        Assert.True(backward[1] > ramp[1], "direction +1 should sample from later in the ramp.");
    }

    [Fact]
    public void SynthesisIsNotAFixedHalfBlend()
    {
        // The occlusion weight decides per pixel which warped frame to trust. With W driven hard one
        // way the result must follow that frame, not sit at the midpoint.
        var dark = Frame(1, 2, (y, x) => 0.0);
        var bright = Frame(1, 2, (y, x) => 1.0);
        var module = new FiganMultiScaleFlow<double>();

        var towardsFirst = module.Synthesise(dark, bright, Flow(1, 2, 0, 0, occlusion: 10.0));
        var towardsSecond = module.Synthesise(dark, bright, Flow(1, 2, 0, 0, occlusion: -10.0));

        Assert.True(towardsFirst[0] < 0.01, $"W high should follow frame0 (dark); got {towardsFirst[0]}.");
        Assert.True(towardsSecond[0] > 0.99, $"W low should follow frame1 (bright); got {towardsSecond[0]}.");
    }

    [Fact]
    public void SynthesisWithNeutralOcclusionAveragesTheFrames()
    {
        // W = 0 through the logistic gives 0.5, the naive baseline — a useful sanity anchor showing the
        // blend is a convex combination.
        var dark = Frame(1, 2, (y, x) => 0.0);
        var bright = Frame(1, 2, (y, x) => 1.0);

        var mid = new FiganMultiScaleFlow<double>()
            .Synthesise(dark, bright, Flow(1, 2, 0, 0, occlusion: 0.0));

        Assert.Equal(0.5, mid[0], 10);
    }

    [Fact]
    public void RefineAppliesTanhToTheSumNotTheResidual()
    {
        // tanh(Gamma + Gamma_res). With Gamma = 0.5 and residual = 0.5 the answer is tanh(1.0);
        // applying tanh to the residual alone would give 0.5 + tanh(0.5) and leave the accumulated
        // flow unbounded across scales.
        var flow = Flow(1, 1, 0.5, 0.5, 0.5);
        var residual = Flow(1, 1, 0.5, 0.5, 0.5);

        var refined = new FiganMultiScaleFlow<double>().Refine(flow, residual);

        Assert.Equal(Math.Tanh(1.0), refined[0], 10);
        Assert.NotEqual(0.5 + Math.Tanh(0.5), refined[0], 6);
    }

    [Fact]
    public void ConstructorRejectsAScalelessConfiguration()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FiganMultiScaleFlow<double>(scales: 0));
    }

    [Fact]
    public void WarpRejectsAFlowThatDoesNotMatchTheFrame()
    {
        // Warping with a coarser flow than the frame silently misaligns every sample, so it must be
        // upsampled first.
        var frame = Frame(4, 4, (y, x) => 1.0);
        var coarseFlow = Flow(2, 2, 0, 0, 0);

        Assert.Throws<ArgumentException>(
            () => new FiganMultiScaleFlow<double>().Warp(frame, coarseFlow, direction: -1));
    }
}
