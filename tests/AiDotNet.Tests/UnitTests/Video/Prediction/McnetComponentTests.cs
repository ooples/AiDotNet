using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Video.Prediction;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.Prediction;

/// <summary>
/// Verifies MCnet's two defining components against the paper (Villegas et al., ICLR 2017,
/// arXiv:1706.08033): the asymmetric motion/content decomposition, and the objective's gradient
/// difference term.
/// </summary>
/// <remarks>
/// Each test targets a way the decomposition gets flattened or the loss gets simplified, rather than
/// merely exercising the code.
/// </remarks>
public class McnetComponentTests
{
    private static Tensor<double> Frames(int time, int h, int w, int c, Func<int, int, int, int, double> value)
    {
        var t = new Tensor<double>(new[] { time, h, w, c });
        for (int f = 0; f < time; f++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int ch = 0; ch < c; ch++)
                        t[(((f * h) + y) * w + x) * c + ch] = value(f, y, x, ch);
        return t;
    }

    // ---------------------------------------------------------------- decomposition

    [Fact]
    public void MotionInputIsFrameDifferencesNotRawFrames()
    {
        // f_dyn consumes x_t - x_{t-1}. Frame f has constant value f, so every difference is exactly 1.
        // Feeding raw frames instead would give 0, 1, 2 — and let the motion pathway see appearance.
        var frames = Frames(3, 2, 2, 1, (f, y, x, c) => f);
        var motion = new McnetDecomposition<double>().MotionInput(frames);

        Assert.Equal(2, motion.Shape[0]);   // one fewer than the input
        for (int i = 0; i < motion.Length; i++) Assert.Equal(1.0, motion[i], 10);
    }

    [Fact]
    public void AStaticSceneDifferencesToZeroWhateverItDepicts()
    {
        // The reason differencing decomposes at all: with no motion there is nothing about content for
        // the motion pathway to encode, however rich the scene.
        var frames = Frames(3, 2, 2, 1, (f, y, x, c) => (y * 7) + (x * 13) + 42);
        var motion = new McnetDecomposition<double>().MotionInput(frames);

        for (int i = 0; i < motion.Length; i++) Assert.Equal(0.0, motion[i], 10);
    }

    [Fact]
    public void ContentInputIsTheLastFrameOnly()
    {
        // s_t = f_cont(x_t): a single frame, no recurrence, no history.
        var frames = Frames(4, 2, 2, 1, (f, y, x, c) => f * 10);
        var content = new McnetDecomposition<double>().ContentInput(frames);

        Assert.Equal(3, content.Shape.Length);          // the time axis is gone
        for (int i = 0; i < content.Length; i++) Assert.Equal(30.0, content[i], 10);
    }

    [Fact]
    public void TheTwoPathwaysReceiveDifferentThings()
    {
        // The asymmetry IS the decomposition. If both got the same tensor the encoders would learn
        // overlapping representations, which is the entanglement the paper removes.
        var frames = Frames(3, 2, 2, 1, (f, y, x, c) => f + 1);
        var d = new McnetDecomposition<double>();

        var motion = d.MotionInput(frames);      // differences -> all 1
        var content = d.ContentInput(frames);    // last frame  -> all 3

        Assert.Equal(1.0, motion[0], 10);
        Assert.Equal(3.0, content[0], 10);
    }

    [Fact]
    public void CombineConcatenatesChannelsRatherThanSumming()
    {
        // g_comb([d_t, s_t]) concatenates so the combination layers can weigh the streams separately.
        // Summing would collapse them into one space and discard the separation.
        var motion = Frames(1, 1, 1, 2, (f, y, x, c) => c == 0 ? 1.0 : 2.0);
        var content = Frames(1, 1, 1, 2, (f, y, x, c) => c == 0 ? 3.0 : 4.0);

        var motion3 = new Tensor<double>(new[] { 1, 1, 2 });
        var content3 = new Tensor<double>(new[] { 1, 1, 2 });
        for (int i = 0; i < 2; i++) { motion3[i] = motion[i]; content3[i] = content[i]; }

        var combined = new McnetDecomposition<double>().Combine(motion3, content3);

        Assert.Equal(4, combined.Shape[2]);                       // 2 + 2, not 2
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 },
            new[] { combined[0], combined[1], combined[2], combined[3] });
    }

    [Fact]
    public void MotionInputRejectsASingleFrame()
    {
        var single = Frames(1, 2, 2, 1, (f, y, x, c) => 1.0);
        Assert.Throws<ArgumentException>(() => new McnetDecomposition<double>().MotionInput(single));
    }

    [Fact]
    public void CombineRejectsMismatchedSpatialDimensions()
    {
        var a = new Tensor<double>(new[] { 2, 2, 1 });
        var b = new Tensor<double>(new[] { 3, 3, 1 });
        Assert.Throws<ArgumentException>(() => new McnetDecomposition<double>().Combine(a, b));
    }

    [Fact]
    public void ConstructorRejectsAScalelessConfiguration()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new McnetDecomposition<double>(scales: 0));
    }

    [Fact]
    public void ResidualPairRejectsAnOutOfRangeScale()
    {
        var d = new McnetDecomposition<double>(scales: 2);
        var a = new Tensor<double>(new[] { 1, 1, 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => d.ResidualPair(a, a, scale: 5));
    }

    // ---------------------------------------------------------------- loss

    [Fact]
    public void PixelLossUsesTheSquaredNormByDefault()
    {
        // p = 2. Errors of 1 and 3 give a mean square of 5, not the L1 mean of 2.
        var predicted = new Vector<double>(new[] { 1.0, 3.0 });
        var target = new Vector<double>(new[] { 0.0, 0.0 });

        Assert.Equal(5.0, McnetLoss<double>.Pixel(predicted, target), 10);
        Assert.Equal(2, McnetLoss<double>.PixelNorm);
    }

    [Fact]
    public void GradientDifferenceIsZeroWhenGradientsMatchDespiteAPixelOffset()
    {
        // THE point of the term: it compares GRADIENTS, not pixels. A constant brightness shift leaves
        // every gradient identical, so L_gdl is zero while the pixel loss is large. A gdl implemented
        // on pixels by mistake would report a big number here.
        var target = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
        var shifted = new Vector<double>(new[] { 10.0, 11.0, 12.0, 13.0 });

        Assert.Equal(0.0, McnetLoss<double>.GradientDifference(shifted, target, 2, 2), 10);
        Assert.True(McnetLoss<double>.Pixel(shifted, target) > 50.0,
            "The pixel loss should be large for a constant offset, isolating what gdl ignores.");
    }

    [Fact]
    public void GradientDifferencePenalisesBlur()
    {
        // A sharp step versus a smoothed ramp: identical endpoints, different gradients. This is the
        // blur that minimising pixel error alone produces, and the term exists to punish it.
        var sharp = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });
        var blurred = new Vector<double>(new[] { 0.0, 0.33, 0.67, 1.0 });

        double loss = McnetLoss<double>.GradientDifference(blurred, sharp, 1, 4);
        Assert.True(loss > 0.0, $"Blur should be penalised; got {loss}.");
    }

    [Fact]
    public void GradientDifferenceCoversBothDirections()
    {
        // The paper sums an i-direction and a j-direction term. A purely vertical edge must register,
        // which it cannot if only the i-direction is implemented on a single-row image and vice versa.
        // Here the difference is purely VERTICAL between two rows.
        var target = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });   // 2x2, row step
        var flat = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });

        Assert.True(McnetLoss<double>.GradientDifference(flat, target, 2, 2) > 0.0,
            "A vertical gradient difference must be penalised.");

        // And purely HORIZONTAL.
        var horizontal = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
        Assert.True(McnetLoss<double>.GradientDifference(flat, horizontal, 2, 2) > 0.0,
            "A horizontal gradient difference must be penalised.");
    }

    [Fact]
    public void TotalAppliesAlphaAndBeta()
    {
        // L = alpha*L_img + beta*L_GAN with alpha = 1 and beta = 0.02 (KTH).
        // 1.0*2.0 + 0.02*100 = 4.0
        Assert.Equal(4.0, McnetLoss<double>.Total(imageLoss: 2.0, adversarialLoss: 100.0), 10);
        Assert.Equal(1.0, McnetLoss<double>.ImageWeight, 12);
        Assert.Equal(0.02, McnetLoss<double>.AdversarialWeightKth, 12);
        Assert.Equal(0.001, McnetLoss<double>.AdversarialWeightUcf, 12);
    }

    [Fact]
    public void GeneratorAdversarialIsTheNonSaturatingForm()
    {
        // -log D(G(.)), unlike FIGAN's literal log(1 - D(G(.))). A confident-real judgement should give
        // ~0 penalty and a confident-fake judgement a large one — the opposite sign behaviour to the
        // saturating form, so the two must not be shared.
        Assert.Equal(0.0, McnetLoss<double>.GeneratorAdversarial(1.0), 6);
        Assert.True(McnetLoss<double>.GeneratorAdversarial(1e-6) > 10.0);
    }

    [Fact]
    public void DiscriminatorLossIsLowestWhenCorrectAndConfident()
    {
        double correct = McnetLoss<double>.DiscriminatorLoss(0.99, 0.01);
        double wrong = McnetLoss<double>.DiscriminatorLoss(0.01, 0.99);
        Assert.True(correct < wrong, $"Correct ({correct:F4}) should be below wrong ({wrong:F4}).");
    }

    [Fact]
    public void LossesStayFiniteAtCertainty()
    {
        foreach (double d in new[] { 0.0, 1.0 })
        {
            double g = McnetLoss<double>.GeneratorAdversarial(d);
            Assert.False(double.IsNaN(g) || double.IsInfinity(g), $"Generator loss non-finite at D={d}.");
        }

        double disc = McnetLoss<double>.DiscriminatorLoss(0.0, 1.0);
        Assert.False(double.IsNaN(disc) || double.IsInfinity(disc), "Discriminator loss non-finite.");
    }
}
