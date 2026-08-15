using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.LossFunctions;

/// <summary>
/// Contract tests for the three reusable adversarial/flow components that Stream-DiffVSR's objectives
/// are built from: <see cref="PatchGANDiscriminator{T}"/>, <see cref="AdversarialLoss{T}"/> and
/// <see cref="FlowLoss{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// The load-bearing assertions here are the GRADIENT ones. A GAN term and a flow term are only real if
/// the discriminator's and flow estimator's forward passes are part of the differentiated graph — a
/// loss assembled from raw buffer arithmetic returns a plausible number while sending the generator no
/// signal whatsoever, which is indistinguishable from having no such term. These tests fail if that
/// regresses.
/// </para>
/// <para>
/// Architectural facts from pix2pix (Isola et al., 2017) section 6.1.2 are asserted structurally, so
/// they cost no forward pass and can use the paper's real 64-filter defaults. The forward and gradient
/// tests use a small filter count purely to stay fast; filter count does not affect the properties
/// being checked (receptive field, grid shape, gradient reachability).
/// </para>
/// </remarks>
public class PatchGANAdversarialFlowLossTests
{
    private readonly ITestOutputHelper _out;

    public PatchGANAdversarialFlowLossTests(ITestOutputHelper output) => _out = output;

    private static Tensor<double> Filled(int[] shape, double value)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value;
        return t;
    }

    /// <summary>Deterministic, non-constant fill so convolutions see real spatial structure.</summary>
    private static Tensor<double> Ramp(int[] shape, double scale = 1.0, int seed = 0)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            // A bounded, non-periodic-looking pattern; no RNG so failures reproduce exactly.
            t[i] = scale * Math.Sin((i + 1 + seed) * 0.7);
        }

        return t;
    }

    private static bool Fin(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    /// <summary>
    /// Largest absolute gradient recorded for <paramref name="target"/>, or 0 when the tape holds no
    /// gradient for it at all.
    /// </summary>
    /// <remarks>
    /// The tape OMITS tensors it found no gradient path to, rather than mapping them to null, so
    /// indexing the dictionary directly throws instead of reporting "no gradient". Distinguishing
    /// "absent" from "present but zero" is the whole point of these tests, so both collapse to 0 here
    /// and the assertions speak about magnitude.
    /// </remarks>
    private static double MaxAbsGradient(
        IDictionary<Tensor<double>, Tensor<double>> grads, Tensor<double> target)
    {
        if (!grads.TryGetValue(target, out var g) || g is null) return 0;

        double maxAbs = 0;
        for (int i = 0; i < g.Length; i++)
        {
            if (Fin(g[i])) maxAbs = Math.Max(maxAbs, Math.Abs(g[i]));
        }

        return maxAbs;
    }

    #region PatchGAN architecture (paper section 6.1.2)

    /// <summary>
    /// The whole point of varying discriminator DEPTH is to hit specific receptive-field sizes. If the
    /// stride schedule or padding drifts, these numbers stop matching the paper's table and the model
    /// is no longer the discriminator it claims to be.
    /// </summary>
    [Theory]
    [InlineData(PatchGANReceptiveField.Pixel1x1, 1)]
    [InlineData(PatchGANReceptiveField.Patch16x16, 16)]
    [InlineData(PatchGANReceptiveField.Patch70x70, 70)]
    [InlineData(PatchGANReceptiveField.Image286x286, 286)]
    public void ReceptiveField_MatchesPaperTable(PatchGANReceptiveField field, int expected)
    {
        var d = new PatchGANDiscriminator<double>(field);
        _out.WriteLine($"{field}: numLayers={d.NumLayers} receptiveField={d.ReceptiveField}");
        Assert.Equal(expected, d.ReceptiveField);
    }

    /// <summary>
    /// "As an exception to the above notation, BatchNorm is not applied to the first C64 layer."
    /// </summary>
    [Fact]
    public void FirstBlock_HasNoBatchNorm_AllOthersDo()
    {
        var d = new PatchGANDiscriminator<double>(PatchGANReceptiveField.Patch70x70);
        int norms = d.GetSubLayers().Count(l => l is BatchNormalizationLayer<double>);

        // 4 Ck blocks, and only blocks 1..3 are normalized.
        _out.WriteLine($"numLayers={d.NumLayers} batchNormCount={norms}");
        Assert.Equal(d.NumLayers - 1, norms);
    }

    /// <summary>
    /// A PatchGAN emits a GRID of verdicts, one channel deep. Losing that (e.g. collapsing to a single
    /// scalar) would silently turn it into a whole-image discriminator.
    /// </summary>
    [Fact]
    public void Forward_ProducesSingleChannelPatchGrid()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 3, numFilters: 4, applySigmoid: false);
        var img = Ramp([3, 32, 32]);

        var verdict = d.Forward(img);
        _out.WriteLine($"verdict shape = [{string.Join(",", verdict.Shape.ToArray())}]");

        Assert.Equal(3, verdict.Shape.Length);
        Assert.Equal(1, verdict.Shape[0]);          // one channel: real-vs-fake per patch
        Assert.True(verdict.Shape[1] > 1 && verdict.Shape[2] > 1,
            "Output must remain a spatial grid of patch verdicts, not a single scalar.");
        for (int i = 0; i < verdict.Length; i++) Assert.True(Fin(verdict[i]));
    }

    /// <summary>
    /// Being fully convolutional, the same discriminator must accept a larger image and simply produce
    /// a larger grid — that is what lets one discriminator train at any patch size.
    /// </summary>
    [Fact]
    public void Forward_LargerInput_ProducesLargerGrid()
    {
        var small = new PatchGANDiscriminator<double>(numLayers: 3, numFilters: 4, applySigmoid: false);
        var big = new PatchGANDiscriminator<double>(numLayers: 3, numFilters: 4, applySigmoid: false);

        var vSmall = small.Forward(Ramp([3, 32, 32]));
        var vBig = big.Forward(Ramp([3, 64, 64]));

        _out.WriteLine($"32x32 -> [{string.Join(",", vSmall.Shape.ToArray())}]; " +
                       $"64x64 -> [{string.Join(",", vBig.Shape.ToArray())}]");
        Assert.True(vBig.Shape[1] > vSmall.Shape[1]);
        Assert.True(vBig.Shape[2] > vSmall.Shape[2]);
    }

    /// <summary>
    /// The flat parameter and gradient vectors are built from one shared ordering. If they disagree in
    /// LENGTH, an optimizer would apply one sub-layer's gradient to another's weights.
    /// </summary>
    [Fact]
    public void ParameterAndGradientVectors_HaveMatchingLengths()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 3, numFilters: 4, applySigmoid: false);
        d.Forward(Ramp([3, 32, 32]));   // resolve lazy shapes so weights exist

        var p = d.GetParameters();
        var g = d.GetParameterGradients();
        _out.WriteLine($"parameters={p.Length} gradients={g.Length}");

        Assert.True(p.Length > 0, "A discriminator with no parameters cannot learn.");
        Assert.Equal(p.Length, g.Length);
    }

    /// <summary>
    /// Round-tripping parameters must be exact; otherwise Clone and deserialize silently produce a
    /// different discriminator.
    /// </summary>
    [Fact]
    public void SetParameters_RoundTripsExactly()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 2, numFilters: 4, applySigmoid: false);
        d.Forward(Ramp([3, 32, 32]));

        var original = d.GetParameters();
        var modified = new Vector<double>(original.Length);
        for (int i = 0; i < original.Length; i++) modified[i] = original[i] + 0.125;

        d.SetParameters(modified);
        var readBack = d.GetParameters();

        Assert.Equal(modified.Length, readBack.Length);
        for (int i = 0; i < modified.Length; i++) Assert.Equal(modified[i], readBack[i], 12);
    }

    [Fact]
    public void SetParameters_WrongLength_Throws()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 2, numFilters: 4, applySigmoid: false);
        d.Forward(Ramp([3, 32, 32]));
        Assert.Throws<ArgumentException>(() => d.SetParameters(new Vector<double>(3)));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NonPositiveDepth_Throws(int numLayers)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new PatchGANDiscriminator<double>(numLayers: numLayers));
    }

    #endregion

    #region AdversarialLoss

    /// <summary>
    /// THE test for this component: the generator's adversarial gradient must actually reach the
    /// generated image through the discriminator. A non-zero, finite gradient here is the only proof
    /// that the discriminator's forward pass is on the tape and the GAN term is real.
    /// </summary>
    [Fact]
    public void GeneratorLoss_GradientReachesGeneratedImage()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 2, numFilters: 4, applySigmoid: false);
        var adv = new AdversarialLoss<double>(d);
        var fake = Ramp([3, 32, 32]);

        using var tape = new GradientTape<double>();
        var loss = adv.ComputeTapeLoss(fake, fake);
        var grads = tape.ComputeGradients(loss, new[] { fake });

        // THROUGH THE HELPER, so the failure produces the message below rather than a
        // KeyNotFoundException. The remarks at the top of this file record that the tape omits
        // tensors it found no gradient path to -- which is precisely the discriminator-off-the-tape
        // case this assertion exists to explain -- so indexing grads[fake] directly threw before
        // reaching the explanation.
        double maxAbs = MaxAbsGradient(grads, fake);

        int nonFinite = 0;
        if (grads.TryGetValue(fake, out var g) && g is not null)
        {
            for (int i = 0; i < g.Length; i++) if (!Fin(g[i])) nonFinite++;
        }

        _out.WriteLine($"loss={loss[0]} grad max|.|={maxAbs} nonFinite={nonFinite}");
        Assert.Equal(0, nonFinite);
        Assert.True(maxAbs > 0,
            "Adversarial gradient w.r.t. the generated image is identically zero — the discriminator " +
            "is not on the autodiff tape, so this GAN term would train nothing.");
    }

    /// <summary>
    /// The discriminator step must NOT push gradient back into the generator: that would train the
    /// generator to make its own output easier to detect. StopGradient on the generated branch is what
    /// prevents it.
    /// </summary>
    [Fact]
    public void DiscriminatorLoss_DoesNotBackpropIntoGeneratedImage()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 2, numFilters: 4, applySigmoid: false);
        var adv = new AdversarialLoss<double>(d);
        var fake = Ramp([3, 32, 32], seed: 1);
        var real = Ramp([3, 32, 32], seed: 7);

        using var tape = new GradientTape<double>();
        var loss = adv.ComputeDiscriminatorTapeLoss(fake, real);
        var grads = tape.ComputeGradients(loss, new[] { fake, real });

        // A tensor with no gradient path is ABSENT from the dictionary rather than mapped to null, so
        // absence is itself the assertion being made here: StopGradient severed the generated branch.
        double fakeMax = MaxAbsGradient(grads, fake);
        double realMax = MaxAbsGradient(grads, real);

        _out.WriteLine($"discriminator step: grad max|.| fake={fakeMax} real={realMax}");

        // BOTH HALVES OF THE CONTRACT. The discriminator step requires that gradient does NOT reach
        // the generated branch AND that it DOES reach the real one. realMax was computed, printed,
        // and never asserted, so a regression that severed BOTH branches -- the whole discriminator
        // falling off the tape -- satisfied the only assertion here and passed.
        Assert.Equal(0.0, fakeMax);
        Assert.True(realMax > 0,
            "Gradient did not reach the real branch either (max|.| = 0), so the discriminator is not " +
            "on the tape at all. fakeMax being 0 then says nothing about StopGradient severing the " +
            "generated branch, which is what this test claims to verify.");
    }

    /// <summary>
    /// Softplus is evaluated through the shifted identity precisely so extreme logits cannot overflow
    /// or take log(0). Large-magnitude inputs drive large logits, which is where the naive
    /// <c>log(1 + exp(x))</c> would produce Inf/NaN.
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(1e3)]
    [InlineData(1e6)]
    public void GeneratorLoss_StaysFiniteAndNonNegative_AtExtremeScales(double scale)
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 2, numFilters: 4, applySigmoid: false);
        var adv = new AdversarialLoss<double>(d);

        var loss = adv.ComputeTapeLoss(Ramp([3, 32, 32], scale), Filled([3, 32, 32], 0.0));

        _out.WriteLine($"scale={scale:g} loss={loss[0]}");
        Assert.True(Fin(loss[0]), $"Loss was non-finite at input scale {scale}.");
        Assert.True(loss[0] >= 0.0, "softplus is non-negative, so this loss must be too.");
    }

    /// <summary>
    /// The probability path clamps before the logarithm, so a discriminator that confidently outputs 0
    /// must still yield a finite (large) loss rather than Inf.
    /// </summary>
    [Fact]
    public void ProbabilityPath_HandlesSaturatedOutputWithoutInfinity()
    {
        var d = new PatchGANDiscriminator<double>(numLayers: 2, numFilters: 4, applySigmoid: true);
        var adv = new AdversarialLoss<double>(d, discriminatorOutputsProbabilities: true);

        // A hugely negative input saturates the sigmoid towards 0, i.e. -log(0) without clamping.
        var loss = adv.ComputeTapeLoss(Filled([3, 32, 32], -1e6), Filled([3, 32, 32], 0.0));

        _out.WriteLine($"saturated probability loss={loss[0]}");
        Assert.True(Fin(loss[0]), "Saturated sigmoid must be clamped away from 0 before log.");
        Assert.True(loss[0] >= 0.0);
    }

    #endregion

    #region FlowLoss

    /// <summary>
    /// Two identical clips have identical motion, so the term must vanish. Anything else means the
    /// estimator is being applied inconsistently between the two branches.
    /// </summary>
    [Fact]
    public void IdenticalClips_GiveZeroFlowLoss()
    {
        var loss = new FlowLoss<double>();
        var clip = Ramp([2, 3, 16, 16]);

        var value = loss.ComputeTapeLoss(clip, clip);
        _out.WriteLine($"identical-clip flow loss = {value[0]}");
        Assert.True(Fin(value[0]));
        Assert.Equal(0.0, value[0], 10);
    }

    /// <summary>
    /// Different motion must produce a strictly positive penalty, otherwise the term is inert.
    /// </summary>
    [Fact]
    public void DifferentMotion_GivesPositiveFlowLoss()
    {
        var loss = new FlowLoss<double>();
        var predicted = Ramp([2, 3, 16, 16], seed: 1);
        var target = Ramp([2, 3, 16, 16], seed: 9);

        var value = loss.ComputeTapeLoss(predicted, target);
        _out.WriteLine($"differing-motion flow loss = {value[0]}");
        Assert.True(Fin(value[0]));
        Assert.True(value[0] > 0.0);
    }

    /// <summary>
    /// The flow term must be differentiable with respect to the RECONSTRUCTION, since that is the only
    /// way it can influence training. This also proves the flow estimator itself is built from
    /// tape-recorded engine operations rather than raw buffer writes.
    /// </summary>
    [Fact]
    public void FlowLoss_GradientReachesPredictedClip()
    {
        var loss = new FlowLoss<double>();
        var predicted = Ramp([2, 3, 16, 16], seed: 1);
        var target = Ramp([2, 3, 16, 16], seed: 9);

        using var tape = new GradientTape<double>();
        var value = loss.ComputeTapeLoss(predicted, target);
        var grads = tape.ComputeGradients(value, new[] { predicted });

        bool present = grads.TryGetValue(predicted, out var g) && g is not null;
        double maxAbs = MaxAbsGradient(grads, predicted);

        _out.WriteLine($"flow loss={value[0]} gradientPresent={present} grad max|.|={maxAbs}");
        Assert.True(maxAbs > 0,
            "The flow term produces NO gradient with respect to the reconstruction " +
            $"(gradient present on tape: {present}). The optical-flow estimator's forward pass is not " +
            "recorded — RAFT.EstimateFlow routes through Predict(), the inference entry point — so this " +
            "term contributes a loss value but trains nothing. The estimator needs a tape-recorded " +
            "forward path for the flow objective to have any effect.");
    }

    /// <summary>A single frame carries no motion; silently returning 0 would hide a caller's mistake.</summary>
    [Fact]
    public void SingleFrame_Throws()
    {
        var loss = new FlowLoss<double>();
        var clip = Ramp([1, 3, 16, 16]);
        Assert.Throws<ArgumentException>(() => loss.ComputeTapeLoss(clip, clip));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void NonSequenceRank_Throws(int rank)
    {
        var loss = new FlowLoss<double>();
        var shape = Enumerable.Repeat(4, rank).ToArray();
        var t = Ramp(shape);
        Assert.Throws<ArgumentException>(() => loss.ComputeTapeLoss(t, t));
    }

    [Fact]
    public void MismatchedShapes_Throw()
    {
        var loss = new FlowLoss<double>();
        Assert.Throws<ArgumentException>(
            () => loss.ComputeTapeLoss(Ramp([2, 3, 16, 16]), Ramp([2, 3, 8, 8])));
    }

    [Fact]
    public void MismatchedRanks_Throw()
    {
        var loss = new FlowLoss<double>();
        Assert.Throws<ArgumentException>(
            () => loss.ComputeTapeLoss(Ramp([2, 3, 16, 16]), Ramp([1, 2, 3, 16, 16])));
    }

    /// <summary>
    /// Flow is defined per sample, so a multi-sample rank-5 clip must be rejected rather than have
    /// different samples' motion quietly averaged together.
    /// </summary>
    [Fact]
    public void MultiSampleBatch_IsAveragedPerSample()
    {
        // ASSERTS THE REAL CONTRACT, WHICH IS NOT "throws". This used to be MultiSampleBatch_Throws,
        // written when a rank-5 batch was rejected. ComputeTapeLoss now walks the batch --
        // "one sample at a time, then average", because optical flow is defined per sample and the
        // batch cannot be folded into another axis -- so the throw it asserted no longer happens.
        // The NotSupportedException it was catching is still there, but it guards an INTERNAL
        // invariant inside ClipLoss that the narrowing loop is precisely what stops being reached.
        //
        // Averaging is the property worth pinning, and "does not throw" would not pin it: a batch of
        // two IDENTICAL samples must score exactly what that sample scores alone. A sum instead of a
        // mean would double it; scoring only the first sample would pass this but fail the second
        // assertion below, where the two samples differ and the batch must land between them.
        var loss = new FlowLoss<double>();

        var single = Ramp([1, 2, 3, 16, 16]);
        var duplicated = new Tensor<double>([2, 2, 3, 16, 16]);
        for (int i = 0; i < single.Length; i++)
        {
            duplicated[i] = single[i];
            duplicated[single.Length + i] = single[i];
        }

        double singleLoss = loss.ComputeTapeLoss(single, single.Clone())[0];
        double duplicatedLoss = loss.ComputeTapeLoss(duplicated, duplicated.Clone())[0];
        Assert.Equal(singleLoss, duplicatedLoss, 10);

        // Two DIFFERENT samples: the mean must sit between the two individual scores, which a
        // first-sample-only implementation could not satisfy.
        var sampleA = Ramp([1, 2, 3, 16, 16]);
        var sampleB = Ramp([1, 2, 3, 16, 16], scale: 2.0, seed: 5);
        var mixed = new Tensor<double>([2, 2, 3, 16, 16]);
        for (int i = 0; i < sampleA.Length; i++)
        {
            mixed[i] = sampleA[i];
            mixed[sampleA.Length + i] = sampleB[i];
        }

        double lossA = loss.ComputeTapeLoss(sampleA, sampleA.Clone())[0];
        double lossB = loss.ComputeTapeLoss(sampleB, sampleB.Clone())[0];
        double mixedLoss = loss.ComputeTapeLoss(mixed, mixed.Clone())[0];
        Assert.Equal((lossA + lossB) / 2.0, mixedLoss, 10);
    }

    /// <summary>A rank-5 clip with a single sample is the normal batched form and must work.</summary>
    [Fact]
    public void SingleSampleRank5_IsAccepted()
    {
        var loss = new FlowLoss<double>();
        var predicted = Ramp([1, 2, 3, 16, 16], seed: 1);
        var target = Ramp([1, 2, 3, 16, 16], seed: 4);

        var value = loss.ComputeTapeLoss(predicted, target);
        _out.WriteLine($"rank-5 single-sample flow loss = {value[0]}");
        Assert.True(Fin(value[0]));
    }

    /// <summary>
    /// The flat-vector API cannot express frame structure, so it must fail loudly rather than compute
    /// something meaningless.
    /// </summary>
    [Fact]
    public void FlatVectorApi_Throws()
    {
        var loss = new FlowLoss<double>();
        var v = new Vector<double>(8);
        Assert.Throws<NotSupportedException>(() => loss.CalculateLoss(v, v));

        // CalculateDerivative is no longer asserted to throw because #1994 removed it from
        // ILossFunction<T> outright -- the tape is the only source of gradients now. A member that
        // does not exist is a stronger guarantee than one that throws, and the compiler enforces it.
    }

    #endregion
}
