using System;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Guards the VAE Encode/Decode contract that a returned tensor is the CALLER'S to keep.
/// </summary>
/// <remarks>
/// <para>
/// A compiled plan is reused across calls, and <c>CompiledModelHost.Predict</c> returns the plan's
/// resident output tensor, overwritten on the next call. That is correct for plumbing consumed
/// immediately, but Encode/Decode are public API. Before this was fixed,
/// <c>ReferenceEquals(Decode(a), Decode(b))</c> was TRUE, so two latents differing by maxAbs 1.116
/// produced byte-identical images.
/// </para>
/// <para>
/// These tests exist because that bug is self-concealing: any test that decodes twice and compares
/// silently measures zero difference, since it is holding one buffer twice. It is what made an untrained
/// VAE round trip look "information-free", and it was initially misattributed to decoder tanh
/// saturation — a real but separate defect.
/// </para>
/// </remarks>
public class VaeDecodeAliasingTests
{
    private static Tensor<double> Textured(int seed)
    {
        var t = new Tensor<double>(new[] { 1, 3, 64, 64 });
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = (rng.NextDouble() * 2.0) - 1.0;
        return t;
    }

    private static double MaxAbsDiff(Tensor<double> a, Tensor<double> b)
    {
        double d = 0.0;
        for (int i = 0; i < a.Length; i++) d = Math.Max(d, Math.Abs(a[i] - b[i]));
        return d;
    }

    [Fact]
    public void TwoDecodesReturnIndependentTensors()
    {
        var model = new UniVSTModel<double>(seed: 42);

        var la = model.EncodeToLatent(Textured(1), sampleMode: false);
        var lb = model.EncodeToLatent(Textured(2), sampleMode: false);

        var da = model.DecodeFromLatent(la);
        var db = model.DecodeFromLatent(lb);

        // The direct statement of the contract: distinct calls must not hand back one buffer.
        Assert.False(ReferenceEquals(da, db),
            "Decode returned the SAME tensor object for two different latents; the caller is holding "
            + "one buffer twice, so any comparison between the two results is meaningless.");
    }

    [Fact]
    public void DistinctLatentsDecodeToDistinctImages()
    {
        // Textured inputs on purpose. Two CONSTANT images are the degenerate case: their latents differ
        // by a near-constant offset, which the decoder's GroupNorm legitimately removes, so constants
        // cannot distinguish an aliasing bug from correct normalization.
        var model = new UniVSTModel<double>(seed: 42);

        var la = model.EncodeToLatent(Textured(1), sampleMode: false);
        var lb = model.EncodeToLatent(Textured(2), sampleMode: false);

        double latentDiff = MaxAbsDiff(la, lb);
        Assert.True(latentDiff > 1e-6,
            $"Fixture problem: the two latents are not meaningfully different (maxAbs {latentDiff:R}).");

        double decodedDiff = MaxAbsDiff(model.DecodeFromLatent(la), model.DecodeFromLatent(lb));
        Assert.True(decodedDiff > 0.0,
            $"Latents differing by maxAbs {latentDiff:R} decoded to byte-identical images "
            + $"(decodedDiff {decodedDiff:R}). The decode is discarding its input.");
    }

    [Fact]
    public void AnEarlierDecodeIsNotMutatedByALaterOne()
    {
        // The consequence that makes the aliasing dangerous rather than merely wasteful: a retained
        // result must not change under the caller's feet when an unrelated decode happens.
        var model = new UniVSTModel<double>(seed: 42);

        var la = model.EncodeToLatent(Textured(3), sampleMode: false);
        var lb = model.EncodeToLatent(Textured(4), sampleMode: false);

        var first = model.DecodeFromLatent(la);
        var snapshot = new double[first.Length];
        for (int i = 0; i < first.Length; i++) snapshot[i] = first[i];

        _ = model.DecodeFromLatent(lb);   // a later, unrelated decode

        for (int i = 0; i < first.Length; i++)
            Assert.Equal(snapshot[i], first[i], 12);
    }

    [Fact]
    public void TwoEncodesReturnIndependentTensors()
    {
        // Encode goes through the same compiled-host path, so it carries the same risk. It happened to
        // look healthy when the decode bug was found only because EncodeToLatent's ScaleLatent produces
        // a fresh tensor downstream, which masked it.
        var model = new UniVSTModel<double>(seed: 42);

        var a = model.VAE.Encode(Textured(5), sampleMode: false);
        var b = model.VAE.Encode(Textured(6), sampleMode: false);

        Assert.False(ReferenceEquals(a, b),
            "Encode returned the SAME tensor object for two different images.");
        Assert.True(MaxAbsDiff(a, b) > 0.0, "Two different images encoded to byte-identical latents.");
    }

    [Fact]
    public void RepeatedDecodeOfTheSameLatentIsStable()
    {
        // Determinism must survive the copy: same input, same output, but as separate objects.
        var model = new UniVSTModel<double>(seed: 42);
        var latent = model.EncodeToLatent(Textured(7), sampleMode: false);

        var first = model.DecodeFromLatent(latent);
        var second = model.DecodeFromLatent(latent);

        Assert.False(ReferenceEquals(first, second));
        Assert.Equal(0.0, MaxAbsDiff(first, second), 12);
    }
}
