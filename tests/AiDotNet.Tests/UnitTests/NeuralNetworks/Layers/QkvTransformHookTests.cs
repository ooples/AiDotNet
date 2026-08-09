using System;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Verifies the <see cref="IQkvTransform{T}"/> hook on <see cref="MultiHeadAttentionLayer{T}"/>.
/// </summary>
/// <remarks>
/// The hook exists so techniques that steer attention by rewriting Q/K/V (UniVST's query blending and
/// key/value AdaIN) can do so without a bespoke attention subclass. The contract that matters most is
/// that it is INERT until attached: every model in the library goes through this layer, so a hook that
/// perturbed the default path would change unrelated results.
/// </remarks>
public class QkvTransformHookTests
{
    /// <summary>Zeroes exactly one of the three projections, so routing is observable.</summary>
    private sealed class ZeroOne : IQkvTransform<double>
    {
        private readonly string _which;
        public ZeroOne(string which) => _which = which;

        private static Tensor<double> Zeroed(Tensor<double> t) => new Tensor<double>(t.Shape.ToArray());

        public Tensor<double> TransformQuery(Tensor<double> query) => _which == "q" ? Zeroed(query) : query;
        public Tensor<double> TransformKey(Tensor<double> key) => _which == "k" ? Zeroed(key) : key;
        public Tensor<double> TransformValue(Tensor<double> value) => _which == "v" ? Zeroed(value) : value;
    }

    /// <summary>Returns the input unchanged — an identity hook must be a no-op end to end.</summary>
    private sealed class Identity : IQkvTransform<double>
    {
        public Tensor<double> TransformQuery(Tensor<double> query) => query;
        public Tensor<double> TransformKey(Tensor<double> key) => key;
        public Tensor<double> TransformValue(Tensor<double> value) => value;
    }

    /// <summary>Returns a wrong-shaped tensor, which must be rejected loudly.</summary>
    private sealed class WrongShape : IQkvTransform<double>
    {
        public Tensor<double> TransformQuery(Tensor<double> query) => new Tensor<double>(new[] { 1, 1 });
        public Tensor<double> TransformKey(Tensor<double> key) => key;
        public Tensor<double> TransformValue(Tensor<double> value) => value;
    }

    private const int Batch = 1, Seq = 4, Embed = 8, Heads = 2;

    private static MultiHeadAttentionLayer<double> Layer() =>
        new MultiHeadAttentionLayer<double>(headCount: Heads, headDimension: Embed / Heads);

    private static Tensor<double> Input(int seed)
    {
        var t = new Tensor<double>(new[] { Batch, Seq, Embed });
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = (rng.NextDouble() * 2.0) - 1.0;
        return t;
    }

    [Fact]
    public void NoTransformIsAttachedByDefault()
    {
        // The hook must be opt-in. Every attention-based model in the library shares this layer.
        Assert.Null(Layer().QkvTransform);
    }

    [Fact]
    public void AnIdentityTransformLeavesTheOutputUnchanged()
    {
        // Proves the hook is wired in without altering the computation: routing Q/K/V through an
        // identity must land on exactly the same numbers as not routing them at all.
        var input = Input(3);

        var plain = Layer();
        var baseline = plain.Forward(input);

        var hooked = Layer();
        hooked.SetParameters(plain.GetParameters());   // same weights, so any delta is the hook's
        hooked.QkvTransform = new Identity();
        var withHook = hooked.Forward(input);

        Assert.Equal(baseline.Length, withHook.Length);
        for (int i = 0; i < baseline.Length; i++)
            Assert.Equal(baseline[i], withHook[i], 12);
    }

    [Fact]
    public void ZeroingTheQueryChangesTheOutput()
    {
        // Confirms the hook actually reaches the computation rather than being ignored.
        var input = Input(5);

        var plain = Layer();
        var baseline = plain.Forward(input);

        var hooked = Layer();
        hooked.SetParameters(plain.GetParameters());
        hooked.QkvTransform = new ZeroOne("q");
        var zeroedQ = hooked.Forward(input);

        bool differs = false;
        for (int i = 0; i < baseline.Length; i++)
            if (Math.Abs(baseline[i] - zeroedQ[i]) > 1e-9) { differs = true; break; }

        Assert.True(differs, "Zeroing the projected query must change the attention output.");
    }

    [Fact]
    public void EachProjectionIsRoutedToItsOwnMethod()
    {
        // A hook that fed the same tensor to all three, or crossed them, would still "work" on the
        // identity test. Zeroing key versus value must give different outputs: a zeroed key flattens
        // the attention weights, while a zeroed value flattens what is being mixed.
        var input = Input(9);

        var plain = Layer();
        var parameters = plain.GetParameters();

        var kLayer = Layer();
        kLayer.SetParameters(parameters);
        kLayer.QkvTransform = new ZeroOne("k");
        var zeroK = kLayer.Forward(input);

        var vLayer = Layer();
        vLayer.SetParameters(parameters);
        vLayer.QkvTransform = new ZeroOne("v");
        var zeroV = vLayer.Forward(input);

        bool differs = false;
        for (int i = 0; i < zeroK.Length; i++)
            if (Math.Abs(zeroK[i] - zeroV[i]) > 1e-9) { differs = true; break; }

        Assert.True(differs, "Zeroing the key and zeroing the value must not produce the same output.");
    }

    [Fact]
    public void AShapeChangingTransformIsRejectedWithAClearMessage()
    {
        // Without this guard the failure surfaces later as a confusing Q_flat reshape error that names
        // an internal tensor instead of the offending transform.
        var hooked = Layer();
        hooked.QkvTransform = new WrongShape();

        var ex = Assert.Throws<InvalidOperationException>(() => hooked.Forward(Input(1)));
        Assert.Contains("TransformQuery", ex.Message);
        Assert.Contains("preserve shape", ex.Message);
    }

    // ------------------------------------------------------------ UniVST's transform

    [Fact]
    public void UniVstTransformPassesThroughWhenNoReferencesAreSet()
    {
        // References change every denoising step. A null reference means "nothing to blend against",
        // and the projection must pass through rather than being blended against a stale tensor.
        var t = new UniVSTQkvTransform<double>();
        var q = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < q.Length; i++) q[i] = i + 1;

        Assert.Same(q, t.TransformQuery(q));
        Assert.Same(q, t.TransformKey(q));
        Assert.Same(q, t.TransformValue(q));
    }

    [Fact]
    public void UniVstQueryBlendUsesGammaRegardlessOfTimestep()
    {
        // gamma = 0.35 on the edit query at EVERY timestep, so the content query keeps 0.65. The
        // key/value path is windowed; this one deliberately is not.
        var t = new UniVSTQkvTransform<double>();
        var edit = new Tensor<double>(new[] { 1, 2 });
        var content = new Tensor<double>(new[] { 1, 2 });
        edit[0] = 1.0; edit[1] = 1.0;
        content[0] = 0.0; content[1] = 0.0;
        t.ContentQuery = content;

        foreach (double fraction in new[] { 0.0, 0.2, 0.5, 1.0 })
        {
            t.TimestepFraction = fraction;
            var blended = t.TransformQuery(edit);
            Assert.Equal(0.35, blended[0], 10);
            Assert.Equal(0.35, blended[1], 10);
        }
    }

    [Fact]
    public void UniVstKeyValueBlendOnlyAppliesInsideTheRampWindow()
    {
        // The ramp covers [0.4T, 1.0T]. Below it the paper applies no key/value alignment, so the
        // projection must pass through untouched rather than being blended at the clamped end-point
        // beta — which is what a naive implementation that only clamps beta would do.
        var t = new UniVSTQkvTransform<double>();
        var key = new Tensor<double>(new[] { 1, 4 });
        var style = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) { key[i] = i * 3.0; style[i] = 100.0 + i; }
        t.StyleKey = style;

        t.TimestepFraction = 0.2;
        Assert.Same(key, t.TransformKey(key));

        t.TimestepFraction = 0.5;
        var blended = t.TransformKey(key);
        Assert.NotSame(key, blended);

        bool moved = false;
        for (int i = 0; i < 4; i++)
            if (Math.Abs(blended[i] - key[i]) > 1e-9) { moved = true; break; }
        Assert.True(moved, "Inside the ramp the key must actually be blended toward the style.");
    }
}
