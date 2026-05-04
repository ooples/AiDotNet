using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression coverage for issue #1213's lazy-ctor migration of composite
/// layers — first reference impl is <see cref="RBMLayer{T}"/>. The new
/// constructor takes only the architectural hidden-unit count; visible-unit
/// count is resolved from <c>input.Shape[^1]</c> on first
/// <see cref="ILayer{T}.Forward(Tensor{T})"/>.
/// </summary>
public class RBMLayerLazyCtorIssue1213Tests
{
    /// <summary>
    /// Lazy ctor: pre-Forward instance reports hidden-bias-only
    /// <see cref="LayerBase{T}.ParameterCount"/>; first Forward resolves
    /// visible dim from input shape and the count jumps to the full
    /// (visible×hidden + visible + hidden) total. This is the canonical
    /// lazy-shape contract from #1209 applied to a composite layer.
    /// </summary>
    [Fact]
    public void LazyCtor_ResolvesShape_OnFirstForward()
    {
        // Architectural-only ctor — no visibleUnits arg.
        using var layer = new RBMLayer<float>(hiddenUnits: 8);

        // Pre-Forward: lazy state. Visible dim is unknown → InputShape
        // contains the -1 sentinel, all parameter tensors are zero-sized
        // placeholders, and ParameterCount returns 0 — matching the
        // actual GetParameters().Length so the optimizer's per-parameter
        // bookkeeping isn't sized for state that doesn't yet exist.
        Assert.False(layer.IsShapeResolved);
        Assert.False(layer.IsInitialized);
        Assert.Equal(-1, layer.GetInputShape()[0]);
        Assert.Equal(8, layer.GetOutputShape()[0]);
        Assert.Equal(0, layer.ParameterCount);

        // First Forward: rank-2 input [batch=2, visible=4]. Layer must
        // resolve visibleUnits = 4 from input.Shape[^1] and materialize
        // the [hidden, visible] weight matrix + visible-bias vector.
        var input = new Tensor<float>(new[] { 2, 4 });
        for (int i = 0; i < input.Length; i++) input[i] = (float)((i + 1) * 0.1);

        var output = layer.Forward(input);

        // Post-Forward: shape resolved.
        Assert.True(layer.IsShapeResolved);
        Assert.True(layer.IsInitialized);
        Assert.Equal(4, layer.GetInputShape()[0]);
        Assert.Equal(8, layer.GetOutputShape()[0]);

        // ParameterCount jumps to full RBM total: weights (8×4 = 32)
        // + visible biases (4) + hidden biases (8) = 44.
        Assert.Equal(44, layer.ParameterCount);

        // Output shape: [batch=2, hidden=8] — same rank as input,
        // last axis swapped from visible to hidden.
        Assert.Equal(2, output.Rank);
        Assert.Equal(2, output.Shape[0]);
        Assert.Equal(8, output.Shape[1]);
    }

    /// <summary>
    /// Lazy and eager ctors must produce identical outputs once both have
    /// the same visible dim resolved. This proves the lazy path doesn't
    /// regress numerics — it only defers shape resolution, not changes the
    /// math. Same seed via the layer's internal initializer so weight init
    /// is identical between the two instances.
    /// </summary>
    [Fact]
    public void LazyCtor_MatchesEagerCtor_AfterSameForward()
    {
        const int visible = 4;
        const int hidden = 8;

        var input = new Tensor<float>(new[] { 1, visible });
        for (int i = 0; i < input.Length; i++) input[i] = (float)((i + 1) * 0.1);

        // Eager: visible dim known at ctor time. Pass null activation
        // explicitly typed to disambiguate from the IVectorActivationFunction
        // overload (both ctors take three int + nullable-activation params).
        using var eager = new RBMLayer<float>(visible, hidden, (AiDotNet.Interfaces.IActivationFunction<float>?)null);
        // Lazy: visible dim resolved on first Forward.
        using var lazy = new RBMLayer<float>(hidden);

        // Match parameters between the two instances so the math is
        // bit-identical regardless of init RNG. RBMLayer's eager ctor
        // runs Xavier init at construction; lazy runs Xavier in
        // EnsureInitialized triggered by first Forward. Force the lazy
        // path to materialize first, then copy eager's params over.
        _ = lazy.Forward(input);
        lazy.SetParameters(eager.GetParameters());

        var eagerOut = eager.Forward(input);
        var lazyOut = lazy.Forward(input);

        Assert.Equal(eagerOut.Length, lazyOut.Length);
        for (int i = 0; i < eagerOut.Length; i++)
        {
            // Sigmoid output: bounded [0, 1]. Tolerance 1e-5 is
            // comfortably below any float-rounding drift between the two
            // forward paths (which run identical engine ops on identical
            // tensors).
            Assert.True(System.Math.Abs(eagerOut[i] - lazyOut[i]) < 1e-5f,
                $"Lazy ctor output[{i}] = {lazyOut[i]:G7} differs from eager " +
                $"output[{i}] = {eagerOut[i]:G7} after parameter sync. " +
                $"The lazy path should produce numerically identical " +
                $"results once shape is resolved and parameters match.");
        }
    }

    /// <summary>
    /// Lazy ctor rejects degenerate input shape. A rank-1 tensor with shape
    /// [0] (zero visible units) can't be a valid RBM input — surface this
    /// as ArgumentException at first Forward instead of silently
    /// allocating a 0-sized weight matrix. Note: this exercises only the
    /// non-positive-visible-units branch of OnFirstForward; the
    /// rank&lt;1 branch is exercised by
    /// <see cref="LazyCtor_RejectsScalarInput"/>.
    /// </summary>
    [Fact]
    public void LazyCtor_RejectsZeroVisibleUnitsInput()
    {
        using var layer = new RBMLayer<float>(hiddenUnits: 8);

        var rank1ZeroLength = new Tensor<float>(new int[] { 0 });
        Assert.Throws<System.ArgumentException>(() => layer.Forward(rank1ZeroLength));
    }

    /// <summary>
    /// Lazy ctor rejects rank-0 (true scalar) input. OnFirstForward's
    /// rank&lt;1 guard fires before the visible-unit check, since a
    /// scalar has no last dimension to read a visible-unit count from.
    /// Pairs with <see cref="LazyCtor_RejectsZeroVisibleUnitsInput"/> to
    /// cover both degenerate-input branches.
    /// </summary>
    [Fact]
    public void LazyCtor_RejectsScalarInput()
    {
        using var layer = new RBMLayer<float>(hiddenUnits: 8);

        var scalar = new Tensor<float>(System.Array.Empty<int>());
        Assert.Throws<System.ArgumentException>(() => layer.Forward(scalar));
    }
}
