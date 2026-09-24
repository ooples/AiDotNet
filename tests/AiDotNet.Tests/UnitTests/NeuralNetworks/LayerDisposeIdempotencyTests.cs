using System;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A second <c>Dispose</c> of a layer must not re-run the layer's own teardown.
///
/// <para><see cref="LayerBase{T}"/> guarded its own <c>Dispose(bool)</c>, but its public <c>Dispose()</c>
/// re-entered every override on each call, and overrides do their teardown before calling base.
/// <see cref="DenseLayer{T}"/>'s override, for one, re-invalidates the engine's cached GPU copy of
/// <c>_weights</c>/<c>_biases</c> -- tensors whose pooled storage the first dispose already returned and a
/// newer layer may now own.</para>
/// </summary>
public partial class LayerDisposeIdempotencyTests
{
    private static CountingDenseLayer MaterializedLayer()
    {
        var layer = new CountingDenseLayer();
        // Resolve the lazy input shape so the weights are really rented from the pool.
        _ = layer.Forward(new Tensor<double>(new[] { 1, 4 }));
        return layer;
    }

    [Fact]
    public void Repeated_direct_disposes_run_the_teardown_and_return_the_buffers_once()
    {
        var layer = MaterializedLayer();

        layer.Dispose();
        layer.Dispose();
        layer.Dispose();

        Assert.Equal(1, layer.DisposeCalls);
        Assert.Equal(1, layer.PooledReturns);
    }

    [Fact]
    public void Guard_then_direct_dispose_tears_down_once()
    {
        var layer = MaterializedLayer();

        Assert.True(DisposeOnceGuard.TryDispose(layer));
        Assert.False(DisposeOnceGuard.TryDispose(layer));
        layer.Dispose();

        Assert.Equal(1, layer.DisposeCalls);
        Assert.Equal(1, layer.PooledReturns);
    }

    [Fact]
    public void Direct_then_guard_dispose_tears_down_once()
    {
        var layer = MaterializedLayer();

        layer.Dispose();
        // The guard has not seen this layer, so it forwards the call -- which must now be a no-op.
        DisposeOnceGuard.TryDispose(layer);

        Assert.Equal(1, layer.DisposeCalls);
        Assert.Equal(1, layer.PooledReturns);
    }

    private sealed partial class CountingDenseLayer : DenseLayer<double>
    {
        public CountingDenseLayer() : base(3, (AiDotNet.Interfaces.IActivationFunction<double>)new IdentityActivation<double>())
        {
        }

        public int DisposeCalls { get; private set; }

        public int PooledReturns { get; private set; }

        protected override void Dispose(bool disposing)
        {
            DisposeCalls++;
            base.Dispose(disposing);
        }

        protected override void ReturnPooledParameters()
        {
            PooledReturns++;
            base.ReturnPooledParameters();
        }
    }
}
