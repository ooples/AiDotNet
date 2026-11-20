using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public sealed class OwnedDerivedTensorCacheTests
{
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GpuReplacementReturnsOwnedBackingAndPreservesRetainedViews(bool retainView)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
            "Actual CUDA required.");
        using var direction = new Tensor<float>(new[] { 1f, 2f, 3f, 4f, 2f, 3f, 4f, 5f }, new[] { 2, 4 });
        using var gain = new Tensor<float>(new[] { .5f, .75f }, new[] { 2 });
        var cache = new DerivedTensorCache<float>(disposeReplacedValues: true);
        var first = cache.GetOrBuild(engine, static (e, s) => e.TensorWeightNorm(s[0], s[1], 1e-24), direction, gain);
        using var expectedTensor = new CpuEngine().TensorWeightNorm(direction, gain, 1e-24);
        var expected = expectedTensor.ToArray();
        var backing = first.TryGetGpuBuffer()!;
        Assert.NotNull(backing);
        using var view = retainView ? new CpuEngine().Reshape(first, new[] { 8 }) : null;
        direction[0] = 1.5f;
        var second = cache.GetOrBuild(engine, static (e, s) => e.TensorWeightNorm(s[0], s[1], 1e-24), direction, gain);
        Assert.True(first.IsDisposed);
        if (view is not null)
        {
            Assert.NotEqual(IntPtr.Zero, backing.Handle); // The outstanding view still owns its original value.
            var actual = view.ToArray();
            for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(expected[i] - actual[i]), 0f, 1e-6f);
            view.Dispose();
        }
        else Assert.Equal(IntPtr.Zero, backing.Handle);
        Assert.NotEqual(expected[0], second.ToArray()[0]);
        cache.Invalidate();
        Assert.True(second.IsDisposed);
        Assert.False(direction.IsDisposed);
    }

    [Fact]
    public void ReplacementAndInvalidationRetireOwnedValuesWhileViewsKeepTheirData()
    {
        var engine = new CpuEngine();
        using var source = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var cache = new DerivedTensorCache<float>(disposeReplacedValues: true);
        var first = cache.GetOrBuild(engine, static (e, s) => e.TensorMultiplyScalar(s[0], 2f), source);
        using var view = engine.Reshape(first, new[] { 1, 2 });
        source[0] = 3f;
        var second = cache.GetOrBuild(engine, static (e, s) => e.TensorMultiplyScalar(s[0], 2f), source);
        Assert.True(first.IsDisposed);
        Assert.Equal(new[] { 2f, 4f }, view.ToArray());
        Assert.Equal(new[] { 6f, 4f }, second.ToArray());
        cache.Invalidate();
        Assert.True(second.IsDisposed);
        Assert.False(source.IsDisposed);
    }

    [Fact]
    public void CallerOwnedSourcesArePreservedWhenBuilderReturnsOne()
    {
        var engine = new CpuEngine();
        using var source = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var cache = new DerivedTensorCache<float>(disposeReplacedValues: true);
        Assert.Same(source, cache.GetOrBuild(engine, static (_, s) => s[0], source));
        source[0] = 4f;
        cache.GetOrBuild(engine, static (e, s) => e.TensorMultiplyScalar(s[0], 2f), source);
        Assert.False(source.IsDisposed);
        cache.Invalidate();
        Assert.Equal(new[] { 4f, 2f }, source.ToArray());
    }

    [Fact]
    public void FailedRebuildPreservesTheExistingOwnedValue()
    {
        var engine = new CpuEngine();
        using var source = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var cache = new DerivedTensorCache<float>(disposeReplacedValues: true);
        var first = cache.GetOrBuild(engine, static (e, s) => e.TensorMultiplyScalar(s[0], 2f), source);
        source[0] = 3f;
        Assert.Throws<InvalidOperationException>(() => cache.GetOrBuild(engine,
            static (_, _) => throw new InvalidOperationException("Injected builder failure"), source));
        Assert.False(first.IsDisposed);
        Assert.Equal(new[] { 2f, 4f }, first.ToArray());
        cache.Invalidate();
        Assert.True(first.IsDisposed);
    }
}
