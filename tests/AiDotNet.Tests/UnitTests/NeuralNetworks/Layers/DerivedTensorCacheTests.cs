using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// The cache behind GRULayer's and LSTMLayer's stacked inference weights and the depthwise conv-BN folded mean: it must
/// hand back the same tensor while nothing changed, and rebuild after any change that could alter the result.
/// </summary>
public class DerivedTensorCacheTests
{
    private static Tensor<double> Vector(params double[] values) => new(values, new[] { values.Length });

    private static int _builds;

    private static Tensor<double> Concat(IEngine engine, Tensor<double>[] sources)
    {
        _builds++;
        return engine.TensorConcatenate(sources, axis: 0);
    }

    [Fact]
    public void Reuses_the_value_while_the_sources_are_unchanged()
    {
        var cache = new DerivedTensorCache<double>();
        var engine = AiDotNetEngine.Current;
        var a = Vector(1, 2);
        var b = Vector(3);

        var first = cache.GetOrBuild(engine, Concat, a, b);
        int builds = _builds;
        var second = cache.GetOrBuild(engine, Concat, a, b);

        Assert.Same(first, second);
        Assert.Equal(builds, _builds);
        Assert.Equal(new[] { 1.0, 2.0, 3.0 }, second.ToArray());
    }

    [Fact]
    public void Rebuilds_after_an_in_place_write_to_a_source()
    {
        var cache = new DerivedTensorCache<double>();
        var engine = AiDotNetEngine.Current;
        var a = Vector(1, 2);
        var b = Vector(3);
        cache.GetOrBuild(engine, Concat, a, b);

        b.SetFlat(0, 7);

        Assert.Equal(new[] { 1.0, 2.0, 7.0 }, cache.GetOrBuild(engine, Concat, a, b).ToArray());
    }

    [Fact]
    public void Rebuilds_when_a_source_is_replaced()
    {
        var cache = new DerivedTensorCache<double>();
        var engine = AiDotNetEngine.Current;
        var a = Vector(1, 2);
        cache.GetOrBuild(engine, Concat, a, Vector(3));

        // A fresh tensor starts at the same version as the one it replaces, so only its identity tells them apart.
        Assert.Equal(new[] { 1.0, 2.0, 9.0 }, cache.GetOrBuild(engine, Concat, a, Vector(9)).ToArray());
    }

    [Fact]
    public void Rebuilds_when_the_source_count_changes()
    {
        var cache = new DerivedTensorCache<double>();
        var engine = AiDotNetEngine.Current;
        var a = Vector(1, 2);
        var b = Vector(3);
        cache.GetOrBuild(engine, Concat, a, b);

        Assert.Equal(new[] { 1.0, 2.0 }, cache.GetOrBuild(engine, Concat, a).ToArray());
    }

    [Fact]
    public void Rebuilds_on_another_engine_and_after_invalidation()
    {
        var cache = new DerivedTensorCache<double>();
        var a = Vector(1, 2);
        var b = Vector(3);
        var first = cache.GetOrBuild(AiDotNetEngine.Current, Concat, a, b);

        var cpu = new CpuEngine();
        var other = cache.GetOrBuild(cpu, Concat, a, b);
        Assert.NotSame(first, other);
        Assert.Same(other, cache.GetOrBuild(cpu, Concat, a, b));

        cache.Invalidate();
        Assert.NotSame(other, cache.GetOrBuild(cpu, Concat, a, b));
    }
}
