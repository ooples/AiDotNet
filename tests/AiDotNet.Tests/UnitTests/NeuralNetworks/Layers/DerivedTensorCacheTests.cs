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

    // GRULayer passes three sources and LSTMLayer four, so the third and fourth slots are checked like the first two.
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void Rebuilds_after_a_write_to_the_third_or_fourth_source(int written)
    {
        var cache = new DerivedTensorCache<double>();
        var engine = AiDotNetEngine.Current;
        var sources = new[] { Vector(1), Vector(2), Vector(3), Vector(4) };
        var first = cache.GetOrBuild(engine, Concat, sources[0], sources[1], sources[2], sources[3]);
        Assert.Same(first, cache.GetOrBuild(engine, Concat, sources[0], sources[1], sources[2], sources[3]));

        sources[written].SetFlat(0, 9);

        var expected = new[] { 1.0, 2.0, 3.0, 4.0 };
        expected[written] = 9.0;
        Assert.Equal(expected, cache.GetOrBuild(engine, Concat, sources[0], sources[1], sources[2], sources[3]).ToArray());
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void Rebuilds_when_the_third_or_fourth_source_is_replaced(int replaced)
    {
        var cache = new DerivedTensorCache<double>();
        var engine = AiDotNetEngine.Current;
        var sources = new[] { Vector(1), Vector(2), Vector(3), Vector(4) };
        cache.GetOrBuild(engine, Concat, sources[0], sources[1], sources[2], sources[3]);

        sources[replaced] = Vector(9);

        var expected = new[] { 1.0, 2.0, 3.0, 4.0 };
        expected[replaced] = 9.0;
        Assert.Equal(expected, cache.GetOrBuild(engine, Concat, sources[0], sources[1], sources[2], sources[3]).ToArray());
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
