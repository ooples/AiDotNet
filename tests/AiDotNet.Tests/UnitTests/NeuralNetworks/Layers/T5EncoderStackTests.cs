using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Invariants of T5's shared relative position bias (Raffel et al. 2020, Sec. 2.1) that the reference
/// implementations rely on by convention and this stack enforces: one table, counted once, trained by every
/// block, and still one table after a clone.
/// </summary>
public class T5EncoderStackTests
{
    private const int Hidden = 8, Layers = 3, Heads = 2, Buckets = 4, MaxDistance = 8, Seq = 5;

    private static T5EncoderStack<double> Stack(bool share = true) =>
        new(Hidden, Layers, Heads, Buckets, MaxDistance, shareRelativeBias: share, seed: 11);

    private static Tensor<double> Input()
    {
        var x = new Tensor<double>(new[] { 1, Seq, Hidden });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.37 * i) * 0.8;
        return x;
    }

    private static double Loss(Tensor<double> y)
    {
        double s = 0;
        for (int i = 0; i < y.Length; i++) s += y[i] * y[i];
        return s;
    }

    [Fact]
    public void The_shared_table_is_one_parameter_counted_once()
    {
        var stack = Stack();
        stack.Forward(Input());

        long blocks = stack.Blocks.Sum(b => b.ParameterCount);
        Assert.Equal(Buckets * Heads, stack.GetRelativeBiasTable().Length);
        Assert.Equal(blocks + (Buckets * Heads), stack.ParameterCount);
        Assert.All(stack.Blocks, b =>
        {
            var attention = Assert.IsType<T5RelativeBiasAttentionLayer<double>>(b.AttentionLayer);
            Assert.True(attention.UsesExternalPositionBias);
            Assert.Equal(0, attention.GetRelativeBiasTable().Length); // no block holds a table, or a reference to one
        });
    }

    [Fact]
    public void Every_block_trains_the_shared_table_and_the_gradient_matches_finite_differences()
    {
        var stack = Stack();
        var input = Input();
        stack.Forward(input);
        var table = stack.GetRelativeBiasTable();

        var engine = AiDotNetEngine.Current;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var y = stack.Forward(input);
            var axes = Enumerable.Range(0, y.Shape.Length).ToArray();
            var loss = engine.ReduceSum(engine.TensorMultiply(y, y), axes, keepDims: false);
            grads = tape.ComputeGradients(loss, new[] { table });
        }

        // Presence first: the tape omits a tensor it found no path to.
        Assert.True(grads.TryGetValue(table, out var analytic), "no gradient reached the shared table");

        // The central difference is the TOTAL derivative through every block. If any block's use of the
        // table were severed, the analytic gradient would miss that block's share and disagree.
        const double h = 1e-6;
        for (int i = 0; i < table.Length; i++)
        {
            double saved = table[i];
            table[i] = saved + h; double up = Loss(stack.Forward(input));
            table[i] = saved - h; double down = Loss(stack.Forward(input));
            table[i] = saved;
            double numeric = (up - down) / (2 * h);
            Assert.True(Math.Abs(numeric - analytic[i]) <= 1e-5 * Math.Max(1.0, Math.Abs(numeric)),
                $"table[{i}]: analytic {analytic[i]:R} vs finite difference {numeric:R}");
        }
    }

    [Fact]
    public void A_clone_predicts_identically_and_keeps_one_independent_table()
    {
        var stack = Stack();
        var input = Input();
        var expected = stack.Forward(input);

        var clone = (T5EncoderStack<double>)stack.Clone();
        var actual = clone.Forward(input);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal(stack.ParameterCount, clone.ParameterCount);
        Assert.NotSame(stack.GetRelativeBiasTable(), clone.GetRelativeBiasTable());
        Assert.All(clone.Blocks, b =>
            Assert.True(((T5RelativeBiasAttentionLayer<double>)b.AttentionLayer).UsesExternalPositionBias));

        // Independence: changing the clone's table moves only the clone.
        clone.GetRelativeBiasTable()[0] += 1.0;
        Assert.Equal(expected.ToArray(), stack.Forward(input).ToArray());
        Assert.NotEqual(expected.ToArray(), clone.Forward(input).ToArray());
    }

    [Fact]
    public void The_per_layer_variant_gives_every_block_its_own_table()
    {
        var stack = Stack(share: false);
        stack.Forward(Input());

        Assert.Equal(0, stack.GetRelativeBiasTable().Length);
        var tables = stack.Blocks
            .Select(b => ((T5RelativeBiasAttentionLayer<double>)b.AttentionLayer).GetRelativeBiasTable())
            .ToList();
        Assert.All(tables, t => Assert.Equal(Buckets * Heads, t.Length));
        Assert.Equal(tables.Count, tables.Distinct().Count());
    }
}
