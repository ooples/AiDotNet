using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// RBFLayer places its centres on the first batch and sets widths to d_max / sqrt(2M) over the centres (Broomhead and Lowe 1988; Haykin 5.10).
/// The original Uniform(0, 1) draw could produce a width near 0, whose unit outputs exactly 0 for every input: the
/// generated Forward_DifferentInputs invariant failed whenever an unseeded run drew such widths.
/// </summary>
public class RBFLayerInitializationTests
{
    private static double[] Widths(RBFLayer<double> layer)
    {
        // GetParameters is centres (outputSize x inputSize) then widths (outputSize).
        var p = layer.GetParameters();
        int count = layer.GetOutputShape()[0];
        var widths = new double[count];
        for (int i = 0; i < count; i++) widths[i] = p[p.Length - count + i];
        return widths;
    }

    private static double[] Centers(RBFLayer<double> layer, int inputSize)
    {
        var p = layer.GetParameters();
        int count = layer.GetOutputShape()[0];
        var centers = new double[count * inputSize];
        for (int i = 0; i < centers.Length; i++) centers[i] = p[i];
        return centers;
    }

    [Theory]
    [InlineData(4, 3)]
    [InlineData(16, 10)]
    [InlineData(2, 1)]
    public void Widths_are_the_center_spread_formula(int inputSize, int centers)
    {
        var layer = new RBFLayer<double>(inputSize, centers);
        var c = Centers(layer, inputSize);
        double maxSquared = 0;
        for (int a = 0; a < centers; a++)
        for (int b = a + 1; b < centers; b++)
        {
            double s = 0;
            for (int d = 0; d < inputSize; d++) s += (c[a * inputSize + d] - c[b * inputSize + d]) * (c[a * inputSize + d] - c[b * inputSize + d]);
            maxSquared = Math.Max(maxSquared, s);
        }
        double expected = maxSquared > 0 ? Math.Sqrt(maxSquared) / Math.Sqrt(2.0 * centers) : 1.0;
        foreach (var w in Widths(layer)) Assert.Equal(expected, w, 12);
    }

    [Fact]
    public void No_unit_starts_saturated_over_many_initializations()
    {
        // Every unit must answer some input in the span of the centres with a value float can represent, so two
        // different inputs give different outputs. The old Uniform(0, 1) widths fail this within a few dozen draws.
        for (int trial = 0; trial < 200; trial++)
        {
            var layer = new RBFLayer<double>(8, 6);
            foreach (var w in Widths(layer))
            {
                double epsilon = 1.0 / (2.0 * w * w);
                Assert.True(epsilon < 88.0, $"trial {trial}: width {w} gives epsilon {epsilon}, which underflows exp in float");
            }
        }
    }

    [Fact]
    public void The_uniform_option_keeps_the_original_draw()
    {
        var layer = new RBFLayer<double>(8, 64, widthInitialization: RbfWidthInitialization.Uniform);
        var widths = Widths(layer);
        Assert.All(widths, w => Assert.InRange(w, 0.0, 1.0));
        Assert.True(widths.Distinct().Count() > 1, "uniform widths should differ from unit to unit");
    }

    [Fact]
    public void Single_center_falls_back_to_the_unit_gaussian()
    {
        var layer = new RBFLayer<double>(5, 1);
        Assert.Equal(1.0, Widths(layer)[0], 12);
    }

    private static Tensor<double> Batch(int rows, int dims, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { rows, dims });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
        return t;
    }

    [Fact]
    public void First_batch_places_the_centres_on_evenly_spaced_rows()
    {
        const int dims = 4, centers = 3, rows = 9;
        var layer = new RBFLayer<double>(dims, centers);
        var batch = Batch(rows, dims, 5);
        layer.Forward(batch);
        var c = Centers(layer, dims);
        for (int i = 0; i < centers; i++)
        {
            int row = i * rows / centers;
            for (int d = 0; d < dims; d++) Assert.Equal(batch[row * dims + d], c[i * dims + d], 15);
        }
    }

    [Fact]
    public void Placement_happens_once_and_a_clone_never_repeats_it()
    {
        const int dims = 4, centers = 3;
        var layer = new RBFLayer<double>(dims, centers);
        layer.Forward(Batch(6, dims, 1));
        var placed = layer.GetParameters().ToArray();

        layer.Forward(Batch(6, dims, 2));
        Assert.Equal(placed, layer.GetParameters().ToArray());

        var clone = (RBFLayer<double>)layer.Clone();
        clone.Forward(Batch(6, dims, 3));
        Assert.Equal(placed, clone.GetParameters().ToArray());
    }

    [Fact]
    public void A_batch_smaller_than_the_layer_places_one_centre_per_row_and_keeps_the_rest()
    {
        const int dims = 4, centers = 5, rows = 2;
        var layer = new RBFLayer<double>(dims, centers);
        var before = Centers(layer, dims);
        var batch = Batch(rows, dims, 9);
        layer.Forward(batch);
        var after = Centers(layer, dims);
        for (int i = 0; i < centers; i++)
        for (int d = 0; d < dims; d++)
            Assert.Equal(i < rows ? batch[i * dims + d] : before[i * dims + d], after[i * dims + d], 15);
    }

    [Fact]
    public void As_initialized_keeps_the_centres_the_layer_was_given()
    {
        const int dims = 4, centers = 3;
        var layer = new RBFLayer<double>(dims, centers, centerInitialization: RbfCenterInitialization.AsInitialized);
        var before = layer.GetParameters().ToArray();
        layer.Forward(Batch(6, dims, 4));
        Assert.Equal(before, layer.GetParameters().ToArray());
    }
}