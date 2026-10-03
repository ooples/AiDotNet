using System;
using System.Linq;
using AiDotNet.ComputerVision;
using AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Checks <see cref="MultiScaleDeformableAttention{T}"/> against an independent scalar implementation of
/// the reference MSDeformAttn forward (Deformable DETR, ms_deform_attn_core_pytorch). The oracle samples with
/// explicit bilinear interpolation, zero padding and align_corners = false, and shares no code with the module.
/// </summary>
public class MultiScaleDeformableAttentionTests
{
    private const int D = 16, Levels = 3, Heads = 4, Points = 2, Batch = 2, Queries = 5;
    private static readonly int[][] Shapes = { new[] { 6, 5 }, new[] { 3, 3 }, new[] { 2, 1 } };
    private static readonly int[] Starts = { 0, 30, 39 };
    private const int Tokens = 41;

    [Fact]
    public void Initialization_MatchesReferenceResetParameters()
    {
        var attention = new MultiScaleDeformableAttention<double>(D, Levels, Heads, Points);
        var p = attention.GetParameters();
        int offsets = Heads * Levels * Points * 2, weights = Heads * Levels * Points;
        int biasStart = D * offsets;
        Assert.All(Enumerable.Range(0, D * offsets), i => Assert.Equal(0.0, p[i]));
        for (int m = 0; m < Heads; m++)
        {
            double theta = m * 2 * Math.PI / Heads;
            double cx = Math.Cos(theta), cy = Math.Sin(theta), s = Math.Max(Math.Abs(cx), Math.Abs(cy));
            for (int l = 0; l < Levels; l++)
                for (int k = 0; k < Points; k++)
                {
                    int index = biasStart + ((((m * Levels) + l) * Points) + k) * 2;
                    Assert.Equal(cx / s * (k + 1), p[index], 12);
                    Assert.Equal(cy / s * (k + 1), p[index + 1], 12);
                }
        }

        int attentionStart = biasStart + offsets;
        Assert.All(Enumerable.Range(attentionStart, (D * weights) + weights), i => Assert.Equal(0.0, p[i]));
        int valueStart = attentionStart + (D * weights) + weights;
        double bound = Math.Sqrt(6.0 / (D + D));
        var valueWeights = Enumerable.Range(valueStart, D * D).Select(i => p[i]).ToArray();
        Assert.All(valueWeights, v => Assert.InRange(v, -bound, bound));
        Assert.Contains(valueWeights, v => v != 0);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void Forward_MatchesScalarReference(int referenceDim)
    {
        var rng = new Random(11 + referenceDim);
        var attention = new MultiScaleDeformableAttention<double>(D, Levels, Heads, Points);
        var parameters = RandomizeParameters(attention, rng);
        var (query, reference, value) = RandomInputs(rng, referenceDim);

        var actual = attention.Forward(query, reference, value, Shapes, Starts);
        var expected = Reference(parameters, query, reference, value, referenceDim);

        Assert.Equal(new[] { Batch, Queries, D }, actual._shape);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-10, $"element {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void TapeGradient_MatchesFiniteDifferences(int referenceDim)
    {
        var rng = new Random(29 + referenceDim);
        var attention = new MultiScaleDeformableAttention<double>(D, Levels, Heads, Points);
        RandomizeParameters(attention, rng);
        var (query, reference, value) = RandomInputs(rng, referenceDim);
        var target = new Tensor<double>(new[] { Batch, Queries, D });
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble() - 0.5;

        double Loss() => TensorModelTrainer<double>.MeanSquaredError(attention.Forward(query, reference, value, Shapes, Starts), target)[0];

        var sources = attention.GetParameterStateChunks().Select(c => c.Tensor).Concat(new[] { reference, query, value }).ToArray();
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var loss = TensorModelTrainer<double>.MeanSquaredError(attention.Forward(query, reference, value, Shapes, Starts), target);
            gradients = tape.ComputeGradients(loss, sources);
        }

        int checkedCount = 0;
        foreach (var source in sources)
        {
            Assert.True(gradients.ContainsKey(source), "a source received no gradient");
            var gradient = gradients[source];
            foreach (int i in Enumerable.Range(0, source.Length).OrderBy(_ => rng.Next()).Take(6))
            {
                double original = source[i];
                const double h = 1e-6;
                source[i] = original + h; double plus = Loss();
                source[i] = original - h; double minus = Loss();
                source[i] = original;
                double fd = (plus - minus) / (2 * h);
                Assert.True(Math.Abs(fd - gradient[i]) <= 1e-6 + (1e-4 * Math.Abs(fd)),
                    $"source of shape [{string.Join(",", source._shape)}] element {i}: tape {gradient[i]:R}, finite difference {fd:R}");
                checkedCount++;
            }
        }

        Assert.Equal(sources.Length * 6, checkedCount);
    }

    [Fact]
    public void Forward_RejectsMisplacedLevelStart()
    {
        var attention = new MultiScaleDeformableAttention<double>(D, Levels, Heads, Points);
        var (query, reference, value) = RandomInputs(new Random(3), 2);
        Assert.Throws<ArgumentException>(() => attention.Forward(query, reference, value, Shapes, new[] { 0, 31, 39 }));
    }

    private static double[] RandomizeParameters(MultiScaleDeformableAttention<double> attention, Random rng)
    {
        var values = new double[attention.ParameterCount];
        for (int i = 0; i < values.Length; i++) values[i] = (rng.NextDouble() - 0.5) * 0.6;
        attention.SetParameters(new Vector<double>(values));
        return values;
    }

    private static (Tensor<double> Query, Tensor<double> Reference, Tensor<double> Value) RandomInputs(Random rng, int referenceDim)
    {
        var query = new Tensor<double>(new[] { Batch, Queries, D });
        for (int i = 0; i < query.Length; i++) query[i] = rng.NextDouble() * 2 - 1;
        var value = new Tensor<double>(new[] { Batch, Tokens, D });
        for (int i = 0; i < value.Length; i++) value[i] = rng.NextDouble() * 2 - 1;
        var reference = new Tensor<double>(new[] { Batch, Queries, Levels, referenceDim });
        for (int i = 0; i < reference.Length; i++)
            reference[i] = referenceDim == 4 && (i % 4) >= 2 ? 0.1 + (rng.NextDouble() * 0.5) : 0.05 + (rng.NextDouble() * 0.9);
        return (query, reference, value);
    }

    /// <summary>Scalar MSDeformAttn forward: projections, softmax, sampling locations and bilinear GridSample.</summary>
    private static double[] Reference(double[] p, Tensor<double> query, Tensor<double> reference, Tensor<double> value, int referenceDim)
    {
        int offsets = Heads * Levels * Points * 2, weights = Heads * Levels * Points;
        int o = 0;
        double[] Take(int n) { var a = p.Skip(o).Take(n).ToArray(); o += n; return a; }
        var offW = Take(D * offsets); var offB = Take(offsets);
        var attW = Take(D * weights); var attB = Take(weights);
        var valW = Take(D * D); var valB = Take(D);
        var outW = Take(D * D); var outB = Take(D);

        double[] Linear(Tensor<double> x, int row, double[] w, double[] b, int outDim)
        {
            var y = new double[outDim];
            for (int j = 0; j < outDim; j++)
            {
                double s = b[j];
                for (int k = 0; k < D; k++) s += x[(row * D) + k] * w[(k * outDim) + j];
                y[j] = s;
            }
            return y;
        }

        int dh = D / Heads;
        var result = new double[Batch * Queries * D];
        for (int n = 0; n < Batch; n++)
        {
            var projected = new double[Tokens][];
            for (int t = 0; t < Tokens; t++) projected[t] = Linear(value, (n * Tokens) + t, valW, valB, D);

            for (int q = 0; q < Queries; q++)
            {
                int row = (n * Queries) + q;
                var off = Linear(query, row, offW, offB, offsets);
                var logit = Linear(query, row, attW, attB, weights);
                var head = new double[D];
                for (int m = 0; m < Heads; m++)
                {
                    int baseIndex = m * Levels * Points;
                    double max = Enumerable.Range(baseIndex, Levels * Points).Max(i => logit[i]);
                    double denom = Enumerable.Range(baseIndex, Levels * Points).Sum(i => Math.Exp(logit[i] - max));
                    for (int l = 0; l < Levels; l++)
                    {
                        int h = Shapes[l][0], w = Shapes[l][1];
                        int refBase = ((row * Levels) + l) * referenceDim;
                        for (int k = 0; k < Points; k++)
                        {
                            int pi = ((((m * Levels) + l) * Points) + k);
                            double ox = off[pi * 2], oy = off[(pi * 2) + 1];
                            double lx, ly;
                            if (referenceDim == 2)
                            {
                                lx = reference[refBase] + (ox / w);
                                ly = reference[refBase + 1] + (oy / h);
                            }
                            else
                            {
                                lx = reference[refBase] + (ox / Points * reference[refBase + 2] * 0.5);
                                ly = reference[refBase + 1] + (oy / Points * reference[refBase + 3] * 0.5);
                            }

                            double weight = Math.Exp(logit[pi] - max) / denom;
                            double px = (lx * w) - 0.5, py = (ly * h) - 0.5;
                            int x0 = (int)Math.Floor(px), y0 = (int)Math.Floor(py);
                            double fx = px - x0, fy = py - y0;
                            for (int c = 0; c < dh; c++)
                            {
                                double Pixel(int yy, int xx) => yy < 0 || yy >= h || xx < 0 || xx >= w
                                    ? 0.0
                                    : projected[Starts[l] + (yy * w) + xx][(m * dh) + c];
                                double sample = (Pixel(y0, x0) * (1 - fx) * (1 - fy)) + (Pixel(y0, x0 + 1) * fx * (1 - fy))
                                    + (Pixel(y0 + 1, x0) * (1 - fx) * fy) + (Pixel(y0 + 1, x0 + 1) * fx * fy);
                                head[(m * dh) + c] += weight * sample;
                            }
                        }
                    }
                }

                for (int j = 0; j < D; j++)
                {
                    double s = outB[j];
                    for (int k = 0; k < D; k++) s += head[k] * outW[(k * D) + j];
                    result[(row * D) + j] = s;
                }
            }
        }

        return result;
    }
}
