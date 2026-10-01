using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML;

/// <summary>
/// SuperNet as differentiable architecture search actually defines it (Liu et al. 2019).
/// </summary>
/// <remarks>
/// <para>
/// Train threw; every gradient was a central finite difference, two forward passes per scalar
/// parameter; and the candidate operations were constant scalings wearing convolution and pooling
/// names ("3x3 Conv (simplified as weighted pass)", "MaxPool" = x * 0.9). Operations were chosen by
/// index, so a search space whose names differ from the default one ran the wrong operation and
/// reported the wrong name. These tests hold the rewrite to the paper.
/// </para>
/// </remarks>
public class SuperNetTrainingTests
{
    private static Tensor<double> Random(int rows, int columns, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(new[] { rows, columns });
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    private static double[] Flatten(System.Collections.Generic.IEnumerable<Matrix<double>> matrices)
        => matrices.SelectMany(m => Enumerable.Range(0, m.Rows).SelectMany(r => Enumerable.Range(0, m.Columns).Select(c => m[r, c]))).ToArray();

    private static double[] Flatten(System.Collections.Generic.IEnumerable<Vector<double>> vectors)
        => vectors.SelectMany(v => Enumerable.Range(0, v.Length).Select(i => v[i])).ToArray();

    /// <summary>Selects one operation on every edge, the way NasAutoMLModelBase applies a found architecture.</summary>
    /// <remarks>
    /// The margin must make every other operation's softmax weight vanish at the tests' tolerances: with
    /// +/-10 each still carried e^-20 (about 2e-9), so a 5x5 convolution leaked into positions a 3x3 one
    /// cannot reach. At +/-50 the leak is e^-100 (about 4e-44).
    /// </remarks>
    private static void Select(SuperNet<double> supernet, int operation)
    {
        foreach (var alpha in supernet.GetArchitectureParameters())
        {
            for (int r = 0; r < alpha.Rows; r++)
            {
                for (int c = 0; c < alpha.Columns; c++) alpha[r, c] = c == operation ? 50.0 : -50.0;
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Train_MovesBothTheArchitectureAndTheWeights()
    {
        await Task.Yield();
        var supernet = new SuperNet<double>(new SearchSpaceBase<double>(), numNodes: 2);
        var alphaBefore = Flatten(supernet.GetArchitectureParameters());
        var weightsBefore = Flatten(supernet.GetWeightParameters().Values);

        supernet.Train(Random(4, 8, 1), Random(4, 8, 2));

        var alphaAfter = Flatten(supernet.GetArchitectureParameters());
        var weightsAfter = Flatten(supernet.GetWeightParameters().Values);
        Assert.True(weightsBefore.Length > 0, "The candidate operations own no weights.");
        Assert.True(alphaBefore.Zip(alphaAfter, (a, b) => Math.Abs(a - b)).Sum() > 0,
            "One DARTS step left the architecture parameters unchanged.");
        Assert.True(weightsBefore.Zip(weightsAfter, (a, b) => Math.Abs(a - b)).Sum() > 0,
            "One DARTS step left the operation weights unchanged.");
        Assert.All(alphaAfter.Concat(weightsAfter), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }

    [Fact(Timeout = 120000)]
    public async Task BackwardArchitecture_MatchesCentralDifferences()
    {
        await Task.Yield();
        var supernet = new SuperNet<double>(new SearchSpaceBase<double>(), numNodes: 2);
        var x = Random(3, 6, 11);
        var y = Random(3, 6, 12);

        supernet.BackwardArchitecture(x, y);
        var analytic = supernet.GetArchitectureGradients().Select(g => g.Clone()).ToList();

        // Double precision, so central differences are a trustworthy reference (float32 is not).
        const double h = 1e-6;
        var alphas = supernet.GetArchitectureParameters();
        for (int n = 0; n < alphas.Count; n++)
        {
            for (int r = 0; r < alphas[n].Rows; r++)
            {
                for (int c = 0; c < alphas[n].Columns; c++)
                {
                    double original = alphas[n][r, c];
                    alphas[n][r, c] = original + h;
                    double plus = supernet.ComputeValidationLoss(x, y);
                    alphas[n][r, c] = original - h;
                    double minus = supernet.ComputeValidationLoss(x, y);
                    alphas[n][r, c] = original;

                    double numeric = (plus - minus) / (2 * h);
                    Assert.True(Math.Abs(numeric - analytic[n][r, c]) <= 1e-5 + 1e-4 * Math.Abs(numeric),
                        $"alpha[{n}][{r},{c}]: tape gradient {analytic[n][r, c]:E6} against central difference {numeric:E6}.");
                }
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MaxPool_SpreadsAnImpulseToItsNeighbours()
    {
        await Task.Yield();
        var space = new SearchSpaceBase<double>();
        var supernet = new SuperNet<double>(space, numNodes: 1);
        Select(supernet, space.Operations.IndexOf("maxpool3x3"));

        var impulse = new Tensor<double>(new[] { 1, 7 });
        impulse[0, 3] = 5.0;
        var output = supernet.Predict(impulse);

        // A width-3 max pool sees the impulse from positions 2, 3 and 4 and nowhere else. The old
        // "max pool" was x * 0.9, which cannot move a value to a neighbouring position.
        for (int f = 0; f < 7; f++)
        {
            double expected = f is >= 2 and <= 4 ? 5.0 : 0.0;
            Assert.Equal(expected, output[0, f], 6);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Convolution_MixesNeighbouringFeatures()
    {
        await Task.Yield();
        var space = new SearchSpaceBase<double>();
        var supernet = new SuperNet<double>(space, numNodes: 1);
        Select(supernet, space.Operations.IndexOf("conv3x3"));

        var impulse = new Tensor<double>(new[] { 1, 9 });
        impulse[0, 4] = 1.0;
        var output = supernet.Predict(impulse);

        // A width-3 kernel's response to an impulse is the kernel itself, centred on the impulse: at most
        // three non-zero outputs, and they must include a neighbour, which no per-feature scaling produces.
        int nonZero = Enumerable.Range(0, 9).Count(f => Math.Abs(output[0, f]) > 1e-12);
        Assert.InRange(nonZero, 1, 3);
        Assert.True(Math.Abs(output[0, 3]) > 1e-12 || Math.Abs(output[0, 5]) > 1e-12,
            "The 3x3 convolution left both neighbours of the impulse at zero.");
        for (int f = 0; f < 9; f++)
        {
            if (f < 3 || f > 5) Assert.Equal(0.0, output[0, f], 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task MobileNetSearchSpace_TrainsEndToEnd()
    {
        await Task.Yield();
        // The builder's default NAS path hands SuperNet the MobileNet search space.
        var supernet = new SuperNet<double>(new MobileNetSearchSpace<double>(), numNodes: 2);
        var weightsBefore = Flatten(supernet.GetWeightParameters().Values);

        supernet.Train(Random(2, 8, 21), Random(2, 8, 22));
        var output = supernet.Predict(Random(2, 8, 23));

        Assert.Equal(new[] { 2, 8 }, output.Shape.ToArray());
        Assert.All(Enumerable.Range(0, output.Length), i => Assert.False(double.IsNaN(output[i])));
        var weightsAfter = Flatten(supernet.GetWeightParameters().Values);
        Assert.True(weightsBefore.Zip(weightsAfter, (a, b) => Math.Abs(a - b)).Sum() > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task DeriveArchitecture_ReportsTheSearchSpacesOwnNames()
    {
        await Task.Yield();
        var space = new MobileNetSearchSpace<double>();
        var supernet = new SuperNet<double>(space, numNodes: 2);
        Select(supernet, space.Operations.IndexOf("conv1x1"));

        var architecture = supernet.DeriveArchitecture();

        // Names used to be looked up by index in a fixed five-name table, so MobileNet's conv1x1 was
        // reported as "conv3x3" and every operation past the fifth as "identity".
        Assert.All(architecture.Operations, op => Assert.Equal("conv1x1", op.Operation));
    }

    [Fact(Timeout = 120000)]
    public async Task UnsupportedOperation_IsRejectedAtConstruction()
    {
        await Task.Yield();
        // Attention over a feature vector has no meaning in this cell; it used to run silently as identity.
        var ex = Assert.Throws<NotSupportedException>(() => new SuperNet<double>(new TransformerSearchSpace<double>(), numNodes: 2));
        Assert.Contains("self_attention", ex.Message);
    }
}
