using System.Linq;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The DARTS supernet (Liu et al. 2019): a softmax mixture over candidate operations on every edge, whose
/// architecture weights are trained by gradient and then discretised to the strongest operation per edge.
/// </summary>
public class SuperNetTests
{
    private static Tensor<double> Random(int rows, int cols, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < tensor.Length; i++)
            tensor.SetFlat(i, random.NextDouble() * 2 - 1);
        return tensor;
    }

    private static SuperNet<double> Create(out Tensor<double> input, out Tensor<double> target)
    {
        var superNet = new SuperNet<double>(new MobileNetSearchSpace<double>(), numNodes: 2);
        input = Random(4, 6, 1);
        target = Random(4, 6, 2);
        superNet.Predict(input); // sizes the operation weights for this input width
        return superNet;
    }

    [Fact]
    public void ArchitectureGradient_MatchesFiniteDifferencesOfTheValidationLoss()
    {
        var superNet = Create(out var input, out var target);
        superNet.BackwardArchitecture(input, target);
        var alpha = superNet.GetArchitectureParameters()[1];
        var gradient = superNet.GetArchitectureGradients()[1];

        const double h = 1e-5;
        foreach (var (row, col) in new[] { (0, 0), (1, 2), (0, 4) })
        {
            double original = alpha[row, col];
            alpha[row, col] = original + h;
            double plus = superNet.ComputeValidationLoss(input, target);
            alpha[row, col] = original - h;
            double minus = superNet.ComputeValidationLoss(input, target);
            alpha[row, col] = original;

            double numeric = (plus - minus) / (2 * h);
            Assert.True(System.Math.Abs(numeric - gradient[row, col]) <= 1e-6 + 1e-4 * System.Math.Abs(numeric),
                $"alpha[{row},{col}]: analytic {gradient[row, col]}, numeric {numeric}");
        }
    }

    [Fact]
    public void DerivedArchitecture_TakesTheStrongestOperationOnEachEdge()
    {
        var superNet = Create(out _, out _);
        var names = new MobileNetSearchSpace<double>().Operations;
        int conv3x3 = names.IndexOf("conv3x3");
        int seBlock = names.IndexOf("se_block");
        var alpha = superNet.GetArchitectureParameters();
        alpha[0][0, conv3x3] = 3.0;
        alpha[1][0, seBlock] = 3.0;
        alpha[1][1, conv3x3] = 3.0;

        var operations = superNet.DeriveArchitecture().Operations;

        Assert.Contains((0, 0, "conv3x3"), operations);
        Assert.Contains((1, 0, "se_block"), operations);
        Assert.Contains((1, 1, "conv3x3"), operations);
        Assert.Equal(3, operations.Count);
    }

    [Fact]
    public void TrainStep_LowersTheTrainingLoss()
    {
        var superNet = Create(out var input, out var target);
        double before = superNet.ComputeTrainingLoss(input, target);

        for (int step = 0; step < 30; step++)
            superNet.TrainStep(input, target, input, target);

        double after = superNet.ComputeTrainingLoss(input, target);
        Assert.True(after < before, $"training loss rose from {before} to {after}");
        Assert.All(superNet.GetArchitectureParameters().SelectMany(m => Enumerable.Range(0, m.Rows * m.Columns)
            .Select(i => m[i / m.Columns, i % m.Columns])), a => Assert.False(double.IsNaN(a)));
    }
}
