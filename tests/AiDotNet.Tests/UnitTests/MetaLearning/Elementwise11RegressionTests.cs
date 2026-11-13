using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

public class Elementwise11RegressionTests
{
    [Fact]
    public void HalfModulation_LargeFeatureCount_PreservesMeanRatio()
    {
        var engine = AiDotNetEngine.Current;
        var raw = new Vector<Half>(Enumerable.Repeat((Half)1, 70000).ToArray());
        var adapted = new Vector<Half>(Enumerable.Repeat((Half)2, 70000).ToArray());
        var factor = ParameterModulation.MeanClampedRatio(engine, raw, adapted, 0.5, 2.0, 1e-10);
        Assert.NotNull(factor);
        Assert.Equal((Half)2, factor[0]);
        Assert.Equal((Half)0.75, ParameterModulation.MeanAbsGate(engine, raw)[0]);
    }

    [Fact]
    public void Labels_ClassBound_DoesNotRoundIntoValidHalfLabel()
    {
        var labels = new Vector<Half>(new[] { (Half)2048 });
        Assert.Equal((Half)2048, ClassifierOutputs<Half>.Labels(labels, 2049)[0]);
        Assert.Throws<ArgumentException>(() => ClassifierOutputs<Half>.Labels(labels, 2048));
    }

    [Fact]
    public void Labels_ReportTheFirstInvalidPosition()
    {
        // Position 1 is out of range (one failed check); position 2 is negative and fractional (two). The scalar
        // scan named position 1; an arg-max over the failure counts named position 2.
        var labels = new Vector<double>(new[] { 0.0, 5.0, -0.5 });
        var error = Assert.Throws<ArgumentException>(() => ClassifierOutputs<double>.Labels(labels, 3));
        Assert.Contains("Label 5 at position 1", error.Message);
    }

    [Fact]
    public void ProbabilityLoss_RejectsALabelBeyondTheProbabilityColumns()
    {
        // Two probability columns: label 2 matches no column and would silently leave an all-zero one-hot row.
        var probabilities = new Tensor<double>(new[] { 0.6, 0.4, 0.3, 0.7 }, new[] { 2, 2 });
        var labels = new Vector<double>(new[] { 0.0, 2.0 });
        Assert.Throws<ArgumentException>(() => ClassifierOutputs<double>.ProbabilityLoss(
            new MeanSquaredErrorLoss<double>(), probabilities, labels));
    }

    [Fact]
    public void StackRows_PadsEachNarrowerQueryTensorRow()
    {
        // Support [1, 3], query [2, 2]: each query row is zero-padded on its own, as the matrix branch does.
        var support = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 1, 3 });
        var query = new Tensor<double>(new[] { 11.0, 12.0, 21.0, 22.0 }, new[] { 2, 2 });
        var stacked = ClassifierOutputs<double>.StackRows(support, query);
        Assert.Equal(new[] { 3, 3 }, stacked.Shape.ToArray());
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 11.0, 12.0, 0.0, 21.0, 22.0, 0.0 }, stacked.ToArray());
    }

    [Fact]
    public void HalfClassificationError_LargeCount_DoesNotOverflow()
    {
        var predicted = new Vector<Half>(Enumerable.Repeat((Half)1, 70000).ToArray());
        var labels = new Vector<Half>(70000);
        Assert.Equal((Half)1, ClassifierOutputs<Half>.ProbabilityLoss(
            new MeanSquaredErrorLoss<Half>(), predicted, labels));
        Assert.Equal((Half)1, ClassifierOutputs<Half>.ProbabilityLoss(
            new MeanSquaredErrorLoss<Half>(), new Vector<Half>(0), labels));
    }

    [Fact]
    public void HalfNearestNeighbors_LargeTieCount_SelectsExactlyKOtherNodes()
    {
        const int n = 1026;
        var distances = Tensor<Half>.CreateDefault(new[] { n, n }, (Half)1);
        var mask = NearestNeighborGraph.NearestMask(AiDotNetEngine.Current, distances, 1024);
        for (int row = 0; row < n; row++)
        {
            Assert.Equal((Half)0, mask[row, row]);
            int count = 0;
            for (int column = 0; column < n; column++) count += (int)mask[row, column];
            Assert.Equal(1024, count);
        }
        Assert.Equal((Half)1, mask[0, 1024]);
        Assert.Equal((Half)0, mask[0, 1025]);
    }

    [Fact]
    public void NearestNeighbors_InfiniteDistance_StillSelectsOtherNode()
    {
        var distances = new Tensor<double>(new[] { 0.0, double.PositiveInfinity, double.PositiveInfinity, 0.0 }, new[] { 2, 2 });
        var mask = NearestNeighborGraph.NearestMask(AiDotNetEngine.Current, distances, 1);
        Assert.Equal(new[] { 0.0, 1.0, 1.0, 0.0 }, mask.ToArray());
    }

    [Theory]
    [InlineData(1e20f)]
    [InlineData(1e-25f)]
    public void AnilDifferenceStep_UsesWideSquaredNorm(float magnitude)
    {
        var type = typeof(ANILAlgorithm<float, Matrix<float>, Tensor<float>>);
        var method = type.GetMethod("DifferenceStep", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Missing finite-difference step helper.");
        var value = new Vector<float>(new[] { magnitude });
        var empty = new Vector<float>(0);
        double step = (double)(method.Invoke(null, new object[] { value, empty, value, empty })
            ?? throw new InvalidOperationException("Missing step."));
        double expected = 1e-5 * (1.0 + magnitude) / magnitude;
        Assert.InRange(step / expected, 0.999999, 1.000001);
    }

    [Fact]
    public void MclHalfContrastiveLoss_IdenticalProjections_EqualsLogTwo()
    {
        var model = new NeuralNetwork<Half>(new NeuralNetworkArchitecture<Half>(
            InputType.OneDimensional, NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple, inputSize: 2, outputSize: 2));
        var algorithm = new MCLAlgorithm<Half, Tensor<Half>, Tensor<Half>>(
            new MCLOptions<Half, Tensor<Half>, Tensor<Half>>(model));
        var method = algorithm.GetType().GetMethod("SupervisedContrastiveLoss", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("Missing contrastive loss.");
        var projections = Tensor<Half>.CreateDefault(new[] { 3, 2 }, (Half)1);
        var loss = (Tensor<Half>)(method.Invoke(algorithm, new object[] { projections, new[] { 0, 0, 0 } })
            ?? throw new InvalidOperationException("Missing loss."));
        Assert.InRange((double)loss[0], Math.Log(2) - 0.02, Math.Log(2) + 0.02);
    }

    // RBF kernel of 4 close points (length 1) plus 1e-3 noise: valid SPD but cond ~4e3, the DKT GP system
    private static double[] IllConditionedKernel()
    {
        var x = new[] { 0.0, 0.25, 0.5, 0.75 };
        var a = new double[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                a[i * 4 + j] = Math.Exp(-(x[i] - x[j]) * (x[i] - x[j]) / 2) + (i == j ? 1e-3 : 0);
        return a;
    }

    // Double reference solve of the system exactly as T stores it
    private static double[] ReferenceSolve<T>(Tensor<T> a, Tensor<T> b)
        => LinalgDispatch<double>.Solve(a.Cast<double>(), b.Cast<double>()).ToArray();

    private static void AssertMatchesReference<T>(Tensor<T> a, Tensor<T> b, double relativeTolerance)
    {
        var reference = ReferenceSolve(a, b);
        var solution = ExampleRows.SolveSymmetricPositiveDefinite(a, b).Cast<double>().ToArray();
        double scale = reference.Max(v => Math.Abs(v));
        Assert.True(scale > 50, "Fixture should give a large GP weight vector.");
        Assert.Equal(reference.Length, solution.Length);
        for (int i = 0; i < reference.Length; i++)
            Assert.InRange(solution[i], reference[i] - relativeTolerance * scale, reference[i] + relativeTolerance * scale);
    }

    [Fact]
    public void HalfSpdSolve_IllConditionedKernel_MatchesDoubleReference()
    {
        var a = new Tensor<Half>(IllConditionedKernel().Select(v => (Half)v).ToArray(), new[] { 4, 4 });
        var b = new Tensor<Half>(new[] { (Half)1, (Half)0, (Half)0, (Half)1 }, new[] { 4, 1 });
        AssertMatchesReference(a, b, 1e-3);
    }

    [Fact]
    public void BFloat16SpdSolve_IllConditionedKernel_MatchesDoubleReference()
    {
        var a = new Tensor<BFloat16>(IllConditionedKernel().Select(v => BFloat16.FromFloat((float)v)).ToArray(),
            new[] { 4, 4 });
        var b = new Tensor<BFloat16>(new[] { BFloat16.One, BFloat16.Zero, BFloat16.Zero, BFloat16.One },
            new[] { 4, 1 });
        AssertMatchesReference(a, b, 8e-3);
    }

    [Fact]
    public void DecimalSpdSolve_IllConditionedKernel_MatchesDoubleReference()
    {
        var a = new Tensor<decimal>(IllConditionedKernel().Select(v => (decimal)v).ToArray(), new[] { 4, 4 });
        var b = new Tensor<decimal>(new[] { 1m, 0m, 0m, 1m }, new[] { 4, 1 });
        AssertMatchesReference(a, b, 1e-9);
    }
}
