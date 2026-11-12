using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
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
}
