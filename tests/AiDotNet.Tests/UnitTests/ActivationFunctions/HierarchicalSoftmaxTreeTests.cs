using System;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActivationFunctions;

/// <summary>
/// Hierarchical softmax as a per-node binary tree (Morin and Bengio, 2005): one weight vector per internal node,
/// class probabilities that sum to 1 for any class count, and a Jacobian that matches finite differences. The
/// expected probabilities come from an independent recursive walk of the tree over the published node weights.
/// </summary>
public class HierarchicalSoftmaxTreeTests
{
    private static double[] Input(int length) => Enumerable.Range(0, length).Select(i => 0.4 * i - 0.7 + 0.05 * i * i).ToArray();

    /// <summary>
    /// Walks the balanced tree in pre-order: a node covering classes [low, high) sends the first
    /// ceil((high - low) / 2) left with probability 1 - sigmoid(w . x) and the rest right with sigmoid(w . x).
    /// </summary>
    private static double[] ReferenceProbabilities(Tensor<double> nodeWeights, double[] x, int numClasses)
    {
        int dim = x.Length;
        double[] weights = nodeWeights.ToArray();
        var probabilities = new double[numClasses];
        int next = 0;

        void Walk(int low, int high, double mass)
        {
            if (high - low == 1)
            {
                probabilities[low] = mass;
                return;
            }

            int node = next++;
            double dot = 0.0;
            for (int j = 0; j < dim; j++) dot += x[j] * weights[node * dim + j];
            double right = 1.0 / (1.0 + Math.Exp(-dot));
            int middle = low + (high - low + 1) / 2;
            Walk(low, middle, mass * (1.0 - right));
            Walk(middle, high, mass * right);
        }

        Walk(0, numClasses, 1.0);
        Assert.Equal(numClasses - 1, next);
        return probabilities;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(13)]
    [InlineData(17)]
    [InlineData(100)]
    public void Probabilities_SumToOne_ForAnyClassCount(int numClasses)
    {
        var activation = new HierarchicalSoftmaxActivation<double>(numClasses);

        double[] probabilities = activation.Activate(new Vector<double>(Input(numClasses))).ToArray();

        Assert.Equal(numClasses, probabilities.Length);
        Assert.All(probabilities, p => Assert.InRange(p, 0.0, 1.0));
        Assert.Equal(1.0, probabilities.Sum(), 12);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(11)]
    public void Probabilities_SumToOne_Float(int numClasses)
    {
        var activation = new HierarchicalSoftmaxActivation<float>(numClasses);

        float[] probabilities = activation.Activate(new Vector<float>(Input(numClasses).Select(v => (float)v).ToArray())).ToArray();

        Assert.Equal(1.0, probabilities.Sum(p => (double)p), 5);
    }

    [Theory]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    [InlineData(5, 5)]
    [InlineData(8, 8)]
    [InlineData(13, 13)]
    [InlineData(6, 4)]
    [InlineData(4, 9)]
    public void Probabilities_AreTheProductOfTheNodeDecisionsOnEachPath(int numClasses, int inputDim)
    {
        var activation = new HierarchicalSoftmaxActivation<double>(numClasses);
        double[] x = Input(inputDim);

        double[] actual = activation.Activate(new Vector<double>(x)).ToArray();
        // Read after the forward pass: a new input dimension draws new weights
        var weights = activation.NodeWeightsTensor;

        // One weight vector per internal node, not one per tree depth
        Assert.Equal(new[] { numClasses - 1, inputDim }, weights.Shape.ToArray());
        Assert.All(weights.ToArray(), w => Assert.InRange(w, -0.5, 0.5));
        Assert.Contains(weights.ToArray(), w => w != 0.0);

        double[] expected = ReferenceProbabilities(weights, x, numClasses);
        for (int c = 0; c < numClasses; c++)
            Assert.True(Math.Abs(expected[c] - actual[c]) <= 1e-12, $"class {c}: expected {expected[c]:R} but was {actual[c]:R}");
    }

    [Fact]
    public void SingleClass_HasProbabilityOne_AndNoNodes()
    {
        var activation = new HierarchicalSoftmaxActivation<double>(1);

        Assert.Equal(new[] { 1.0 }, activation.Activate(new Vector<double>(new[] { 3.5 })).ToArray());
        Assert.Equal(new[] { 0, 1 }, activation.NodeWeightsTensor.Shape.ToArray());
        var jacobian = activation.Derivative(new Vector<double>(new[] { 3.5 }));
        Assert.Equal(1, jacobian.Rows);
        Assert.Equal(1, jacobian.Columns);
        Assert.Equal(0.0, jacobian[0, 0]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void Constructor_RejectsFewerThanOneClass(int numClasses)
        => Assert.Throws<ArgumentOutOfRangeException>(() => new HierarchicalSoftmaxActivation<double>(numClasses));

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    [InlineData(11)]
    public void Jacobian_MatchesCentralFiniteDifferences_AndItsColumnsSumToZero(int numClasses)
    {
        var activation = new HierarchicalSoftmaxActivation<double>(numClasses);
        double[] x = Input(numClasses).Select(v => v * 0.3).ToArray();

        var jacobian = activation.Derivative(new Vector<double>(x));

        Assert.Equal(numClasses, jacobian.Rows);
        Assert.Equal(numClasses, jacobian.Columns);
        const double step = 1e-6;
        bool anyNonZero = false;
        for (int j = 0; j < numClasses; j++)
        {
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[j] += step;
            minus[j] -= step;
            double[] high = activation.Activate(new Vector<double>(plus)).ToArray();
            double[] low = activation.Activate(new Vector<double>(minus)).ToArray();

            double columnSum = 0.0;
            for (int c = 0; c < numClasses; c++)
            {
                double numerical = (high[c] - low[c]) / (2 * step);
                Assert.True(Math.Abs(numerical - jacobian[c, j]) <= 1e-8, $"d p[{c}] / d x[{j}]: finite difference {numerical:R}, jacobian {jacobian[c, j]:R}");
                columnSum += jacobian[c, j];
                anyNonZero |= Math.Abs(jacobian[c, j]) > 1e-6;
            }

            // The probabilities always sum to 1, so their derivatives sum to 0
            Assert.True(Math.Abs(columnSum) <= 1e-12, $"column {j} sums to {columnSum:R}");
        }

        Assert.True(anyNonZero, "the Jacobian is identically zero");
    }

    [Fact]
    public void TensorOverload_AgreesWithTheVectorOverload_RowByRow()
    {
        const int numClasses = 6, inputDim = 6;
        var activation = new HierarchicalSoftmaxActivation<double>(numClasses);
        double[][] rows =
        {
            Input(inputDim),
            Input(inputDim).Select(v => -v).ToArray(),
            Input(inputDim).Select(v => 0.1 * v + 0.3).ToArray(),
            new double[inputDim],
        };

        var batch = activation.Activate(new Tensor<double>(rows.SelectMany(r => r).ToArray(), new[] { 2, 2, inputDim }));

        Assert.Equal(new[] { 2, 2, numClasses }, batch.Shape.ToArray());
        double[] flat = batch.ToArray();
        for (int r = 0; r < rows.Length; r++)
        {
            double[] single = activation.Activate(new Vector<double>(rows[r])).ToArray();
            // The vector overload is the tensor overload on a rank-1 tensor
            Assert.Equal(single, activation.Activate(new Tensor<double>(rows[r], new[] { inputDim })).ToArray());
            for (int c = 0; c < numClasses; c++)
                Assert.True(Math.Abs(single[c] - flat[r * numClasses + c]) <= 1e-14, $"row {r}, class {c}: {single[c]:R} vs {flat[r * numClasses + c]:R}");
            Assert.Equal(1.0, single.Sum(), 12);
        }

        // All-zero input: every node is undecided, so a class at depth d gets 2^-d
        double[] undecided = flat.Skip(3 * numClasses).ToArray();
        Assert.All(undecided, p => Assert.True(Math.Abs(p - 0.25) <= 1e-15 || Math.Abs(p - 0.125) <= 1e-15, $"unexpected probability {p:R}"));
        Assert.Equal(2, undecided.Count(p => Math.Abs(p - 0.25) <= 1e-15));
    }

    [Theory]
    [InlineData(1e3)]
    [InlineData(1e6)]
    public void SaturatedNodes_StayFinite_AndStillSumToOne(double scale)
    {
        const int numClasses = 7;
        var activation = new HierarchicalSoftmaxActivation<double>(numClasses);

        double[] probabilities = activation.Activate(new Vector<double>(Input(numClasses).Select(v => v * scale).ToArray())).ToArray();

        Assert.All(probabilities, p => Assert.False(double.IsNaN(p) || double.IsInfinity(p), "a probability is not finite"));
        Assert.Equal(1.0, probabilities.Sum(), 9);
    }

    [Fact]
    public void NodeWeightsTensor_IsACopy()
    {
        var activation = new HierarchicalSoftmaxActivation<double>(4);
        var input = new Vector<double>(Input(4));
        double[] before = activation.Activate(input).ToArray();

        var weights = activation.NodeWeightsTensor;
        for (int n = 0; n < 3; n++)
            for (int j = 0; j < 4; j++) weights[n, j] = 100.0;

        Assert.Equal(before, activation.Activate(input).ToArray());
    }
}
