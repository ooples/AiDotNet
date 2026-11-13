using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Reference tests for the manual backprop path of <see cref="SparseLinearLayer{T}"/>
/// (ComputeGradients / UpdateParameters, driven by SparseNeuralNetwork's training loop).
/// </summary>
public sealed class SparseLinearLayerManualGradientTests
{
    [Fact]
    public void ComputeGradients_SigmoidActivation_MatchesFiniteDifferences()
    {
        // Sigmoid's derivative belongs at the pre-activation z: s(z)(1 - s(z)). Evaluating it at the
        // output instead gives s(s(z))(1 - s(s(z))), which the central difference below rejects.
        var layer = new SparseLinearLayer<double>(3, 2, sparsity: 0.0, activationFunction: new SigmoidActivation<double>());
        var x = new Tensor<double>([0.7, -1.3, 2.1], [1, 3]);

        const double h = 1e-6;
        var expected = new double[3];
        for (int i = 0; i < 3; i++)
        {
            var plus = x.Clone();
            plus[0, i] += h;
            var minus = x.Clone();
            minus[0, i] -= h;
            expected[i] = (Sum(layer.Forward(plus)) - Sum(layer.Forward(minus))) / (2 * h);
        }

        layer.SetTrainingMode(true);
        layer.Forward(x);
        var inputGradient = layer.ComputeGradients(new Tensor<double>([1.0, 1.0], [1, 2]));

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(expected[i], inputGradient[0, i], precision: 6);
        }
    }

    [Fact]
    public void UpdateParameters_ChangesTheStoredSparseWeights()
    {
        // SparseTensor.Values returns a copy, so writing into it never reached the weights.
        // With identity activation and dL/dy = 1: dW[o, i] = x[i] and dB[o] = 1.
        var layer = new SparseLinearLayer<double>(3, 2, sparsity: 0.0, activationFunction: new IdentityActivation<double>());
        var x = new Tensor<double>([1.0, 2.0, 3.0], [1, 3]);
        var before = layer.GetParameters();
        Assert.Equal(8, before.Length); // 6 stored weights (row-major) + 2 biases

        layer.SetTrainingMode(true);
        layer.Forward(x);
        layer.ComputeGradients(new Tensor<double>([1.0, 1.0], [1, 2]));
        layer.UpdateParameters(0.1);
        var after = layer.GetParameters();

        for (int k = 0; k < 6; k++)
        {
            Assert.Equal(before[k] - 0.1 * x[0, k % 3], after[k], precision: 12);
        }

        Assert.Equal(before[6] - 0.1, after[6], precision: 12);
        Assert.Equal(before[7] - 0.1, after[7], precision: 12);
    }

    [Fact]
    public void ComputeGradients_SparsePattern_MatchesTheDenseReferenceAtEveryStoredWeight()
    {
        // With identity activation: dW[o, i] = sum_b g[b, o] x[b, i] sampled at each stored (o, i), and
        // dX[b, i] = sum_o g[b, o] W[o, i] over the stored weights only.
        var layer = new SparseLinearLayer<double>(4, 3, sparsity: 0.5, activationFunction: new IdentityActivation<double>());
        var x = new Tensor<double>([0.5, -1.0, 2.0, 0.25, 1.5, 0.75, -0.5, -2.0, -1.25, 0.1, 0.6, 3.0], [3, 4]);
        var g = new Tensor<double>([1.0, -2.0, 0.5, 0.3, 0.7, -1.1, -0.4, 2.2, 1.6], [3, 3]);

        layer.SetTrainingMode(true);
        layer.Forward(x);
        var inputGradient = layer.ComputeGradients(g);

        var weights = typeof(SparseLinearLayer<double>)
            .GetField("_weights", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)?
            .GetValue(layer) as SparseTensor<double>
            ?? throw new Xunit.Sdk.XunitException("SparseLinearLayer has no sparse _weights field.");
        var weightGradient = typeof(SparseLinearLayer<double>)
            .GetField("_weightsGradient", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)?
            .GetValue(layer) as Tensor<double>
            ?? throw new Xunit.Sdk.XunitException("ComputeGradients stored no weight gradient.");

        int[] rows = weights.RowIndices;
        int[] columns = weights.ColumnIndices;
        double[] values = weights.Values;
        Assert.True(values.Length > 0 && values.Length < 12, "The layer should store a strict subset of weights.");
        Assert.Equal(values.Length, weightGradient.Length);

        var expectedInputGradient = new double[3, 4];
        for (int k = 0; k < values.Length; k++)
        {
            double expectedWeightGradient = 0;
            for (int b = 0; b < 3; b++)
            {
                expectedWeightGradient += g[b, rows[k]] * x[b, columns[k]];
                expectedInputGradient[b, columns[k]] += g[b, rows[k]] * values[k];
            }

            Assert.Equal(expectedWeightGradient, weightGradient.GetFlat(k), precision: 12);
        }

        Assert.Equal(new[] { 3, 4 }, inputGradient.Shape.ToArray());
        for (int b = 0; b < 3; b++)
        {
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(expectedInputGradient[b, i], inputGradient[b, i], precision: 12);
            }
        }
    }

    private static double Sum(Tensor<double> tensor)
    {
        double total = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            total += tensor.GetFlat(i);
        }

        return total;
    }
}
