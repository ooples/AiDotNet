using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
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
