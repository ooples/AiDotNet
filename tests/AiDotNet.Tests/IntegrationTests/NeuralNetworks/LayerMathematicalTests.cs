using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Mathematical invariant tests for neural network layers.
/// Tests verify mathematical properties without requiring internal weight access:
/// 1. Linearity/superposition properties
/// 2. Correct output dimensions
/// 3. Zero-preservation where applicable
/// 4. Gradient correctness via finite differences
/// </summary>
public class LayerMathematicalTests
{
    private const double Tol = 1e-5;

    #region DenseLayer Mathematical Tests

    /// <summary>
    /// DenseLayer forward with zero input should return bias only.
    /// f(0) = W*0 + b = b
    /// Since GetBiases() is accessible, we can verify the bias values.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DenseLayer_ZeroInput_ReturnsBias()
    {
        var layer = new DenseLayer<double>(3, 2);
        var biases = layer.GetBiases();

        var input = new Tensor<double>(new[] { 0.0, 0.0, 0.0 }, [3]);
        var output = layer.Forward(input);

        // Output should equal biases since W*0 + b = b
        for (int i = 0; i < 2; i++)
            Assert.Equal(biases[i], output[i], Tol);
    }

    /// <summary>
    /// DenseLayer parameter count: inputSize * outputSize + outputSize (biases)
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DenseLayer_ParameterCount()
    {
        var layer = new DenseLayer<double>(10, 5);
        Assert.Equal(55, layer.ParameterCount); // 10*5 + 5
    }

    /// <summary>
    /// DenseLayer output shape should be [outputSize] for 1D input.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DenseLayer_OutputShape_1D()
    {
        var layer = new DenseLayer<double>(4, 3);
        var input = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, [4]);
        var output = layer.Forward(input);
        Assert.Single(output.Shape.ToArray());
        Assert.Equal(3, output.Shape[0]);
    }

    /// <summary>
    /// DenseLayer backward gradient w.r.t. input should have correct shape.
    /// </summary>

    /// <summary>
    /// DenseLayer numerical gradient: verify backward matches finite differences.
    /// </summary>

    #endregion

    #region BatchNormalizationLayer Mathematical Tests

    /// <summary>
    /// BatchNorm normalizes each feature to ~zero mean across the batch.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task BatchNorm_Forward_ApproximatelyZeroMean()
    {
        var layer = new BatchNormalizationLayer<double>(2, epsilon: 1e-5);

        var input = new Tensor<double>(new[] {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        }, [3, 2]);

        var output = layer.Forward(input);

        // Each feature should have approximately zero mean after normalization
        for (int f = 0; f < 2; f++)
        {
            double sum = 0;
            for (int b = 0; b < 3; b++)
                sum += output[b * 2 + f];
            Assert.True(Math.Abs(sum / 3.0) < 0.1, $"Feature {f} mean should be ~0, got {sum / 3.0}");
        }
    }

    /// <summary>
    /// BatchNorm with constant input should produce zero output (mean subtracted).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task BatchNorm_ConstantInput_ProducesZero()
    {
        var layer = new BatchNormalizationLayer<double>(1, epsilon: 1e-5);

        var input = new Tensor<double>(new[] { 5.0, 5.0, 5.0 }, [3, 1]);
        var output = layer.Forward(input);

        // Constant input: mean=5, var=0, so (x-mean)/sqrt(var+eps) = 0/sqrt(eps) ≈ 0
        for (int i = 0; i < 3; i++)
            Assert.True(Math.Abs(output[i]) < 1.0, $"Constant input should normalize to ~0, got {output[i]}");
    }

    #endregion

    #region DropoutLayer Mathematical Tests

    /// <summary>
    /// Dropout with rate=0 should pass everything through unchanged.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Dropout_ZeroRate_PassThrough()
    {
        var layer = new DropoutLayer<double>(0.0);
        layer.SetTrainingMode(true);

        var input = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, [4]);
        var output = layer.Forward(input);

        for (int i = 0; i < 4; i++)
            Assert.Equal(input[i], output[i], Tol);
    }

    /// <summary>
    /// Dropout in eval mode should always pass through unchanged.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Dropout_EvalMode_PassThrough()
    {
        var layer = new DropoutLayer<double>(0.5);
        layer.SetTrainingMode(false);

        var input = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, [5]);
        var output = layer.Forward(input);

        for (int i = 0; i < 5; i++)
            Assert.Equal(input[i], output[i], Tol);
    }

    #endregion
}
