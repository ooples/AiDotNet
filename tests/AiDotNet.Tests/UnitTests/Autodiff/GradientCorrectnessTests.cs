using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

/// <summary>
/// Tests to verify that autodiff gradients match manual gradient implementations.
/// </summary>
public class GradientCorrectnessTests
{
    private const double Tolerance = 1e-4; // Tolerance for gradient comparisons

    [Fact]
    public void DenseLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int inputSize = 10;
        const int outputSize = 5;
        const int batchSize = 3;

        var layer = new DenseLayer<float>(inputSize, outputSize, new ReLUActivation<float>());

        // Create test input
        var input = new Tensor<float>(new[] { batchSize, inputSize });
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 2 - 1); // Range: [-1, 1]
        }

        // Create test output gradient
        var outputGradient = new Tensor<float>(new[] { batchSize, outputSize });
        for (int i = 0; i < outputGradient.Length; i++)
        {
            outputGradient[i] = (float)(random.NextDouble() * 2 - 1);
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        var forwardOutput = layer.Forward(input);
        var manualInputGradient = layer.Backward(outputGradient);
        var manualWeightGradient = layer.GetParameters(); // Simplified - would need accessor

        // Reset layer state
        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        forwardOutput = layer.Forward(input);
        var autodiffInputGradient = layer.Backward(outputGradient);
        var autodiffWeightGradient = layer.GetParameters(); // Simplified

        // Assert - Gradients should be nearly identical
        Assert.Equal(manualInputGradient.Shape, autodiffInputGradient.Shape);

        // Compare input gradients element-wise
        for (int i = 0; i < manualInputGradient.Length; i++)
        {
            var diff = Math.Abs(manualInputGradient[i] - autodiffInputGradient[i]);
            Assert.True(diff < Tolerance,
                $"Input gradient mismatch at index {i}: manual={manualInputGradient[i]}, autodiff={autodiffInputGradient[i]}, diff={diff}");
        }
    }

    [Fact]
    public void ActivationLayer_ReLU_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 4, 8 }; // Batch size 4, 8 features
        var layer = new ActivationLayer<float>(shape, new ReLUActivation<float>());

        // Create test input with mix of positive and negative values
        var input = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 4 - 2); // Range: [-2, 2]
        }

        // Create test output gradient
        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            outputGradient[i] = (float)(random.NextDouble());
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        var forwardOutput = layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        // Reset layer state
        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        forwardOutput = layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert - Gradients should be nearly identical
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
            Assert.True(diff < Tolerance,
                $"Gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}, diff={diff}");
        }
    }

    [Fact]
    public void ActivationLayer_Sigmoid_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 2, 6 };
        var layer = new ActivationLayer<float>(shape, new SigmoidActivation<float>());

        var input = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            outputGradient[i] = (float)(random.NextDouble());
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
            Assert.True(diff < Tolerance,
                $"Sigmoid gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}, diff={diff}");
        }
    }

    [Fact]
    public void ActivationLayer_Tanh_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 3, 4 };
        var layer = new ActivationLayer<float>(shape, new TanhActivation<float>());

        var input = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            outputGradient[i] = (float)(random.NextDouble());
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
            Assert.True(diff < Tolerance,
                $"Tanh gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}, diff={diff}");
        }
    }

    [Fact]
    public void BatchNormalizationLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int features = 4;
        const int batchSize = 3;
        var shape = new[] { batchSize, features };

        var layer = new BatchNormalizationLayer<float>(features);

        var input = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 2);
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            outputGradient[i] = (float)(random.NextDouble());
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
            Assert.True(diff < Tolerance,
                $"BatchNorm gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}, diff={diff}");
        }
    }

    [Fact]
    public void DropoutLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const double dropoutRate = 0.3;
        var shape = new[] { 5, 10 };

        var layer = new DropoutLayer<float>(shape, (float)dropoutRate);
        layer.IsTraining = true; // Enable dropout

        var input = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble());
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            outputGradient[i] = (float)(random.NextDouble());
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients (using same dropout mask)
        layer.UseAutodiff = true;
        layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        // Note: Dropout uses stochastic masking, so we check that both produce
        // the same pattern (zeros in same places) rather than exact values
        for (int i = 0; i < manualGradient.Length; i++)
        {
            // Both should be zero or both should be non-zero (same dropout mask)
            bool manualIsZero = Math.Abs(manualGradient[i]) < 1e-8;
            bool autodiffIsZero = Math.Abs(autodiffGradient[i]) < 1e-8;

            Assert.Equal(manualIsZero, autodiffIsZero);

            // If both are non-zero, they should match
            if (!manualIsZero && !autodiffIsZero)
            {
                var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
                Assert.True(diff < Tolerance,
                    $"Dropout gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}");
            }
        }
    }

    /// <summary>
    /// Helper method to verify numerical gradients using finite differences.
    /// This provides an additional check that both manual and autodiff are correct.
    /// </summary>
    private void VerifyGradientsWithFiniteDifferences<T>(
        LayerBase<T> layer,
        Tensor<T> input,
        Tensor<T> computedGradient,
        double epsilon = 1e-5)
    {
        // TODO: Implement numerical gradient checking
        // This would use (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
        // to compute numerical approximation of gradient and compare
    }
}
