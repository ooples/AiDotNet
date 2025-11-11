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

    [Fact]
    public void AddLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 4, 6 };
        var layer = new AddLayer<float>(shape);

        var input1 = CreateRandomTensor(shape);
        var input2 = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(new[] { input1, input2 });
        var manualGradients = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(new[] { input1, input2 });
        var autodiffGradients = layer.Backward(outputGradient);

        // Assert - Both input gradients should match
        Assert.Equal(2, manualGradients.Length);
        Assert.Equal(2, autodiffGradients.Length);

        for (int inputIdx = 0; inputIdx < 2; inputIdx++)
        {
            var manual = manualGradients[inputIdx];
            var autodiff = autodiffGradients[inputIdx];

            Assert.Equal(manual.Shape, autodiff.Shape);

            for (int i = 0; i < manual.Length; i++)
            {
                var diff = Math.Abs(manual[i] - autodiff[i]);
                Assert.True(diff < Tolerance,
                    $"AddLayer gradient mismatch at input {inputIdx}, index {i}: manual={manual[i]}, autodiff={autodiff[i]}");
            }
        }
    }

    [Fact]
    public void MultiplyLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 3, 5 };
        var layer = new MultiplyLayer<float>(shape);

        var input1 = CreateRandomTensor(shape);
        var input2 = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(new[] { input1, input2 });
        var manualGradients = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(new[] { input1, input2 });
        var autodiffGradients = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(2, manualGradients.Length);
        Assert.Equal(2, autodiffGradients.Length);

        for (int inputIdx = 0; inputIdx < 2; inputIdx++)
        {
            var manual = manualGradients[inputIdx];
            var autodiff = autodiffGradients[inputIdx];

            Assert.Equal(manual.Shape, autodiff.Shape);

            for (int i = 0; i < manual.Length; i++)
            {
                var diff = Math.Abs(manual[i] - autodiff[i]);
                Assert.True(diff < Tolerance,
                    $"MultiplyLayer gradient mismatch at input {inputIdx}, index {i}: manual={manual[i]}, autodiff={autodiff[i]}");
            }
        }
    }

    [Fact]
    public void ResidualLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 2, 8 };
        var innerLayer = new DenseLayer<float>(8, 8, new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(shape, innerLayer);

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        innerLayer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();
        innerLayer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        innerLayer.UseAutodiff = true;
        layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
            Assert.True(diff < Tolerance,
                $"ResidualLayer gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}");
        }
    }

    [Fact]
    public void LayerNormalizationLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int features = 6;
        const int batchSize = 4;
        var shape = new[] { batchSize, features };

        var layer = new LayerNormalizationLayer<float>(features);

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

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
                $"LayerNorm gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}");
        }
    }

    [Fact]
    public void MultiLayerNetwork_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - Create a small network: Dense -> ReLU -> Dense
        const int inputSize = 8;
        const int hiddenSize = 6;
        const int outputSize = 4;
        const int batchSize = 3;

        var dense1 = new DenseLayer<float>(inputSize, hiddenSize, new ReLUActivation<float>());
        var dense2 = new DenseLayer<float>(hiddenSize, outputSize, new ReLUActivation<float>());

        var input = CreateRandomTensor(new[] { batchSize, inputSize });
        var outputGradient = CreateRandomTensor(new[] { batchSize, outputSize });

        // Act - Manual gradients
        dense1.UseAutodiff = false;
        dense2.UseAutodiff = false;

        var hidden = dense1.Forward(input);
        var output = dense2.Forward(hidden);
        var grad2 = dense2.Backward(outputGradient);
        var manualGradient = dense1.Backward(grad2);

        dense1.ResetState();
        dense2.ResetState();

        // Act - Autodiff gradients
        dense1.UseAutodiff = true;
        dense2.UseAutodiff = true;

        hidden = dense1.Forward(input);
        output = dense2.Forward(hidden);
        grad2 = dense2.Backward(outputGradient);
        var autodiffGradient = dense1.Backward(grad2);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(manualGradient[i] - autodiffGradient[i]);
            Assert.True(diff < Tolerance,
                $"Multi-layer gradient mismatch at index {i}: manual={manualGradient[i]}, autodiff={autodiffGradient[i]}");
        }
    }

    /// <summary>
    /// Helper to create random tensors for testing.
    /// </summary>
    private static Tensor<float> CreateRandomTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }
}
