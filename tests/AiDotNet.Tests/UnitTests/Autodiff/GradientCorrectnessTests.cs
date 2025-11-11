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

    [Fact]
    public void Softmax_AutodiffGradients_MatchNumericalGradients()
    {
        // Arrange
        const int batchSize = 2;
        const int features = 4;
        var shape = new[] { batchSize, features };

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Autodiff gradients
        using (var tape = new Autodiff.GradientTape<float>())
        {
            var inputNode = Autodiff.TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = Autodiff.TensorOperations<float>.Softmax(inputNode, axis: -1);
            output.Gradient = outputGradient;

            // Backward pass
            var topoOrder = GetTopologicalOrder(output);
            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                var node = topoOrder[i];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            var autodiffGradient = inputNode.Gradient!;

            // Numerical gradient
            const float epsilon = 1e-4f;
            var numericalGradient = new Tensor<float>(shape);

            for (int i = 0; i < input.Length; i++)
            {
                // Forward + epsilon
                var inputPlus = input.Clone();
                inputPlus[i] += epsilon;
                var nodePlus = Autodiff.TensorOperations<float>.Variable(inputPlus, requiresGradient: false);
                var outputPlus = Autodiff.TensorOperations<float>.Softmax(nodePlus, axis: -1);

                // Forward - epsilon
                var inputMinus = input.Clone();
                inputMinus[i] -= epsilon;
                var nodeMinus = Autodiff.TensorOperations<float>.Variable(inputMinus, requiresGradient: false);
                var outputMinus = Autodiff.TensorOperations<float>.Softmax(nodeMinus, axis: -1);

                // Numerical gradient
                float gradSum = 0;
                for (int j = 0; j < outputGradient.Length; j++)
                {
                    float diff = (outputPlus.Value[j] - outputMinus.Value[j]) / (2 * epsilon);
                    gradSum += outputGradient[j] * diff;
                }
                numericalGradient[i] = gradSum;
            }

            // Assert - gradients should match within tolerance
            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                var diff = Math.Abs(autodiffGradient[i] - numericalGradient[i]);
                Assert.True(diff < Tolerance,
                    $"Softmax gradient mismatch at index {i}: autodiff={autodiffGradient[i]}, numerical={numericalGradient[i]}");
            }
        }
    }

    [Fact]
    public void MaxPool2D_AutodiffGradients_CorrectRouting()
    {
        // Arrange - Simple 2x2 max pool on 4x4 input
        var input = new Tensor<float>(new int[] { 1, 1, 4, 4 });
        // Create pattern where max positions are known
        for (int i = 0; i < 16; i++)
            input[i] = i;

        var outputGradient = new Tensor<float>(new int[] { 1, 1, 2, 2 });
        outputGradient[0, 0, 0, 0] = 1.0f;
        outputGradient[0, 0, 0, 1] = 2.0f;
        outputGradient[0, 0, 1, 0] = 3.0f;
        outputGradient[0, 0, 1, 1] = 4.0f;

        // Act
        using (var tape = new Autodiff.GradientTape<float>())
        {
            var inputNode = Autodiff.TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = Autodiff.TensorOperations<float>.MaxPool2D(inputNode, new int[] { 2, 2 });
            output.Gradient = outputGradient;

            // Backward pass
            var topoOrder = GetTopologicalOrder(output);
            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                var node = topoOrder[i];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            var gradient = inputNode.Gradient!;

            // Assert - gradients should only flow to max positions
            // Max positions: [5, 7, 13, 15] (bottom-right of each 2x2 window)
            Assert.Equal(0f, gradient[0]); // Not max
            Assert.Equal(1f, gradient[5]); // Max of top-left window
            Assert.Equal(2f, gradient[7]); // Max of top-right window
            Assert.Equal(3f, gradient[13]); // Max of bottom-left window
            Assert.Equal(4f, gradient[15]); // Max of bottom-right window
        }
    }

    [Fact]
    public void AvgPool2D_AutodiffGradients_EqualDistribution()
    {
        // Arrange - 2x2 avg pool on 4x4 input
        var input = CreateRandomTensor(new int[] { 1, 1, 4, 4 });
        var outputGradient = new Tensor<float>(new int[] { 1, 1, 2, 2 });
        outputGradient[0, 0, 0, 0] = 4.0f;  // Will be distributed as 1.0 to each of 4 elements
        outputGradient[0, 0, 0, 1] = 8.0f;  // Will be distributed as 2.0 to each
        outputGradient[0, 0, 1, 0] = 12.0f; // Will be distributed as 3.0 to each
        outputGradient[0, 0, 1, 1] = 16.0f; // Will be distributed as 4.0 to each

        // Act
        using (var tape = new Autodiff.GradientTape<float>())
        {
            var inputNode = Autodiff.TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = Autodiff.TensorOperations<float>.AvgPool2D(inputNode, new int[] { 2, 2 });
            output.Gradient = outputGradient;

            // Backward pass
            var topoOrder = GetTopologicalOrder(output);
            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                var node = topoOrder[i];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            var gradient = inputNode.Gradient!;

            // Assert - each element in window gets equal share (gradient / 4)
            // Top-left window
            Assert.Equal(1.0f, gradient[0, 0, 0, 0]);
            Assert.Equal(1.0f, gradient[0, 0, 0, 1]);
            Assert.Equal(1.0f, gradient[0, 0, 1, 0]);
            Assert.Equal(1.0f, gradient[0, 0, 1, 1]);

            // Top-right window
            Assert.Equal(2.0f, gradient[0, 0, 0, 2]);
            Assert.Equal(2.0f, gradient[0, 0, 0, 3]);
        }
    }

    [Fact]
    public void Concat_AutodiffGradients_CorrectSplitting()
    {
        // Arrange
        var input1 = CreateRandomTensor(new int[] { 2, 3 });
        var input2 = CreateRandomTensor(new int[] { 2, 4 });
        var outputGradient = CreateRandomTensor(new int[] { 2, 7 }); // 3 + 4 = 7

        // Act
        using (var tape = new Autodiff.GradientTape<float>())
        {
            var node1 = Autodiff.TensorOperations<float>.Variable(input1, "input1", requiresGradient: true);
            var node2 = Autodiff.TensorOperations<float>.Variable(input2, "input2", requiresGradient: true);
            tape.Watch(node1);
            tape.Watch(node2);

            var nodes = new List<Autodiff.ComputationNode<float>> { node1, node2 };
            var output = Autodiff.TensorOperations<float>.Concat(nodes, axis: 1);
            output.Gradient = outputGradient;

            // Backward pass
            var topoOrder = GetTopologicalOrder(output);
            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                var node = topoOrder[i];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            var grad1 = node1.Gradient!;
            var grad2 = node2.Gradient!;

            // Assert - gradients should be split correctly
            Assert.Equal(new int[] { 2, 3 }, grad1.Shape);
            Assert.Equal(new int[] { 2, 4 }, grad2.Shape);

            // Check values match corresponding slices
            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(outputGradient[r, c], grad1[r, c]);
                }
                for (int c = 0; c < 4; c++)
                {
                    Assert.Equal(outputGradient[r, 3 + c], grad2[r, c]);
                }
            }
        }
    }

    [Fact]
    public void Pad_AutodiffGradients_CorrectCropping()
    {
        // Arrange
        var input = CreateRandomTensor(new int[] { 3, 4 });
        var padWidth = new int[,] { { 1, 1 }, { 2, 2 } }; // Pad 1 row top/bottom, 2 cols left/right
        var outputGradient = CreateRandomTensor(new int[] { 5, 8 }); // 3+2 rows, 4+4 cols

        // Act
        using (var tape = new Autodiff.GradientTape<float>())
        {
            var inputNode = Autodiff.TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = Autodiff.TensorOperations<float>.Pad(inputNode, padWidth, 0f);
            output.Gradient = outputGradient;

            // Backward pass
            var topoOrder = GetTopologicalOrder(output);
            for (int i = topoOrder.Count - 1; i >= 0; i--)
            {
                var node = topoOrder[i];
                if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
                {
                    node.BackwardFunction(node.Gradient);
                }
            }

            var gradient = inputNode.Gradient!;

            // Assert - gradient should match center region of output gradient
            Assert.Equal(new int[] { 3, 4 }, gradient.Shape);

            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    Assert.Equal(outputGradient[r + 1, c + 2], gradient[r, c]);
                }
            }
        }
    }

    /// <summary>
    /// Gets topological order for gradient computation.
    /// </summary>
    private static List<Autodiff.ComputationNode<T>> GetTopologicalOrder<T>(Autodiff.ComputationNode<T> root)
    {
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var result = new List<Autodiff.ComputationNode<T>>();

        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));
                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                        stack.Push((parent, false));
                }
            }
        }

        return result;
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
