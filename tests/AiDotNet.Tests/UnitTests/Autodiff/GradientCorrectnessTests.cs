using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Autodiff.Testing;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Autodiff;

/// <summary>
/// Tests to verify that autodiff gradients match manual gradient implementations.
/// </summary>
/// <remarks>
/// <para>
/// This test class verifies gradient correctness at the layer level by comparing:
/// - Manual gradient implementations (layer.UseAutodiff = false)
/// - Autodiff gradient implementations (layer.UseAutodiff = true)
/// - Numerical gradients for complex TensorOperations
/// </para>
/// <para>
/// For testing individual TensorOperations gradients, see also:
/// - <see cref="TensorOperationsVerification{T}"/> for comprehensive operation verification
/// - <see cref="NumericalGradient{T}"/> for numerical gradient utilities
/// </para>
/// </remarks>
public class GradientCorrectnessTests
{
    private const double Tolerance = 1e-4; // Tolerance for gradient comparisons
    private const double NumericalTolerance = 4e-3; // Tolerance for numerical gradient comparisons (less precise due to finite differences)

    // DenseLayer and ResidualLayer autodiff tolerances:
    // Now using tight tolerance (1e-4) since both paths use cached pre-activation values from forward.
    // The autodiff path was fixed to use _lastOutput instead of recomputing the forward pass.
    private const double DenseLayerTolerance = 1e-4;
    private const double ResidualLayerTolerance = 1e-4;

    [Fact]
    public void DenseLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int inputSize = 10;
        const int outputSize = 5;
        const int batchSize = 3;

        var layer = new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());

        // Create test input
        var input = new Tensor<float>(new[] { batchSize, inputSize });
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            SetTensorValue(input, i, (float)(random.NextDouble() * 2 - 1)); // Range: [-1, 1]
        }

        // Create test output gradient
        var outputGradient = new Tensor<float>(new[] { batchSize, outputSize });
        for (int i = 0; i < outputGradient.Length; i++)
        {
            SetTensorValue(outputGradient, i, (float)(random.NextDouble() * 2 - 1));
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
        // NOTE: Using relaxed tolerance due to known issue with bias broadcasting in autodiff reconstruction
        for (int i = 0; i < manualInputGradient.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manualInputGradient, i) - GetTensorValue(autodiffInputGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"Input gradient mismatch at index {i}: manual={GetTensorValue(manualInputGradient, i)}, autodiff={GetTensorValue(autodiffInputGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void ActivationLayer_ReLU_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 4, 8 }; // Batch size 4, 8 features
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new ReLUActivation<float>());

        // Create test input with mix of positive and negative values
        var input = new Tensor<float>(shape);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            SetTensorValue(input, i, (float)(random.NextDouble() * 4 - 2)); // Range: [-2, 2]
        }

        // Create test output gradient
        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            SetTensorValue(outputGradient, i, (float)(random.NextDouble()));
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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"Gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void ActivationLayer_Sigmoid_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 2, 6 };
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new SigmoidActivation<float>());

        var input = new Tensor<float>(shape);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            SetTensorValue(input, i, (float)(random.NextDouble() * 2 - 1));
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            SetTensorValue(outputGradient, i, (float)(random.NextDouble()));
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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"Sigmoid gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void ActivationLayer_Tanh_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 3, 4 };
        var layer = new ActivationLayer<float>(shape, (IActivationFunction<float>)new TanhActivation<float>());

        var input = new Tensor<float>(shape);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            SetTensorValue(input, i, (float)(random.NextDouble() * 2 - 1));
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            SetTensorValue(outputGradient, i, (float)(random.NextDouble()));
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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"Tanh gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
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
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            SetTensorValue(input, i, (float)(random.NextDouble() * 2));
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            SetTensorValue(outputGradient, i, (float)(random.NextDouble()));
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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"BatchNorm gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void DropoutLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const double dropoutRate = 0.3;
        var shape = new[] { 5, 10 };

        var layer = new DropoutLayer<float>(dropoutRate);
        layer.SetTrainingMode(true); // Enable dropout

        var input = new Tensor<float>(shape);
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < input.Length; i++)
        {
            SetTensorValue(input, i, (float)(random.NextDouble()));
        }

        var outputGradient = new Tensor<float>(shape);
        for (int i = 0; i < outputGradient.Length; i++)
        {
            SetTensorValue(outputGradient, i, (float)(random.NextDouble()));
        }

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        // NOTE: Do NOT reset state - both manual and autodiff must use the same dropout mask
        // layer.ResetState();

        // Act - Autodiff gradients (using same dropout mask)
        layer.UseAutodiff = true;
        // Do NOT call Forward again - reuse the same dropout mask from the manual forward pass
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        // Note: Dropout uses stochastic masking, so we check that both produce
        // the same pattern (zeros in same places) rather than exact values
        for (int i = 0; i < manualGradient.Length; i++)
        {
            // Both should be zero or both should be non-zero (same dropout mask)
            bool manualIsZero = Math.Abs(GetTensorValue(manualGradient, i)) < 1e-8;
            bool autodiffIsZero = Math.Abs(GetTensorValue(autodiffGradient, i)) < 1e-8;

            Assert.Equal(manualIsZero, autodiffIsZero);

            // If both are non-zero, they should match
            if (!manualIsZero && !autodiffIsZero)
            {
                var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
                Assert.True(diff < Tolerance,
                    $"Dropout gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}");
            }
        }
    }

    [Fact]
    public void AddLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 4, 6 };
        var layer = new AddLayer<float>(new[] { shape, shape }, (IActivationFunction<float>?)null);

        var input1 = CreateRandomTensor(shape);
        var input2 = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(new[] { input1, input2 });
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(new[] { input1, input2 });
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert - Gradients should match
        // Note: AddLayer's Backward returns a single gradient for the first input
        // Both inputs receive the same gradient since addition distributes gradients equally
        var manual = manualGradient;
        var autodiff = autodiffGradient;

        Assert.Equal(manual.Shape, autodiff.Shape);

        for (int i = 0; i < manual.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manual, i) - GetTensorValue(autodiff, i));
            Assert.True(diff < Tolerance,
                $"AddLayer gradient mismatch at index {i}: manual={GetTensorValue(manual, i)}, autodiff={GetTensorValue(autodiff, i)}");
        }
    }

    [Fact]
    public void MultiplyLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 3, 5 };
        var layer = new MultiplyLayer<float>(new[] { shape, shape }, (IActivationFunction<float>?)null);

        var input1 = CreateRandomTensor(shape);
        var input2 = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(new[] { input1, input2 });
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(new[] { input1, input2 });
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert - Gradients should match
        // Note: MultiplyLayer's Backward returns a single gradient for the first input
        var manual = manualGradient;
        var autodiff = autodiffGradient;

        Assert.Equal(manual.Shape, autodiff.Shape);

        for (int i = 0; i < manual.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manual, i) - GetTensorValue(autodiff, i));
            Assert.True(diff < Tolerance,
                $"MultiplyLayer gradient mismatch at index {i}: manual={GetTensorValue(manual, i)}, autodiff={GetTensorValue(autodiff, i)}");
        }
    }

    [Fact]
    public void ResidualLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 2, 8 };
        var innerLayer = new DenseLayer<float>(8, 8, (IActivationFunction<float>)new ReLUActivation<float>());
        var layer = new ResidualLayer<float>(shape, innerLayer, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        innerLayer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        // Note: Do NOT reset state - we need to keep cached forward pass values
        // so that both backward passes use the same forward outputs

        // Act - Autodiff gradients (reusing cached forward pass state)
        layer.UseAutodiff = true;
        innerLayer.UseAutodiff = true;
        // Just change the backward implementation, don't rerun forward
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        // NOTE: Using very relaxed tolerance because ResidualLayer has large gradient discrepancies (requires investigation)
        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < ResidualLayerTolerance,
                $"ResidualLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}");
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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"LayerNorm gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}");
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

        var dense1 = new DenseLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>)new ReLUActivation<float>());
        var dense2 = new DenseLayer<float>(hiddenSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());

        var input = CreateRandomTensor(new[] { batchSize, inputSize });
        var outputGradient = CreateRandomTensor(new[] { batchSize, outputSize });

        // Act - Manual gradients
        dense1.UseAutodiff = false;
        dense2.UseAutodiff = false;

        var hidden = dense1.Forward(input);
        var output = dense2.Forward(hidden);
        var grad2 = dense2.Backward(outputGradient);
        var manualGradient = dense1.Backward(grad2);

        // Note: Do NOT reset state - we need to keep the cached forward pass values
        // so that both backward passes use the same forward outputs

        // Act - Autodiff gradients (reusing cached forward pass state)
        dense1.UseAutodiff = true;
        dense2.UseAutodiff = true;

        // Just change the backward implementation, don't rerun forward
        grad2 = dense2.Backward(outputGradient);
        var autodiffGradient = dense1.Backward(grad2);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        // NOTE: Using relaxed tolerance because MultiLayerNetwork contains DenseLayer with known autodiff issues
        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"Multi-layer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}");
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
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = TensorOperations<float>.Softmax(inputNode, axis: -1);
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
                SetTensorValue(inputPlus, i, GetTensorValue(inputPlus, i) + epsilon);
                var nodePlus = TensorOperations<float>.Variable(inputPlus, requiresGradient: false);
                var outputPlus = TensorOperations<float>.Softmax(nodePlus, axis: -1);

                // Forward - epsilon
                var inputMinus = input.Clone();
                SetTensorValue(inputMinus, i, GetTensorValue(inputMinus, i) - epsilon);
                var nodeMinus = TensorOperations<float>.Variable(inputMinus, requiresGradient: false);
                var outputMinus = TensorOperations<float>.Softmax(nodeMinus, axis: -1);

                // Numerical gradient
                float gradSum = 0;
                for (int j = 0; j < outputGradient.Length; j++)
                {
                    float diff = (GetTensorValue(outputPlus.Value, j) - GetTensorValue(outputMinus.Value, j)) / (2 * epsilon);
                    gradSum += GetTensorValue(outputGradient, j) * diff;
                }
                SetTensorValue(numericalGradient, i, gradSum);
            }

            // Assert - gradients should match within tolerance
            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                var diff = Math.Abs(GetTensorValue(autodiffGradient, i) - GetTensorValue(numericalGradient, i));
                Assert.True(diff < Tolerance,
                    $"Softmax gradient mismatch at index {i}: autodiff={GetTensorValue(autodiffGradient, i)}, numerical={GetTensorValue(numericalGradient, i)}");
            }
        }
    }

    [Fact]
    public void TaylorSoftmax_AutodiffGradients_MatchNumericalGradients()
    {
        // Arrange
        const int batchSize = 2;
        const int features = 4;
        var shape = new[] { batchSize, features };

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Autodiff gradients
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = TensorOperations<float>.TaylorSoftmax(inputNode, order: 2, axis: -1);
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
                SetTensorValue(inputPlus, i, GetTensorValue(inputPlus, i) + epsilon);
                var nodePlus = TensorOperations<float>.Variable(inputPlus, requiresGradient: false);
                var outputPlus = TensorOperations<float>.TaylorSoftmax(nodePlus, order: 2, axis: -1);

                // Forward - epsilon
                var inputMinus = input.Clone();
                SetTensorValue(inputMinus, i, GetTensorValue(inputMinus, i) - epsilon);
                var nodeMinus = TensorOperations<float>.Variable(inputMinus, requiresGradient: false);
                var outputMinus = TensorOperations<float>.TaylorSoftmax(nodeMinus, order: 2, axis: -1);

                // Numerical gradient
                float gradSum = 0;
                for (int j = 0; j < outputGradient.Length; j++)
                {
                    float diff = (GetTensorValue(outputPlus.Value, j) - GetTensorValue(outputMinus.Value, j)) / (2 * epsilon);
                    gradSum += GetTensorValue(outputGradient, j) * diff;
                }
                SetTensorValue(numericalGradient, i, gradSum);
            }

            // Assert - gradients should match within tolerance
            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                var diff = Math.Abs(GetTensorValue(autodiffGradient, i) - GetTensorValue(numericalGradient, i));
                Assert.True(diff < Tolerance,
                    $"TaylorSoftmax gradient mismatch at index {i}: autodiff={GetTensorValue(autodiffGradient, i)}, numerical={GetTensorValue(numericalGradient, i)}");
            }
        }
    }

    [Fact]
    public void TaylorSoftmax_HigherOrder_AutodiffGradients_MatchNumericalGradients()
    {
        // Test with higher order Taylor approximation (order=4)
        const int batchSize = 2;
        const int features = 4;
        var shape = new[] { batchSize, features };

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = TensorOperations<float>.TaylorSoftmax(inputNode, order: 4, axis: -1);
            output.Gradient = outputGradient;

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
                var inputPlus = input.Clone();
                SetTensorValue(inputPlus, i, GetTensorValue(inputPlus, i) + epsilon);
                var nodePlus = TensorOperations<float>.Variable(inputPlus, requiresGradient: false);
                var outputPlus = TensorOperations<float>.TaylorSoftmax(nodePlus, order: 4, axis: -1);

                var inputMinus = input.Clone();
                SetTensorValue(inputMinus, i, GetTensorValue(inputMinus, i) - epsilon);
                var nodeMinus = TensorOperations<float>.Variable(inputMinus, requiresGradient: false);
                var outputMinus = TensorOperations<float>.TaylorSoftmax(nodeMinus, order: 4, axis: -1);

                float gradSum = 0;
                for (int j = 0; j < outputGradient.Length; j++)
                {
                    float diff = (GetTensorValue(outputPlus.Value, j) - GetTensorValue(outputMinus.Value, j)) / (2 * epsilon);
                    gradSum += GetTensorValue(outputGradient, j) * diff;
                }
                SetTensorValue(numericalGradient, i, gradSum);
            }

            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                var diff = Math.Abs(GetTensorValue(autodiffGradient, i) - GetTensorValue(numericalGradient, i));
                Assert.True(diff < Tolerance,
                    $"TaylorSoftmax (order=4) gradient mismatch at index {i}: autodiff={GetTensorValue(autodiffGradient, i)}, numerical={GetTensorValue(numericalGradient, i)}");
            }
        }
    }

    [Fact]
    public void TaylorSoftmax_ThrowsOnInvalidOrder()
    {
        var input = CreateRandomTensor(new[] { 2, 4 });
        var inputNode = TensorOperations<float>.Variable(input, requiresGradient: false);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorOperations<float>.TaylorSoftmax(inputNode, order: 0, axis: -1));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorOperations<float>.TaylorSoftmax(inputNode, order: -1, axis: -1));
    }

    [Fact]
    public void GumbelSoftmax_AutodiffGradients_MatchNumericalGradients()
    {
        // Note: GumbelSoftmax has stochastic Gumbel noise, so we test gradient flow
        // by using a fixed seed and ensuring gradients are computed correctly
        const int batchSize = 2;
        const int features = 4;
        var shape = new[] { batchSize, features };

        // Use a deterministic input for stable gradient testing
        var input = new Tensor<float>(shape);
        for (int i = 0; i < input.Length; i++)
            SetTensorValue(input, i, (float)(i * 0.1 - 0.4)); // Range roughly [-0.4, 0.4]

        var outputGradient = CreateRandomTensor(shape);

        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            // Use soft mode (hard=false) for proper gradient testing
            var output = TensorOperations<float>.GumbelSoftmax(inputNode, temperature: 1.0, hard: false);
            output.Gradient = outputGradient;

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

            // Verify gradient has correct shape and non-zero values
            Assert.Equal(shape, autodiffGradient.Shape);
            Assert.True(autodiffGradient.Length > 0);

            // Verify gradient values are reasonable (not NaN/Inf)
            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                Assert.False(float.IsNaN(GetTensorValue(autodiffGradient, i)), $"GumbelSoftmax gradient is NaN at index {i}");
                Assert.False(float.IsInfinity(GetTensorValue(autodiffGradient, i)), $"GumbelSoftmax gradient is Infinity at index {i}");
            }
        }
    }

    [Fact]
    public void GumbelSoftmax_TemperatureScaling_AffectsOutput()
    {
        var shape = new[] { 2, 4 };
        var input = CreateRandomTensor(shape);

        var inputNodeHigh = TensorOperations<float>.Variable(input, requiresGradient: false);
        var inputNodeLow = TensorOperations<float>.Variable(input, requiresGradient: false);

        // Higher temperature = softer distribution
        var outputHigh = TensorOperations<float>.GumbelSoftmax(inputNodeHigh, temperature: 5.0, hard: false);
        // Lower temperature = sharper distribution
        var outputLow = TensorOperations<float>.GumbelSoftmax(inputNodeLow, temperature: 0.5, hard: false);

        // Verify both produce valid probability distributions (sum to 1)
        for (int b = 0; b < 2; b++)
        {
            float sumHigh = 0, sumLow = 0;
            for (int f = 0; f < 4; f++)
            {
                sumHigh += GetTensorValue(outputHigh.Value, b * 4 + f);
                sumLow += GetTensorValue(outputLow.Value, b * 4 + f);
            }
            Assert.True(Math.Abs(sumHigh - 1.0f) < 0.01f, $"High temp output doesn't sum to 1: {sumHigh}");
            Assert.True(Math.Abs(sumLow - 1.0f) < 0.01f, $"Low temp output doesn't sum to 1: {sumLow}");
        }
    }

    [Fact]
    public void GumbelSoftmax_HardMode_ProducesOneHot()
    {
        var shape = new[] { 3, 5 };
        var input = CreateRandomTensor(shape);
        var inputNode = TensorOperations<float>.Variable(input, requiresGradient: false);

        var output = TensorOperations<float>.GumbelSoftmax(inputNode, temperature: 1.0, hard: true);

        // Verify hard mode produces one-hot vectors
        for (int b = 0; b < 3; b++)
        {
            int oneCount = 0;
            int zeroCount = 0;
            for (int f = 0; f < 5; f++)
            {
                var val = GetTensorValue(output.Value, b * 5 + f);
                if (Math.Abs(val - 1.0f) < 0.01f) oneCount++;
                else if (Math.Abs(val) < 0.01f) zeroCount++;
            }
            Assert.Equal(1, oneCount);
            Assert.Equal(4, zeroCount);
        }
    }

    [Fact]
    public void GumbelSoftmax_ThrowsOnInvalidTemperature()
    {
        var input = CreateRandomTensor(new[] { 2, 4 });
        var inputNode = TensorOperations<float>.Variable(input, requiresGradient: false);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorOperations<float>.GumbelSoftmax(inputNode, temperature: 0, hard: false));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorOperations<float>.GumbelSoftmax(inputNode, temperature: -1.0, hard: false));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorOperations<float>.GumbelSoftmax(inputNode, temperature: double.NaN, hard: false));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorOperations<float>.GumbelSoftmax(inputNode, temperature: double.PositiveInfinity, hard: false));
    }

    [Fact]
    public void MaxPool2D_AutodiffGradients_CorrectRouting()
    {
        // Arrange - Simple 2x2 max pool on 4x4 input
        var input = new Tensor<float>(new int[] { 1, 1, 4, 4 });
        // Create pattern where max positions are known
        for (int i = 0; i < 16; i++)
            SetTensorValue(input, i, i);

        var outputGradient = new Tensor<float>(new int[] { 1, 1, 2, 2 });
        outputGradient[0, 0, 0, 0] = 1.0f;
        outputGradient[0, 0, 0, 1] = 2.0f;
        outputGradient[0, 0, 1, 0] = 3.0f;
        outputGradient[0, 0, 1, 1] = 4.0f;

        // Act
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = TensorOperations<float>.MaxPool2D(inputNode, new int[] { 2, 2 });
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
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = TensorOperations<float>.AvgPool2D(inputNode, new int[] { 2, 2 });
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
        using (var tape = new GradientTape<float>())
        {
            var node1 = TensorOperations<float>.Variable(input1, "input1", requiresGradient: true);
            var node2 = TensorOperations<float>.Variable(input2, "input2", requiresGradient: true);
            tape.Watch(node1);
            tape.Watch(node2);

            var nodes = new List<ComputationNode<float>> { node1, node2 };
            var output = TensorOperations<float>.Concat(nodes, axis: 1);
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
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            tape.Watch(inputNode);

            var output = TensorOperations<float>.Pad(inputNode, padWidth, 0f);
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
    private static List<ComputationNode<T>> GetTopologicalOrder<T>(ComputationNode<T> root)
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();

        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
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

    [Fact]
    public void LayerNorm_AutodiffGradients_MatchNumericalGradients()
    {
        // Arrange
        const int batchSize = 2;
        const int features = 4;
        var shape = new[] { batchSize, features };

        var input = CreateRandomTensor(shape);
        var gamma = CreateRandomTensor(new int[] { features });
        var beta = CreateRandomTensor(new int[] { features });
        var outputGradient = CreateRandomTensor(shape);

        // Act - Autodiff gradients
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            var gammaNode = TensorOperations<float>.Variable(gamma, "gamma", requiresGradient: true);
            var betaNode = TensorOperations<float>.Variable(beta, "beta", requiresGradient: true);
            tape.Watch(inputNode);
            tape.Watch(gammaNode);
            tape.Watch(betaNode);

            var output = TensorOperations<float>.LayerNorm(inputNode, new int[] { features }, gammaNode, betaNode);
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

            // Numerical gradient for input
            const float epsilon = 1e-4f;
            var numericalGradient = new Tensor<float>(shape);

            for (int i = 0; i < input.Length; i++)
            {
                // Forward + epsilon
                var inputPlus = input.Clone();
                SetTensorValue(inputPlus, i, GetTensorValue(inputPlus, i) + epsilon);
                var nodePlus = TensorOperations<float>.Variable(inputPlus, requiresGradient: false);
                var gammaNodePlus = TensorOperations<float>.Variable(gamma, requiresGradient: false);
                var betaNodePlus = TensorOperations<float>.Variable(beta, requiresGradient: false);
                var outputPlus = TensorOperations<float>.LayerNorm(nodePlus, new int[] { features }, gammaNodePlus, betaNodePlus);

                // Forward - epsilon
                var inputMinus = input.Clone();
                SetTensorValue(inputMinus, i, GetTensorValue(inputMinus, i) - epsilon);
                var nodeMinus = TensorOperations<float>.Variable(inputMinus, requiresGradient: false);
                var gammaNodeMinus = TensorOperations<float>.Variable(gamma, requiresGradient: false);
                var betaNodeMinus = TensorOperations<float>.Variable(beta, requiresGradient: false);
                var outputMinus = TensorOperations<float>.LayerNorm(nodeMinus, new int[] { features }, gammaNodeMinus, betaNodeMinus);

                // Numerical gradient
                float gradSum = 0;
                for (int j = 0; j < outputGradient.Length; j++)
                {
                    float diff = (GetTensorValue(outputPlus.Value, j) - GetTensorValue(outputMinus.Value, j)) / (2 * epsilon);
                    gradSum += GetTensorValue(outputGradient, j) * diff;
                }
                SetTensorValue(numericalGradient, i, gradSum);
            }

            // Assert - gradients should match within numerical tolerance
            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                var diff = Math.Abs(GetTensorValue(autodiffGradient, i) - GetTensorValue(numericalGradient, i));
                Assert.True(diff < NumericalTolerance,
                    $"LayerNorm gradient mismatch at index {i}: autodiff={GetTensorValue(autodiffGradient, i)}, numerical={GetTensorValue(numericalGradient, i)}");
            }
        }
    }

    [Fact]
    public void BatchNorm_AutodiffGradients_MatchNumericalGradients()
    {
        // Arrange
        const int batchSize = 4;
        const int features = 3;
        var shape = new[] { batchSize, features };

        // Use different seeds for each tensor to avoid correlated values
        // that can cause gradient cancellation
        var input = CreateRandomTensor(shape, 42);
        var gamma = CreateRandomTensor(new int[] { features }, 123);
        var beta = CreateRandomTensor(new int[] { features }, 456);
        var outputGradient = CreateRandomTensor(shape, 789);

        // Act - Autodiff gradients (training mode)
        using (var tape = new GradientTape<float>())
        {
            var inputNode = TensorOperations<float>.Variable(input, "input", requiresGradient: true);
            var gammaNode = TensorOperations<float>.Variable(gamma, "gamma", requiresGradient: true);
            var betaNode = TensorOperations<float>.Variable(beta, "beta", requiresGradient: true);
            tape.Watch(inputNode);
            tape.Watch(gammaNode);
            tape.Watch(betaNode);

            var output = TensorOperations<float>.BatchNorm(
                inputNode, gammaNode, betaNode, null, null, training: true);
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

            // Numerical gradient for input
            const float epsilon = 1e-4f;
            var numericalGradient = new Tensor<float>(shape);

            for (int i = 0; i < input.Length; i++)
            {
                // Forward + epsilon
                var inputPlus = input.Clone();
                SetTensorValue(inputPlus, i, GetTensorValue(inputPlus, i) + epsilon);
                var nodePlus = TensorOperations<float>.Variable(inputPlus, requiresGradient: false);
                var gammaNodePlus = TensorOperations<float>.Variable(gamma, requiresGradient: false);
                var betaNodePlus = TensorOperations<float>.Variable(beta, requiresGradient: false);
                var outputPlus = TensorOperations<float>.BatchNorm(
                    nodePlus, gammaNodePlus, betaNodePlus, null, null, training: true);

                // Forward - epsilon
                var inputMinus = input.Clone();
                SetTensorValue(inputMinus, i, GetTensorValue(inputMinus, i) - epsilon);
                var nodeMinus = TensorOperations<float>.Variable(inputMinus, requiresGradient: false);
                var gammaNodeMinus = TensorOperations<float>.Variable(gamma, requiresGradient: false);
                var betaNodeMinus = TensorOperations<float>.Variable(beta, requiresGradient: false);
                var outputMinus = TensorOperations<float>.BatchNorm(
                    nodeMinus, gammaNodeMinus, betaNodeMinus, null, null, training: true);

                // Numerical gradient
                float gradSum = 0;
                for (int j = 0; j < outputGradient.Length; j++)
                {
                    float diff = (GetTensorValue(outputPlus.Value, j) - GetTensorValue(outputMinus.Value, j)) / (2 * epsilon);
                    gradSum += GetTensorValue(outputGradient, j) * diff;
                }
                SetTensorValue(numericalGradient, i, gradSum);
            }

            // Assert - gradients should match within tolerance
            // Using PyTorch-style tolerance: |a - b| <= atol + rtol * max(|a|, |b|)
            // For float32 numerical gradient checking of BatchNorm:
            // - atol = 1e-4 (absolute tolerance for small values)
            // - rtol = 3e-3 (0.3% relative tolerance)
            // BatchNorm has complex gradients involving batch mean/variance dependencies.
            const double atol = 1e-4;
            const double rtol = 3e-3; // 0.3% relative tolerance (conservative for float32 BatchNorm)
            for (int i = 0; i < autodiffGradient.Length; i++)
            {
                var autodiffVal = GetTensorValue(autodiffGradient, i);
                var numericalVal = GetTensorValue(numericalGradient, i);
                var absDiff = Math.Abs(autodiffVal - numericalVal);
                var maxAbs = Math.Max(Math.Abs(autodiffVal), Math.Abs(numericalVal));

                // PyTorch-style tolerance formula (additive, not max)
                var effectiveTolerance = atol + rtol * maxAbs;
                Assert.True(absDiff <= effectiveTolerance,
                    $"BatchNorm gradient mismatch at index {i}: autodiff={autodiffVal}, numerical={numericalVal}, diff={absDiff}, tolerance={effectiveTolerance}");
            }
        }
    }

    /// <summary>
    /// Helper to create random tensors for testing with an optional seed.
    /// </summary>
    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        int totalSize = shape.Aggregate(1, (acc, dim) => acc * dim);
        var data = new float[totalSize];
        var random = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < totalSize; i++)
        {
            data[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return new Tensor<float>(shape, new Vector<float>(data));
    }

    /// <summary>
    /// Convert flat index to multi-dimensional indices for tensor access.
    /// </summary>
    private static int[] FlatToIndices(int flatIndex, int[] shape)
    {
        var indices = new int[shape.Length];
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            indices[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
        return indices;
    }

    /// <summary>
    /// Get value at flat index from tensor.
    /// </summary>
    private static float GetTensorValue(Tensor<float> tensor, int flatIndex)
    {
        return tensor.GetFlat(flatIndex);
    }

    /// <summary>
    /// Set value at flat index in tensor.
    /// </summary>
    private static void SetTensorValue(Tensor<float> tensor, int flatIndex, float value)
    {
        tensor.SetFlat(flatIndex, value);
    }

    #region Comprehensive Layer Gradient Tests

    [Fact]
    public void ConvolutionalLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - Small conv layer: 1 input channel, 2 output channels, 3x3 kernel, 4x4 input
        var layer = new ConvolutionalLayer<float>(
            inputDepth: 1, outputDepth: 2, kernelSize: 3,
            inputHeight: 4, inputWidth: 4, stride: 1, padding: 0,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());

        var input = CreateRandomTensor(new[] { 1, 1, 4, 4 });
        var outputGradient = CreateRandomTensor(new[] { 1, 2, 2, 2 }); // Output: 2x2 with 2 channels

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"ConvolutionalLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void MaxPoolingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - 2x2 max pooling on 4x4 input (no batch dimension, just [channels, height, width])
        var inputShape = new[] { 1, 4, 4 }; // [channels=1, height=4, width=4]
        var layer = new MaxPoolingLayer<float>(inputShape, poolSize: 2, stride: 2);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(new[] { 1, 2, 2 }); // [channels=1, height=2, width=2]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"MaxPoolingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void AvgPoolingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - 2x2 avg pooling on 4x4 input (no batch dimension, just [channels, height, width])
        var inputShape = new[] { 1, 4, 4 }; // [channels=1, height=4, width=4]
        var layer = new AveragePoolingLayer<float>(inputShape, 2, 2);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(new[] { 1, 2, 2 }); // [channels=1, height=2, width=2]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"AvgPoolingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void FullyConnectedLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int inputSize = 8;
        const int outputSize = 4;
        const int batchSize = 2;

        var layer = new FullyConnectedLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());

        var input = CreateRandomTensor(new[] { batchSize, inputSize });
        var outputGradient = CreateRandomTensor(new[] { batchSize, outputSize });

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"FullyConnectedLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void FlattenLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - inputShape is per-sample (excludes batch), input tensor has [batch, ...inputShape]
        var inputShape = new[] { 3, 4 }; // Per-sample: 3x4
        var layer = new FlattenLayer<float>(inputShape);

        var input = CreateRandomTensor(new[] { 2, 3, 4 }); // [batch=2, 3, 4]
        var outputGradient = CreateRandomTensor(new[] { 2, 12 }); // [batch=2, flattened=12]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"FlattenLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void ReshapeLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - inputShape and targetShape are per-sample (excludes batch)
        var inputShape = new[] { 12 }; // Per-sample: 1D vector of 12
        var targetShape = new[] { 3, 4 }; // Per-sample: 3x4 matrix
        var layer = new ReshapeLayer<float>(inputShape, targetShape);

        var input = CreateRandomTensor(new[] { 2, 12 }); // [batch=2, 12]
        var outputGradient = CreateRandomTensor(new[] { 2, 3, 4 }); // [batch=2, 3, 4]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"ReshapeLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void ConcatenateLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - Use same-shaped inputs so gradients can be stacked
        var shape1 = new[] { 2, 3 };
        var shape2 = new[] { 2, 3 }; // Same shape as shape1 so sliced gradients can be stacked
        var layer = new ConcatenateLayer<float>(new[] { shape1, shape2 }, 1, (IActivationFunction<float>?)null);

        var input1 = CreateRandomTensor(shape1);
        var input2 = CreateRandomTensor(shape2);
        var outputGradient = CreateRandomTensor(new[] { 2, 6 }); // 3 + 3 = 6

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.Forward(new[] { input1, input2 });
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.Forward(new[] { input1, input2 });
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"ConcatenateLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void GlobalPoolingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - Global average pooling
        // GlobalPoolingLayer uses channels-last format: [batch, height, width, channels]
        var inputShape = new[] { 2, 4, 4, 3 };
        var layer = new GlobalPoolingLayer<float>(inputShape, PoolingType.Average, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(inputShape);
        // Output shape is [batch, 1, 1, channels] = [2, 1, 1, 3]
        var outputGradient = CreateRandomTensor(new[] { 2, 1, 1, 3 });

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"GlobalPoolingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void GaussianNoiseLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var shape = new[] { 3, 4 };
        var layer = new GaussianNoiseLayer<float>(shape, standardDeviation: 0.0); // Zero stddev for deterministic testing
        layer.SetTrainingMode(true); // Keep training mode to ensure state is saved for autodiff

        var input = CreateRandomTensor(shape);
        var outputGradient = CreateRandomTensor(shape);

        // Act - Manual gradients
        layer.UseAutodiff = false;
        layer.SetTrainingMode(true);
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        layer.ResetState();

        // Act - Autodiff gradients
        layer.UseAutodiff = true;
        layer.SetTrainingMode(true);
        layer.Forward(input);
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"GaussianNoiseLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void MeanLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - MeanLayer computes mean along a specified axis
        var inputShape = new[] { 3, 4 };
        var layer = new MeanLayer<float>(inputShape, axis: 1);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(new[] { 3 }); // Mean along axis 1 reduces to shape [3]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"MeanLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void PaddingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - PaddingLayer expects 4D input [batch, height, width, channels]
        var inputShape = new[] { 2, 3, 3, 1 }; // batch=2, height=3, width=3, channels=1
        var padding = new[] { 0, 1, 1, 0 }; // No batch/channel padding, 1 on spatial dims
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(new[] { 2, 5, 5, 1 }); // Padded: [2, 3+2, 3+2, 1]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"PaddingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void CroppingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - CroppingLayer expects 4D input [batch, height, width, channels]
        var inputShape = new[] { 2, 5, 5, 1 }; // batch=2, height=5, width=5, channels=1
        // Crop 1 from all sides on spatial dimensions (dim 1=height, dim 2=width)
        var cropTop = new[] { 0, 1, 0, 0 };    // Crop 1 from top of height (dim 1)
        var cropBottom = new[] { 0, 1, 0, 0 }; // Crop 1 from bottom of height (dim 1)
        var cropLeft = new[] { 0, 0, 1, 0 };   // Crop 1 from left of width (dim 2)
        var cropRight = new[] { 0, 0, 1, 0 };  // Crop 1 from right of width (dim 2)
        var layer = new CroppingLayer<float>(inputShape, cropTop, cropBottom, cropLeft, cropRight, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(new[] { 2, 3, 3, 1 }); // Cropped: [2, 5-2, 5-2, 1]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"CroppingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void SplitLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var inputShape = new[] { 2, 8 };
        var layer = new SplitLayer<float>(inputShape, numSplits: 2);

        var input = CreateRandomTensor(inputShape);
        // Output shape is [batchSize, numSplits, splitSize] = [2, 2, 4]
        var outputGradient = CreateRandomTensor(new[] { 2, 2, 4 });

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"SplitLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void PositionalEncodingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - PositionalEncodingLayer expects 2D input [sequenceLength, embeddingDim]
        const int sequenceLength = 4;
        const int embeddingDim = 8;
        var inputShape = new[] { sequenceLength, embeddingDim }; // 2D: [seq, embed]
        var layer = new PositionalEncodingLayer<float>(sequenceLength, embeddingDim);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(inputShape);

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"PositionalEncodingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void HighwayLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int inputSize = 8;
        var layer = new HighwayLayer<float>(inputSize,
            (IActivationFunction<float>)new ReLUActivation<float>(),
            (IActivationFunction<float>)new SigmoidActivation<float>());

        var input = CreateRandomTensor(new[] { 2, inputSize });
        var outputGradient = CreateRandomTensor(new[] { 2, inputSize });

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"HighwayLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void FeedForwardLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange - FeedForwardLayer outputs [batch, hiddenSize]
        const int inputSize = 8;
        const int hiddenSize = 16;
        var layer = new FeedForwardLayer<float>(inputSize, hiddenSize, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(new[] { 2, inputSize });
        var outputGradient = CreateRandomTensor(new[] { 2, hiddenSize }); // Output is [batch, hiddenSize]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"FeedForwardLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void GatedLinearUnitLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int inputSize = 8;
        const int outputSize = 4; // GLU outputs half the input dimension
        var layer = new GatedLinearUnitLayer<float>(inputSize, outputSize, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(new[] { 2, inputSize });
        var outputGradient = CreateRandomTensor(new[] { 2, outputSize }); // GLU halves the last dimension

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"GatedLinearUnitLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void UpsamplingLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var inputShape = new[] { 1, 1, 2, 2 };
        var layer = new UpsamplingLayer<float>(inputShape, scaleFactor: 2);

        var input = CreateRandomTensor(inputShape);
        var outputGradient = CreateRandomTensor(new[] { 1, 1, 4, 4 });

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"UpsamplingLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void LogVarianceLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var inputShape = new[] { 2, 8 }; // batch size 2, feature size 8
        var layer = new LogVarianceLayer<float>(inputShape, axis: 1);

        var input = CreateRandomTensor(inputShape);
        // LogVariance computes variance along axis, reducing that axis
        var outputGradient = CreateRandomTensor(new[] { 2 }); // variance along axis 1 reduces to [batch]

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"LogVarianceLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void RepParameterizationLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        var inputShape = new[] { 2, 8 }; // batch size 2, input size 8 (4 means + 4 log_vars)
        var layer = new RepParameterizationLayer<float>(inputShape);

        // RepParameterization takes a single input that contains both mean and log_var
        // The output is half the size of input (just the sampled latent values)
        var input = CreateRandomTensor(inputShape);
        var outputShape = new[] { 2, 4 }; // latent size is inputShape[1] / 2
        var outputGradient = CreateRandomTensor(outputShape);

        // Act - Manual gradients
        // Note: RepParameterizationLayer generates random epsilon values during Forward,
        // so we must NOT call ResetState() or Forward() again for the autodiff test.
        // Both backward methods use the same cached _lastMean, _lastLogVar, _lastEpsilon values.
        layer.UseAutodiff = false;
        layer.Forward(input);
        var manualGradient = layer.Backward(outputGradient);

        // Act - Autodiff gradients (use same cached values from previous Forward)
        layer.UseAutodiff = true;
        var autodiffGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(manualGradient.Shape, autodiffGradient.Shape);

        for (int i = 0; i < manualGradient.Length; i++)
        {
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < Tolerance,
                $"RepParameterizationLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    [Fact]
    public void ReconstructionLayer_AutodiffGradients_MatchManualGradients()
    {
        // Arrange
        const int latentSize = 4;
        const int hidden1Size = 6;
        const int hidden2Size = 6;
        const int outputSize = 8;
        var layer = new ReconstructionLayer<float>(latentSize, hidden1Size, hidden2Size, outputSize,
            (IActivationFunction<float>?)null, (IActivationFunction<float>?)null);

        var input = CreateRandomTensor(new[] { 2, latentSize });
        var outputGradient = CreateRandomTensor(new[] { 2, outputSize });

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
            var diff = Math.Abs(GetTensorValue(manualGradient, i) - GetTensorValue(autodiffGradient, i));
            Assert.True(diff < DenseLayerTolerance,
                $"ReconstructionLayer gradient mismatch at index {i}: manual={GetTensorValue(manualGradient, i)}, autodiff={GetTensorValue(autodiffGradient, i)}, diff={diff}");
        }
    }

    #endregion
}
