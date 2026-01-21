using AiDotNet.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Autodiff;

/// <summary>
/// Integration tests for GradientCheckpointing to verify memory-efficient training functionality.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify that gradient checkpointing correctly:
/// 1. Preserves forward pass computation results
/// 2. Recomputes activations during backward pass
/// 3. Produces correct gradients identical to non-checkpointed versions
/// 4. Handles edge cases like empty layers, single layers, etc.
/// </para>
/// <para><b>For Beginners:</b> Gradient checkpointing is a memory-saving technique that trades
/// computation time for memory by not storing all intermediate activations. These tests ensure
/// the technique produces mathematically correct results.
/// </para>
/// </remarks>
public class GradientCheckpointingIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Helper Methods

    /// <summary>
    /// Creates a double tensor with the specified shape and values.
    /// </summary>
    private static Tensor<double> CreateTensor(int[] shape, params double[] values)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < values.Length && i < tensor.Length; i++)
        {
            tensor[i] = values[i];
        }
        return tensor;
    }

    /// <summary>
    /// Creates a float tensor with the specified shape and values.
    /// </summary>
    private static Tensor<float> CreateTensorFloat(int[] shape, params float[] values)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < values.Length && i < tensor.Length; i++)
        {
            tensor[i] = values[i];
        }
        return tensor;
    }

    /// <summary>
    /// Creates a ones tensor.
    /// </summary>
    private static ComputationNode<double> CreateOnes(int size, string name)
    {
        var tensor = new Tensor<double>(new[] { size });
        for (int i = 0; i < size; i++) tensor[i] = 1.0;
        return TensorOperations<double>.Constant(tensor, name);
    }

    #endregion

    #region Basic Checkpoint Tests

    [Fact]
    public void Checkpoint_SimpleFunction_PreservesForwardValue()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2, 3 }, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            "input",
            requiresGradient: true);

        // Act - checkpoint a simple operation
        var output = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Sum(input),
            new[] { input });

        // Assert - value should be preserved
        Assert.NotNull(output.Value);
        // Sum of 1+2+3+4+5+6 = 21
        Assert.Equal(21.0, output.Value[0], Tolerance);
    }

    [Fact]
    public void Checkpoint_SimpleFunction_BackwardProducesCorrectGradients()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2, 2 }, 1.0, 2.0, 3.0, 4.0),
            "input",
            requiresGradient: true);

        // Act - checkpoint sum operation
        var output = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Sum(input),
            new[] { input });

        output.Backward();

        // Assert - gradient of sum with respect to each element should be 1
        Assert.NotNull(input.Gradient);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(1.0, input.Gradient.Data.Span[i], Tolerance);
        }
    }

    [Fact]
    public void Checkpoint_ChainedOperations_GradientsMatchNonCheckpointed()
    {
        // Arrange
        // Non-checkpointed version
        var input1 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 3 }, 1.0, 2.0, 3.0),
            "input1",
            requiresGradient: true);
        var squared1 = TensorOperations<double>.Square(input1);
        var sum1 = TensorOperations<double>.Sum(squared1);
        sum1.Backward();
        var expectedGradient = input1.Gradient;

        // Checkpointed version
        var input2 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 3 }, 1.0, 2.0, 3.0),
            "input2",
            requiresGradient: true);
        var checkpointedOutput = GradientCheckpointing<double>.Checkpoint(
            () =>
            {
                var squared = TensorOperations<double>.Square(input2);
                return TensorOperations<double>.Sum(squared);
            },
            new[] { input2 });
        checkpointedOutput.Backward();

        // Assert - gradients should match
        Assert.NotNull(expectedGradient);
        Assert.NotNull(input2.Gradient);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(expectedGradient[i], input2.Gradient[i], Tolerance);
        }
    }

    [Fact]
    public void Checkpoint_MultipleOperations_PreservesIntermediateComputation()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // Act - checkpoint a chain: square -> add -> sum
        var output = GradientCheckpointing<double>.Checkpoint(
            () =>
            {
                var squared = TensorOperations<double>.Square(input);
                var plusOne = TensorOperations<double>.Add(squared, CreateOnes(2, "ones"));
                return TensorOperations<double>.Sum(plusOne);
            },
            new[] { input });

        // Assert - (2^2 + 1) + (3^2 + 1) = 5 + 10 = 15
        Assert.Equal(15.0, output.Value![0], Tolerance);

        // Backward
        output.Backward();

        // Gradient: d/dx[(x^2 + 1)] = 2x
        // For x=2: grad=4, for x=3: grad=6
        Assert.NotNull(input.Gradient);
        Assert.Equal(4.0, input.Gradient[0], Tolerance);
        Assert.Equal(6.0, input.Gradient[1], Tolerance);
    }

    #endregion

    #region Multi-Output Checkpoint Tests

    [Fact]
    public void CheckpointMultiOutput_TwoOutputs_BothValuesPreserved()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 3 }, 1.0, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // Act - checkpoint function with two outputs
        var outputs = GradientCheckpointing<double>.CheckpointMultiOutput(
            () =>
            {
                var sum = TensorOperations<double>.Sum(input);
                var mean = TensorOperations<double>.Mean(input);
                return new List<ComputationNode<double>> { sum, mean };
            },
            new[] { input });

        // Assert
        Assert.Equal(2, outputs.Count);
        Assert.Equal(6.0, outputs[0].Value![0], Tolerance); // Sum: 1+2+3=6
        Assert.Equal(2.0, outputs[1].Value![0], Tolerance); // Mean: 6/3=2
    }

    [Fact]
    public void CheckpointMultiOutput_BackwardOnOutput_ProducesGradients()
    {
        // Arrange - use separate inputs for each output to avoid shared gradient accumulation
        var input1 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 1.0, 2.0),
            "input1",
            requiresGradient: true);
        var input2 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 3.0, 4.0),
            "input2",
            requiresGradient: true);

        // Act - create multi-output checkpoint with separate input paths
        var outputs = GradientCheckpointing<double>.CheckpointMultiOutput(
            () =>
            {
                var sum1 = TensorOperations<double>.Sum(input1);  // Output 0: depends only on input1
                var sum2 = TensorOperations<double>.Sum(input2);  // Output 1: depends only on input2
                return new List<ComputationNode<double>> { sum1, sum2 };
            },
            new[] { input1, input2 });

        // Assert outputs have correct values
        Assert.Equal(2, outputs.Count);
        Assert.Equal(3.0, outputs[0].Value![0], Tolerance); // Sum of [1,2] = 3
        Assert.Equal(7.0, outputs[1].Value![0], Tolerance); // Sum of [3,4] = 7

        // Backward on first output
        outputs[0].Backward();

        // Assert - gradients should be computed
        Assert.NotNull(input1.Gradient);
        Assert.True(input1.Gradient.Length > 0, "Input1 should have gradients after backward");
    }

    #endregion

    #region Sequential Checkpoint Tests

    [Fact]
    public void SequentialCheckpoint_EmptyLayers_ReturnsInput()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 1.0, 2.0),
            "input",
            requiresGradient: true);

        var emptyLayers = new List<Func<ComputationNode<double>, ComputationNode<double>>>();

        // Act
        var output = GradientCheckpointing<double>.SequentialCheckpoint(emptyLayers, input, segmentSize: 2);

        // Assert - should return input unchanged
        Assert.Same(input, output);
    }

    [Fact]
    public void SequentialCheckpoint_SingleLayer_ProducesCorrectOutput()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 3 }, 1.0, 2.0, 3.0),
            "input",
            requiresGradient: true);

        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.Square(x)
        };

        // Act
        var output = GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: 2);

        // Assert - [1, 4, 9]
        Assert.NotNull(output.Value);
        Assert.Equal(1.0, output.Value[0], Tolerance);
        Assert.Equal(4.0, output.Value[1], Tolerance);
        Assert.Equal(9.0, output.Value[2], Tolerance);
    }

    [Fact]
    public void SequentialCheckpoint_MultipleLayers_SegmentsCorrectly()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // 4 layers: square -> add 1 -> square -> add 1
        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.Square(x),
            x => TensorOperations<double>.Add(x, CreateOnes(2, "ones1")),
            x => TensorOperations<double>.Square(x),
            x => TensorOperations<double>.Add(x, CreateOnes(2, "ones2"))
        };

        // Act - segment size 2 means 2 segments: [square, add1] and [square, add1]
        var output = GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: 2);

        // Assert
        // Input: [2, 3]
        // After square: [4, 9]
        // After add 1: [5, 10]
        // After square: [25, 100]
        // After add 1: [26, 101]
        Assert.NotNull(output.Value);
        Assert.Equal(26.0, output.Value[0], Tolerance);
        Assert.Equal(101.0, output.Value[1], Tolerance);
    }

    [Fact]
    public void SequentialCheckpoint_GradientsMatchNonCheckpointed()
    {
        // Arrange
        // Non-checkpointed version
        var input1 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input1",
            requiresGradient: true);

        var intermediate1 = TensorOperations<double>.Square(input1);
        intermediate1 = TensorOperations<double>.Add(intermediate1, CreateOnes(2, "ones1"));
        var result1 = TensorOperations<double>.Sum(intermediate1);
        result1.Backward();

        // Checkpointed version
        var input2 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input2",
            requiresGradient: true);

        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.Square(x),
            x => TensorOperations<double>.Add(x, CreateOnes(2, "ones2")),
            x => TensorOperations<double>.Sum(x)
        };

        var result2 = GradientCheckpointing<double>.SequentialCheckpoint(layers, input2, segmentSize: 2);
        result2.Backward();

        // Assert - gradients should match
        Assert.NotNull(input1.Gradient);
        Assert.NotNull(input2.Gradient);

        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(input1.Gradient[i], input2.Gradient[i], Tolerance);
        }
    }

    [Fact]
    public void SequentialCheckpoint_SegmentSizeLargerThanLayerCount_Works()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.Square(x)
        };

        // Act - segment size 10, but only 1 layer
        var output = GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: 10);

        // Assert - should still work correctly
        Assert.NotNull(output.Value);
        Assert.Equal(4.0, output.Value[0], Tolerance);
        Assert.Equal(9.0, output.Value[1], Tolerance);
    }

    [Fact]
    public void SequentialCheckpoint_ZeroOrNegativeSegmentSize_DefaultsToOne()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.Square(x),
            x => TensorOperations<double>.Add(x, CreateOnes(2, "ones"))
        };

        // Act - segment size 0, should default to 1
        var output = GradientCheckpointing<double>.SequentialCheckpoint(layers, input, segmentSize: 0);

        // Assert - [4+1, 9+1] = [5, 10]
        Assert.NotNull(output.Value);
        Assert.Equal(5.0, output.Value[0], Tolerance);
        Assert.Equal(10.0, output.Value[1], Tolerance);
    }

    #endregion

    #region Memory Estimation Tests

    [Fact]
    public void EstimateMemorySavings_BasicEstimation_ReturnsCorrectValues()
    {
        // Arrange
        int numLayers = 10;
        long activationSize = 1000;
        int segmentSize = 2;

        // Act
        var (withoutCheckpoint, withCheckpoint, savingsPercent) =
            GradientCheckpointing<double>.EstimateMemorySavings(numLayers, activationSize, segmentSize);

        // Assert
        // Without: 10 * 1000 = 10000
        Assert.Equal(10000, withoutCheckpoint);

        // With: (2 * 1000) + (5 * 1000) = 7000
        // numSegments = (10 + 2 - 1) / 2 = 5
        Assert.Equal(7000, withCheckpoint);

        // Savings: 1 - 7000/10000 = 0.3 = 30%
        Assert.Equal(30.0, savingsPercent, 1.0);
    }

    [Fact]
    public void EstimateMemorySavings_SingleSegment_MinimalSavings()
    {
        // Arrange - segment size equals layer count
        int numLayers = 4;
        long activationSize = 1000;
        int segmentSize = 4;

        // Act
        var (withoutCheckpoint, withCheckpoint, savingsPercent) =
            GradientCheckpointing<double>.EstimateMemorySavings(numLayers, activationSize, segmentSize);

        // Assert
        Assert.Equal(4000, withoutCheckpoint);
        // With: (4 * 1000) + (1 * 1000) = 5000
        Assert.Equal(5000, withCheckpoint);
        // Savings negative (more memory with checkpointing for this case)
        Assert.True(savingsPercent < 0);
    }

    [Fact]
    public void EstimateMemorySavings_LargeModel_SignificantSavings()
    {
        // Arrange - typical transformer-like model
        int numLayers = 24;
        long activationSize = 100_000_000; // 100MB per layer
        int segmentSize = 4;

        // Act
        var (withoutCheckpoint, withCheckpoint, savingsPercent) =
            GradientCheckpointing<double>.EstimateMemorySavings(numLayers, activationSize, segmentSize);

        // Assert
        // Without: 24 * 100M = 2.4GB
        Assert.Equal(2_400_000_000, withoutCheckpoint);

        // numSegments = (24 + 4 - 1) / 4 = 6
        // With: (4 * 100M) + (6 * 100M) = 1GB
        Assert.Equal(1_000_000_000, withCheckpoint);

        // Savings: ~58%
        Assert.True(savingsPercent > 50);
    }

    #endregion

    #region Extension Method Tests

    [Fact]
    public void WithCheckpoint_ExtensionMethod_WorksCorrectly()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // Act - use extension method
        var output = input.WithCheckpoint(x => TensorOperations<double>.Square(x));

        // Assert
        Assert.NotNull(output.Value);
        Assert.Equal(4.0, output.Value[0], Tolerance);
        Assert.Equal(9.0, output.Value[1], Tolerance);
    }

    [Fact]
    public void WithCheckpoint_ExtensionMethod_GradientsFlow()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // Act
        var squared = input.WithCheckpoint(x => TensorOperations<double>.Square(x));
        var summed = TensorOperations<double>.Sum(squared);
        summed.Backward();

        // Assert - d/dx[sum(x^2)] = 2x
        Assert.NotNull(input.Gradient);
        Assert.Equal(4.0, input.Gradient[0], Tolerance);
        Assert.Equal(6.0, input.Gradient[1], Tolerance);
    }

    [Fact]
    public void WithSequentialCheckpoint_ExtensionMethod_WorksCorrectly()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        var layers = new List<Func<ComputationNode<double>, ComputationNode<double>>>
        {
            x => TensorOperations<double>.Square(x),
            x => TensorOperations<double>.Add(x, CreateOnes(2, "ones"))
        };

        // Act - use extension method
        var output = input.WithSequentialCheckpoint(layers, segmentSize: 2);

        // Assert - [4+1, 9+1] = [5, 10]
        Assert.NotNull(output.Value);
        Assert.Equal(5.0, output.Value[0], Tolerance);
        Assert.Equal(10.0, output.Value[1], Tolerance);
    }

    #endregion

    #region Complex Scenarios

    [Fact]
    public void Checkpoint_NestedOperations_GradientsCorrect()
    {
        // Arrange - nested checkpoints
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // Act - checkpoint within checkpoint
        var innerOutput = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Square(input),
            new[] { input });

        var outerOutput = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Sum(innerOutput),
            new[] { innerOutput });

        outerOutput.Backward();

        // Assert - gradient of sum(x^2) = 2x
        Assert.NotNull(input.Gradient);
        Assert.Equal(4.0, input.Gradient[0], Tolerance);
        Assert.Equal(6.0, input.Gradient[1], Tolerance);
    }

    [Fact]
    public void Checkpoint_WithGradientTape_IntegratesCorrectly()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 3 }, 1.0, 2.0, 3.0),
            "input",
            requiresGradient: true);

        // Act - use checkpoint within gradient tape context
        using (var tape = new GradientTape<double>(persistent: true))
        {
            tape.Watch(input);

            var checkpointedOutput = GradientCheckpointing<double>.Checkpoint(
                () => TensorOperations<double>.Sum(TensorOperations<double>.Square(input)),
                new[] { input });

            checkpointedOutput.Backward();
        }

        // Assert
        Assert.NotNull(input.Gradient);
        // Gradient of sum(x^2) = 2x
        Assert.Equal(2.0, input.Gradient[0], Tolerance);
        Assert.Equal(4.0, input.Gradient[1], Tolerance);
        Assert.Equal(6.0, input.Gradient[2], Tolerance);
    }

    [Fact]
    public void Checkpoint_MultipleInputs_AllReceiveGradients()
    {
        // Arrange
        var input1 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 1.0, 2.0),
            "input1",
            requiresGradient: true);
        var input2 = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 3.0, 4.0),
            "input2",
            requiresGradient: true);

        // Act - checkpoint with multiple inputs
        var output = GradientCheckpointing<double>.Checkpoint(
            () =>
            {
                var product = TensorOperations<double>.ElementwiseMultiply(input1, input2);
                return TensorOperations<double>.Sum(product);
            },
            new[] { input1, input2 });

        output.Backward();

        // Assert
        // sum(input1 * input2) = 1*3 + 2*4 = 11
        Assert.Equal(11.0, output.Value![0], Tolerance);

        // d/d(input1) = input2 values
        Assert.NotNull(input1.Gradient);
        Assert.Equal(3.0, input1.Gradient[0], Tolerance);
        Assert.Equal(4.0, input1.Gradient[1], Tolerance);

        // d/d(input2) = input1 values
        Assert.NotNull(input2.Gradient);
        Assert.Equal(1.0, input2.Gradient[0], Tolerance);
        Assert.Equal(2.0, input2.Gradient[1], Tolerance);
    }

    [Fact]
    public void Checkpoint_FloatType_WorksCorrectly()
    {
        // Arrange - test with float type
        var input = TensorOperations<float>.Variable(
            CreateTensorFloat(new[] { 2 }, 2.0f, 3.0f),
            "input",
            requiresGradient: true);

        // Act
        var output = GradientCheckpointing<float>.Checkpoint(
            () => TensorOperations<float>.Sum(TensorOperations<float>.Square(input)),
            new[] { input });

        output.Backward();

        // Assert
        Assert.Equal(13.0f, output.Value![0], 1e-4f); // 4 + 9 = 13
        Assert.NotNull(input.Gradient);
        Assert.Equal(4.0f, input.Gradient[0], 1e-4f);
        Assert.Equal(6.0f, input.Gradient[1], 1e-4f);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Checkpoint_EmptyInputsList_HandlesGracefully()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 1.0, 2.0),
            "input",
            requiresGradient: true);

        // Act - empty inputs list (not null)
        var output = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Sum(input),
            Array.Empty<ComputationNode<double>>());

        // Assert - should still produce correct output
        Assert.Equal(3.0, output.Value![0], Tolerance);
    }

    [Fact]
    public void SequentialCheckpoint_NullLayersList_ReturnsInput()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 2 }, 1.0, 2.0),
            "input",
            requiresGradient: true);

        // Act - pass null (note: this should be handled gracefully or throw)
        var output = GradientCheckpointing<double>.SequentialCheckpoint(null!, input, segmentSize: 2);

        // Assert
        Assert.Same(input, output);
    }

    [Fact]
    public void Checkpoint_ScalarOutput_WorksCorrectly()
    {
        // Arrange
        var input = TensorOperations<double>.Variable(
            CreateTensor(new[] { 1 }, 5.0),
            "input",
            requiresGradient: true);

        // Act
        var output = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Square(input),
            new[] { input });

        output.Backward();

        // Assert
        Assert.Equal(25.0, output.Value![0], Tolerance);
        Assert.NotNull(input.Gradient);
        Assert.Equal(10.0, input.Gradient[0], Tolerance); // d/dx[x^2] = 2x = 10
    }

    [Fact]
    public void Checkpoint_LargeTensor_HandlesCorrectly()
    {
        // Arrange - larger tensor to test memory handling
        var size = 1000;
        var tensor = new Tensor<double>(new[] { size });
        for (int i = 0; i < size; i++) tensor[i] = i + 1;

        var input = TensorOperations<double>.Variable(tensor, "input", requiresGradient: true);

        // Act
        var output = GradientCheckpointing<double>.Checkpoint(
            () => TensorOperations<double>.Sum(input),
            new[] { input });

        output.Backward();

        // Assert
        // Sum of 1 to 1000 = n(n+1)/2 = 1000*1001/2 = 500500
        Assert.Equal(500500.0, output.Value![0], Tolerance);
        Assert.NotNull(input.Gradient);
        // Gradient of sum is 1 for all elements
        Assert.Equal(1.0, input.Gradient[0], Tolerance);
        Assert.Equal(1.0, input.Gradient[size - 1], Tolerance);
    }

    #endregion
}
