namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

/// <summary>
/// Integration tests for pooling layer implementations testing any-rank tensor support,
/// forward/backward passes, training vs inference mode, serialization, and cloning.
/// </summary>
public class PoolingLayersIntegrationTests
{
    #region MaxPoolingLayer Tests

    [Fact]
    public void MaxPoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - 3D input [channels, height, width]
        int[] inputShape = [3, 8, 8];
        int poolSize = 2;
        int stride = 2;
        var layer = new MaxPoolingLayer<float>(inputShape, poolSize, stride);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [channels, height/2, width/2]
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(3, output.Shape[0]); // channels preserved
        Assert.Equal(4, output.Shape[1]); // height reduced by pool size
        Assert.Equal(4, output.Shape[2]); // width reduced by pool size
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MaxPoolingLayer_ForwardPass_4DInput_ProducesValidOutput()
    {
        // Arrange - 4D input [batch, channels, height, width]
        int[] inputShape = [3, 8, 8]; // layer expects CHW
        int poolSize = 2;
        int stride = 2;
        var layer = new MaxPoolingLayer<float>(inputShape, poolSize, stride);
        var input = CreateRandomTensor<float>([2, 3, 8, 8]); // NCHW input

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [batch, channels, height/2, width/2]
        Assert.Equal(4, output.Shape.Length);
        Assert.Equal(2, output.Shape[0]); // batch preserved
        Assert.Equal(3, output.Shape[1]); // channels preserved
        Assert.Equal(4, output.Shape[2]); // height reduced
        Assert.Equal(4, output.Shape[3]); // width reduced
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MaxPoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [2, 6, 6];
        int poolSize = 2;
        int stride = 2;
        var layer = new MaxPoolingLayer<float>(inputShape, poolSize, stride);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void MaxPoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [3, 8, 8];
        var original = new MaxPoolingLayer<float>(inputShape, 2, 2);
        var input = CreateRandomTensor<float>(inputShape);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (MaxPoolingLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(original.PoolSize, cloned.PoolSize);
        Assert.Equal(original.Stride, cloned.Stride);
        AssertTensorsEqual(originalOutput, clonedOutput);
    }

    #endregion

    #region AveragePoolingLayer Tests

    [Fact]
    public void AveragePoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - 3D input [channels, height, width]
        int[] inputShape = [4, 10, 10];
        int poolSize = 2;
        int stride = 2;
        var layer = new AveragePoolingLayer<float>(inputShape, poolSize, stride);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [channels, height/2, width/2]
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(4, output.Shape[0]); // channels preserved
        Assert.Equal(5, output.Shape[1]); // height reduced
        Assert.Equal(5, output.Shape[2]); // width reduced
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AveragePoolingLayer_ForwardPass_4DInput_ProducesValidOutput()
    {
        // Arrange - 4D input [batch, channels, height, width]
        int[] inputShape = [4, 8, 8];
        int poolSize = 2;
        int stride = 2;
        var layer = new AveragePoolingLayer<float>(inputShape, poolSize, stride);
        var input = CreateRandomTensor<float>([3, 4, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [batch, channels, height/2, width/2]
        Assert.Equal(4, output.Shape.Length);
        Assert.Equal(3, output.Shape[0]); // batch preserved
        Assert.Equal(4, output.Shape[1]); // channels preserved
        Assert.Equal(4, output.Shape[2]); // height reduced
        Assert.Equal(4, output.Shape[3]); // width reduced
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AveragePoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [2, 6, 6];
        int poolSize = 2;
        int stride = 2;
        var layer = new AveragePoolingLayer<float>(inputShape, poolSize, stride);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void AveragePoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [4, 8, 8];
        var original = new AveragePoolingLayer<float>(inputShape, 2, 2);
        var input = CreateRandomTensor<float>(inputShape);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (AveragePoolingLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        AssertTensorsEqual(originalOutput, clonedOutput);
    }

    #endregion

    #region GlobalPoolingLayer Tests

    [Fact]
    public void GlobalPoolingLayer_MaxPooling_ProducesValidOutput()
    {
        // Arrange - 3D input [channels, height, width]
        int[] inputShape = [8, 7, 7];
        var layer = new GlobalPoolingLayer<float>(inputShape, PoolingType.Max);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert - output should be [channels] (1D) or [channels, 1, 1]
        Assert.False(ContainsNaN(output));
        // Global pooling reduces spatial dimensions
        Assert.True(output.Length <= input.Length);
    }

    [Fact]
    public void GlobalPoolingLayer_AveragePooling_ProducesValidOutput()
    {
        // Arrange
        int[] inputShape = [8, 7, 7];
        var layer = new GlobalPoolingLayer<float>(inputShape, PoolingType.Average);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
        Assert.True(output.Length <= input.Length);
    }

    [Fact]
    public void GlobalPoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [4, 6, 6];
        var layer = new GlobalPoolingLayer<float>(inputShape, PoolingType.Max);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void GlobalPoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [8, 7, 7];
        var original = new GlobalPoolingLayer<float>(inputShape, PoolingType.Max);
        var input = CreateRandomTensor<float>(inputShape);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (GlobalPoolingLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        AssertTensorsEqual(originalOutput, clonedOutput);
    }

    #endregion

    #region AdaptiveAveragePoolingLayer Tests

    [Fact]
    public void AdaptiveAveragePoolingLayer_ForwardPass_ProducesValidOutput()
    {
        // Arrange - Adaptive pooling to fixed output size
        int inputChannels = 4;
        int inputHeight = 16;
        int inputWidth = 16;
        int outputHeight = 4;
        int outputWidth = 4;
        var layer = new AdaptiveAveragePoolingLayer<float>(inputChannels, inputHeight, inputWidth, outputHeight, outputWidth);
        var input = CreateRandomTensor<float>([inputChannels, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert - output should have adaptive dimensions
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AdaptiveAveragePoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputChannels = 4;
        int inputHeight = 12;
        int inputWidth = 12;
        int outputHeight = 3;
        int outputWidth = 3;
        var layer = new AdaptiveAveragePoolingLayer<float>(inputChannels, inputHeight, inputWidth, outputHeight, outputWidth);
        var input = CreateRandomTensor<float>([inputChannels, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void AdaptiveAveragePoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new AdaptiveAveragePoolingLayer<float>(4, 16, 16, 4, 4);
        var input = CreateRandomTensor<float>([4, 16, 16]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (AdaptiveAveragePoolingLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        AssertTensorsEqual(originalOutput, clonedOutput);
    }

    #endregion

    #region PoolingLayer (Generic) Tests

    [Fact]
    public void PoolingLayer_MaxType_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 8;
        int inputWidth = 8;
        int poolSize = 2;
        int stride = 2;
        var layer = new PoolingLayer<float>(inputDepth, inputHeight, inputWidth, poolSize, stride, PoolingType.Max);
        var input = CreateRandomTensor<float>([inputDepth, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PoolingLayer_AverageType_ForwardPass_ProducesValidOutput()
    {
        // Arrange
        int inputDepth = 3;
        int inputHeight = 8;
        int inputWidth = 8;
        int poolSize = 2;
        int stride = 2;
        var layer = new PoolingLayer<float>(inputDepth, inputHeight, inputWidth, poolSize, stride, PoolingType.Average);
        var input = CreateRandomTensor<float>([inputDepth, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PoolingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int inputDepth = 2;
        int inputHeight = 6;
        int inputWidth = 6;
        int poolSize = 2;
        int stride = 2;
        var layer = new PoolingLayer<float>(inputDepth, inputHeight, inputWidth, poolSize, stride, PoolingType.Max);
        var input = CreateRandomTensor<float>([inputDepth, inputHeight, inputWidth]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor<float>(output.Shape);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void PoolingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new PoolingLayer<float>(3, 8, 8, 2, 2, PoolingType.Max);
        var input = CreateRandomTensor<float>([3, 8, 8]);
        var originalOutput = original.Forward(input);

        // Act
        var cloned = (PoolingLayer<float>)original.Clone();
        var clonedOutput = cloned.Forward(input);

        // Assert
        Assert.NotSame(original, cloned);
        Assert.Equal(original.PoolSize, cloned.PoolSize);
        Assert.Equal(original.Stride, cloned.Stride);
        AssertTensorsEqual(originalOutput, clonedOutput);
    }

    #endregion

    #region Training vs Inference Mode Tests

    [Fact]
    public void MaxPoolingLayer_TrainingVsInference_ProducesSameOutput()
    {
        // Arrange
        int[] inputShape = [3, 8, 8];
        var layer = new MaxPoolingLayer<float>(inputShape, 2, 2);
        var input = CreateRandomTensor<float>(inputShape);

        // Act - Training mode
        layer.SetTrainingMode(true);
        var trainingOutput = layer.Forward(input);

        // Act - Inference mode
        layer.SetTrainingMode(false);
        var inferenceOutput = layer.Forward(input);

        // Assert - Pooling should produce same output in both modes
        AssertTensorsEqual(trainingOutput, inferenceOutput);
    }

    [Fact]
    public void AveragePoolingLayer_TrainingVsInference_ProducesSameOutput()
    {
        // Arrange
        int[] inputShape = [4, 8, 8];
        var layer = new AveragePoolingLayer<float>(inputShape, 2, 2);
        var input = CreateRandomTensor<float>(inputShape);

        // Act - Training mode
        layer.SetTrainingMode(true);
        var trainingOutput = layer.Forward(input);

        // Act - Inference mode
        layer.SetTrainingMode(false);
        var inferenceOutput = layer.Forward(input);

        // Assert
        AssertTensorsEqual(trainingOutput, inferenceOutput);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void MaxPoolingLayer_SingleChannel_Works()
    {
        // Arrange - single channel input
        int[] inputShape = [1, 8, 8];
        var layer = new MaxPoolingLayer<float>(inputShape, 2, 2);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(1, output.Shape[0]);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AveragePoolingLayer_PoolSize3_Works()
    {
        // Arrange - 3x3 pooling
        int[] inputShape = [2, 9, 9];
        var layer = new AveragePoolingLayer<float>(inputShape, 3, 3);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Shape[0]); // channels
        Assert.Equal(3, output.Shape[1]); // 9/3
        Assert.Equal(3, output.Shape[2]); // 9/3
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MaxPoolingLayer_NonSquareInput_Works()
    {
        // Arrange - non-square input
        int[] inputShape = [3, 8, 12];
        var layer = new MaxPoolingLayer<float>(inputShape, 2, 2);
        var input = CreateRandomTensor<float>(inputShape);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(3, output.Shape[0]); // channels
        Assert.Equal(4, output.Shape[1]); // 8/2
        Assert.Equal(6, output.Shape[2]); // 12/2
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region Helper Methods

    private static Tensor<T> CreateRandomTensor<T>(int[] shape) where T : struct, IComparable<T>
    {
        var tensor = new Tensor<T>(shape);
        var random = new Random(42);
        var span = tensor.AsSpan();

        for (int i = 0; i < span.Length; i++)
        {
            double value = random.NextDouble() * 2 - 1; // [-1, 1]
            span[i] = (T)Convert.ChangeType(value, typeof(T));
        }

        return tensor;
    }

    private static bool ContainsNaN<T>(Tensor<T> tensor) where T : struct, IComparable<T>
    {
        foreach (var value in tensor.Data.ToArray())
        {
            if (value is float f && float.IsNaN(f)) return true;
            if (value is double d && double.IsNaN(d)) return true;
        }
        return false;
    }

    private static void AssertTensorsEqual<T>(Tensor<T> expected, Tensor<T> actual, float tolerance = 1e-5f)
        where T : struct, IComparable<T>
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < expected.Length; i++)
        {
            var exp = Convert.ToDouble(expected[i]);
            var act = Convert.ToDouble(actual[i]);
            Assert.True(Math.Abs(exp - act) < tolerance, $"Tensors differ at index {i}: expected {exp}, got {act}");
        }
    }

    #endregion
}
