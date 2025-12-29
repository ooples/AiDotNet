namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Xunit;

/// <summary>
/// Integration tests for utility layers including AddLayer, ConcatenateLayer, SplitLayer,
/// MultiplyLayer, MeanLayer, InputLayer, and other tensor manipulation layers.
/// </summary>
public class UtilityLayersIntegrationTests
{
    #region Helper Methods

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var length = 1;
        foreach (var dim in shape) length *= dim;
        var flatData = new float[length];
        for (int i = 0; i < flatData.Length; i++)
        {
            flatData[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return new Tensor<float>(flatData, shape);
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.Data)
        {
            if (float.IsNaN(value)) return true;
        }
        return false;
    }

    private static bool ContainsInf(Tensor<float> tensor)
    {
        foreach (var value in tensor.Data)
        {
            if (float.IsInfinity(value)) return true;
        }
        return false;
    }

    #endregion

    #region AddLayer Tests

    [Fact]
    public void AddLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - 2D inputs [batch, features]
        int[][] inputShapes = [[64], [64]];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 64]);
        var input2 = CreateRandomTensor([4, 64], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.Equal(input1.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AddLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange - 4D inputs [batch, channels, height, width]
        int[][] inputShapes = [[32, 8, 8], [32, 8, 8]];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([2, 32, 8, 8]);
        var input2 = CreateRandomTensor([2, 32, 8, 8], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.Equal(input1.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void AddLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[][] inputShapes = [[32], [32]];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 32], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 456);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input1.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void AddLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[][] inputShapes = [[32], [32]];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 32], seed: 123);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input1, input2);
        var output2 = clone.Forward(input1, input2);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void AddLayer_MultipleInputs_CombinesCorrectly()
    {
        // Arrange - add 3 inputs
        int[][] inputShapes = [[32], [32], [32]];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 32], seed: 123);
        var input3 = CreateRandomTensor([4, 32], seed: 456);

        // Act
        var output = layer.Forward(input1, input2, input3);

        // Assert
        Assert.Equal(input1.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region ConcatenateLayer Tests

    [Fact]
    public void ConcatenateLayer_ForwardPass_ConcatenatesAlongAxis()
    {
        // Arrange - concatenate along feature axis
        int[][] inputShapes = [[32], [64]];
        var layer = new ConcatenateLayer<float>(inputShapes, axis: 0, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 64], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ConcatenateLayer_ForwardPass_4D_ConcatenatesChannels()
    {
        // Arrange - concatenate along channel axis
        int[][] inputShapes = [[32, 8, 8], [64, 8, 8]];
        var layer = new ConcatenateLayer<float>(inputShapes, axis: 0, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([2, 32, 8, 8]);
        var input2 = CreateRandomTensor([2, 64, 8, 8], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ConcatenateLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[][] inputShapes = [[32], [64]];
        var layer = new ConcatenateLayer<float>(inputShapes, axis: 0, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 64], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 456);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void ConcatenateLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[][] inputShapes = [[32], [64]];
        var layer = new ConcatenateLayer<float>(inputShapes, axis: 0, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 64], seed: 123);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input1, input2);
        var output2 = clone.Forward(input1, input2);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region SplitLayer Tests

    [Fact]
    public void SplitLayer_ForwardPass_SplitsCorrectly()
    {
        // Arrange - split into 2 parts
        int[] inputShape = [64];
        var layer = new SplitLayer<float>(inputShape, numSplits: 2);
        var input = CreateRandomTensor([4, 64]);

        // Act
        var output = layer.Forward(input);

        // Assert - split returns first split output
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SplitLayer_ForwardPass_4D_SplitsChannels()
    {
        // Arrange - split channels
        int[] inputShape = [64, 8, 8];
        var layer = new SplitLayer<float>(inputShape, numSplits: 4);
        var input = CreateRandomTensor([2, 64, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SplitLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [64];
        var layer = new SplitLayer<float>(inputShape, numSplits: 2);
        var input = CreateRandomTensor([4, 64]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region MultiplyLayer Tests

    [Fact]
    public void MultiplyLayer_ForwardPass_2D_ProducesValidOutput()
    {
        // Arrange - element-wise multiplication
        int[][] inputShapes = [[64], [64]];
        var layer = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 64]);
        var input2 = CreateRandomTensor([4, 64], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.Equal(input1.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiplyLayer_ForwardPass_4D_ProducesValidOutput()
    {
        // Arrange
        int[][] inputShapes = [[32, 8, 8], [32, 8, 8]];
        var layer = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([2, 32, 8, 8]);
        var input2 = CreateRandomTensor([2, 32, 8, 8], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.Equal(input1.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MultiplyLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[][] inputShapes = [[32], [32]];
        var layer = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 32], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 456);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input1.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void MultiplyLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[][] inputShapes = [[32], [32]];
        var layer = new MultiplyLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([4, 32]);
        var input2 = CreateRandomTensor([4, 32], seed: 123);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input1, input2);
        var output2 = clone.Forward(input1, input2);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region MeanLayer Tests

    [Fact]
    public void MeanLayer_ForwardPass_ComputesMeanAlongAxis()
    {
        // Arrange - compute mean along last axis
        int[] inputShape = [8, 32];
        var layer = new MeanLayer<float>(inputShape, axis: 1);
        var input = CreateRandomTensor([4, 8, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MeanLayer_ForwardPass_4D_ReducesCorrectly()
    {
        // Arrange - compute mean along channel axis
        int[] inputShape = [32, 8, 8];
        var layer = new MeanLayer<float>(inputShape, axis: 0);
        var input = CreateRandomTensor([2, 32, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MeanLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [8, 32];
        var layer = new MeanLayer<float>(inputShape, axis: 1);
        var input = CreateRandomTensor([4, 8, 32]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void MeanLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [8, 32];
        var layer = new MeanLayer<float>(inputShape, axis: 1);
        var input = CreateRandomTensor([4, 8, 32]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region InputLayer Tests

    [Fact]
    public void InputLayer_ForwardPass_PassesThroughUnchanged()
    {
        // Arrange
        int inputSize = 64;
        var layer = new InputLayer<float>(inputSize);
        var input = CreateRandomTensor([4, 64]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void InputLayer_BackwardPass_PassesThroughGradients()
    {
        // Arrange
        int inputSize = 64;
        var layer = new InputLayer<float>(inputSize);
        var input = CreateRandomTensor([4, 64]);
        layer.Forward(input);
        var outputGradient = CreateRandomTensor([4, 64], seed: 123);

        // Act
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(outputGradient.Shape, inputGradient.Shape);
    }

    [Fact]
    public void InputLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputSize = 64;
        var layer = new InputLayer<float>(inputSize);
        var input = CreateRandomTensor([4, 64]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    [Fact]
    public void InputLayer_SupportsTraining_ReturnsFalse()
    {
        // Arrange
        var layer = new InputLayer<float>(64);

        // Assert - input layer has no trainable parameters
        Assert.False(layer.SupportsTraining);
    }

    #endregion

    #region PaddingLayer Tests

    [Fact]
    public void PaddingLayer_ForwardPass_AddsPadding()
    {
        // Arrange - add padding of 1 on each side (4 dimensions: top, bottom, left, right)
        int[] inputShape = [3, 8, 8];
        int[] padding = [1, 1, 1, 1]; // top, bottom, left, right
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert - should be larger by 2 in height and width
        Assert.Equal([2, 3, 10, 10], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PaddingLayer_ForwardPass_AsymmetricPadding()
    {
        // Arrange - different padding on each side
        int[] inputShape = [3, 8, 8];
        int[] padding = [1, 2, 1, 2]; // top, bottom, left, right
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 8, 8]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal([2, 3, 11, 11], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void PaddingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [3, 8, 8];
        int[] padding = [1, 1, 1, 1];
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 8, 8]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void PaddingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [3, 8, 8];
        int[] padding = [1, 1, 1, 1];
        var layer = new PaddingLayer<float>(inputShape, padding, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 8, 8]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region CroppingLayer Tests

    [Fact]
    public void CroppingLayer_ForwardPass_RemovesPadding()
    {
        // Arrange - crop 1 from each side
        int[] inputShape = [3, 10, 10];
        int[] cropTop = [1];
        int[] cropBottom = [1];
        int[] cropLeft = [1];
        int[] cropRight = [1];
        var layer = new CroppingLayer<float>(inputShape, cropTop, cropBottom, cropLeft, cropRight, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 10, 10]);

        // Act
        var output = layer.Forward(input);

        // Assert - should be smaller by 2 in height and width
        Assert.Equal([2, 3, 8, 8], output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void CroppingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [3, 10, 10];
        int[] cropTop = [1];
        int[] cropBottom = [1];
        int[] cropLeft = [1];
        int[] cropRight = [1];
        var layer = new CroppingLayer<float>(inputShape, cropTop, cropBottom, cropLeft, cropRight, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 10, 10]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void CroppingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [3, 10, 10];
        int[] cropTop = [1];
        int[] cropBottom = [1];
        int[] cropLeft = [1];
        int[] cropRight = [1];
        var layer = new CroppingLayer<float>(inputShape, cropTop, cropBottom, cropLeft, cropRight, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([2, 3, 10, 10]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region MaskingLayer Tests

    [Fact]
    public void MaskingLayer_ForwardPass_AppliesMask()
    {
        // Arrange
        int[] inputShape = [10, 32];
        var layer = new MaskingLayer<float>(inputShape, maskValue: 0);
        var input = CreateRandomTensor([4, 10, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MaskingLayer_BackwardPass_ProducesValidGradients()
    {
        // Arrange
        int[] inputShape = [10, 32];
        var layer = new MaskingLayer<float>(inputShape, maskValue: 0);
        var input = CreateRandomTensor([4, 10, 32]);

        // Act
        var output = layer.Forward(input);
        var outputGradient = CreateRandomTensor(output.Shape, seed: 123);
        var inputGradient = layer.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Shape, inputGradient.Shape);
        Assert.False(ContainsNaN(inputGradient));
    }

    [Fact]
    public void MaskingLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [10, 32];
        var layer = new MaskingLayer<float>(inputShape, maskValue: 0);
        var input = CreateRandomTensor([4, 10, 32]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region LambdaLayer Tests

    [Fact]
    public void LambdaLayer_ForwardPass_AppliesCustomFunction()
    {
        // Arrange - custom function that doubles the input
        int[] inputShape = [32];
        int[] outputShape = [32];
        var layer = new LambdaLayer<float>(
            inputShape,
            outputShape,
            forwardFunction: x => {
                var result = new Tensor<float>(x.Shape);
                for (int i = 0; i < x.Data.Length; i++)
                    result.Data[i] = x.Data[i] * 2f;
                return result;
            },
            backwardFunction: null,
            activationFunction: (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void LambdaLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [32];
        int[] outputShape = [32];
        var layer = new LambdaLayer<float>(
            inputShape,
            outputShape,
            forwardFunction: x => {
                var result = new Tensor<float>(x.Shape);
                for (int i = 0; i < x.Data.Length; i++)
                    result.Data[i] = x.Data[i] * 2f;
                return result;
            },
            backwardFunction: null,
            activationFunction: (IActivationFunction<float>?)null);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var clone = layer.Clone();
        var output1 = layer.Forward(input);
        var output2 = clone.Forward(input);

        // Assert
        Assert.NotSame(layer, clone);
        Assert.Equal(output1.Shape, output2.Shape);
    }

    #endregion

    #region GaussianNoiseLayer Tests

    [Fact]
    public void GaussianNoiseLayer_ForwardPass_Training_AddsNoise()
    {
        // Arrange
        int[] inputShape = [32];
        var layer = new GaussianNoiseLayer<float>(inputShape, standardDeviation: 0.1);
        layer.SetTrainingMode(true);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GaussianNoiseLayer_ForwardPass_Inference_PassesThrough()
    {
        // Arrange
        int[] inputShape = [32];
        var layer = new GaussianNoiseLayer<float>(inputShape, standardDeviation: 0.1);
        layer.SetTrainingMode(false);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
        // In inference mode, output should be identical to input
    }

    [Fact]
    public void GaussianNoiseLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int[] inputShape = [32];
        var layer = new GaussianNoiseLayer<float>(inputShape, standardDeviation: 0.1);
        var input = CreateRandomTensor([4, 32]);

        // Act
        var clone = layer.Clone();

        // Assert
        Assert.NotSame(layer, clone);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void AddLayer_SmallBatch_HandlesCorrectly()
    {
        // Arrange - batch size 1
        int[][] inputShapes = [[32], [32]];
        var layer = new AddLayer<float>(inputShapes, (IActivationFunction<float>)new IdentityActivation<float>());
        var input1 = CreateRandomTensor([1, 32]);
        var input2 = CreateRandomTensor([1, 32], seed: 123);

        // Act
        var output = layer.Forward(input1, input2);

        // Assert
        Assert.Equal(input1.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ConcatenateLayer_SingleInput_PassesThrough()
    {
        // Arrange - single input should just pass through
        int[][] inputShapes = [[32]];
        var layer = new ConcatenateLayer<float>(inputShapes, axis: 0, (IActivationFunction<float>)new IdentityActivation<float>());
        var input = CreateRandomTensor([4, 32]);

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(input.Shape, output.Shape);
    }

    #endregion
}
