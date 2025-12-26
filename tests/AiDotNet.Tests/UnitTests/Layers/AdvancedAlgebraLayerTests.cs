using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Layers;

/// <summary>
/// Unit tests for advanced algebra neural network layers.
/// </summary>
public class AdvancedAlgebraLayerTests
{
    #region OctonionLinearLayer Tests

    [Fact]
    public void OctonionLinearLayer_Construction_SetsCorrectShape()
    {
        // Arrange & Act
        var layer = new OctonionLinearLayer<double>(4, 2);

        // Assert
        Assert.Equal(4, layer.InputFeatures);
        Assert.Equal(2, layer.OutputFeatures);
        // InputFeatures * 8 = 32, OutputFeatures * 8 = 16
        // (InputShape and OutputShape are protected, so we test via Forward)
    }

    [Fact]
    public void OctonionLinearLayer_ParameterCount_IsCorrect()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(4, 2);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // (4 * 2 + 2) * 8 = (8 + 2) * 8 = 80
        Assert.Equal(80, paramCount);
    }

    [Fact]
    public void OctonionLinearLayer_Forward_ProducesCorrectShape()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(4, 2);
        int batchSize = 3;
        var input = new Tensor<double>([batchSize, 32]); // 4 octonions * 8 components

        // Initialize with small values
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < 32; i++)
            {
                input[b, i] = 0.1 * (i + 1);
            }
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(16, output.Shape[1]); // 2 octonions * 8 components
    }

    [Fact]
    public void OctonionLinearLayer_GetSetParameters_RoundTrips()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(2, 1);
        var originalParams = layer.GetParameters();

        // Modify parameters
        var modifiedParams = new Vector<double>(originalParams.Length);
        for (int i = 0; i < modifiedParams.Length; i++)
        {
            modifiedParams[i] = originalParams[i] * 2.0;
        }
        layer.SetParameters(modifiedParams);

        // Act
        var retrievedParams = layer.GetParameters();

        // Assert
        Assert.Equal(modifiedParams.Length, retrievedParams.Length);
        for (int i = 0; i < modifiedParams.Length; i++)
        {
            Assert.Equal(modifiedParams[i], retrievedParams[i], precision: 10);
        }
    }

    [Fact]
    public void OctonionLinearLayer_ResetState_ClearsInternalState()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(2, 1);
        var input = new Tensor<double>([1, 16]);
        for (int i = 0; i < 16; i++) input[0, i] = 0.1;

        layer.Forward(input);

        // Act
        layer.ResetState();

        // Assert - ResetState shouldn't throw and layer should be usable
        var output = layer.Forward(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void OctonionLinearLayer_SupportsTraining_IsTrue()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(4, 2);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void OctonionLinearLayer_SupportsJitCompilation_IsFalse()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(4, 2);

        // Assert
        Assert.False(layer.SupportsJitCompilation);
    }

    #endregion

    #region HyperbolicLinearLayer Tests

    [Fact]
    public void HyperbolicLinearLayer_Construction_SetsCorrectShape()
    {
        // Arrange & Act
        var layer = new HyperbolicLinearLayer<double>(10, 5);

        // Assert
        Assert.Equal(10, layer.InputFeatures);
        Assert.Equal(5, layer.OutputFeatures);
    }

    [Fact]
    public void HyperbolicLinearLayer_Construction_RequiresNegativeCurvature()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new HyperbolicLinearLayer<double>(10, 5, curvature: 1.0));
        Assert.Throws<ArgumentException>(() => new HyperbolicLinearLayer<double>(10, 5, curvature: 0.0));
    }

    [Fact]
    public void HyperbolicLinearLayer_Forward_ProducesCorrectShape()
    {
        // Arrange
        var layer = new HyperbolicLinearLayer<double>(10, 5);
        int batchSize = 4;
        var input = new Tensor<double>([batchSize, 10]);

        // Initialize with small values (inside Poincare ball)
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < 10; i++)
            {
                input[b, i] = 0.01 * (i + 1);
            }
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(5, output.Shape[1]);
    }

    [Fact]
    public void HyperbolicLinearLayer_ParameterCount_IsCorrect()
    {
        // Arrange
        var layer = new HyperbolicLinearLayer<double>(10, 5);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // Weights: 5 * 10 = 50, Biases: 5 * 10 = 50, Total: 100
        Assert.Equal(100, paramCount);
    }

    [Fact]
    public void HyperbolicLinearLayer_GetSetParameters_RoundTrips()
    {
        // Arrange
        var layer = new HyperbolicLinearLayer<double>(5, 3);
        var originalParams = layer.GetParameters();

        // Modify parameters (keep small to stay in valid region)
        var modifiedParams = new Vector<double>(originalParams.Length);
        for (int i = 0; i < modifiedParams.Length; i++)
        {
            modifiedParams[i] = originalParams[i] * 0.5;
        }
        layer.SetParameters(modifiedParams);

        // Act
        var retrievedParams = layer.GetParameters();

        // Assert
        Assert.Equal(modifiedParams.Length, retrievedParams.Length);
        for (int i = 0; i < modifiedParams.Length; i++)
        {
            Assert.Equal(modifiedParams[i], retrievedParams[i], precision: 10);
        }
    }

    [Fact]
    public void HyperbolicLinearLayer_SupportsTraining_IsTrue()
    {
        // Arrange
        var layer = new HyperbolicLinearLayer<double>(10, 5);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    #endregion

    #region SparseLinearLayer Tests

    [Fact]
    public void SparseLinearLayer_Construction_SetsCorrectShape()
    {
        // Arrange & Act
        var layer = new SparseLinearLayer<double>(100, 50, sparsity: 0.9);

        // Assert
        Assert.Equal(100, layer.InputFeatures);
        Assert.Equal(50, layer.OutputFeatures);
    }

    [Fact]
    public void SparseLinearLayer_Construction_RequiresValidSparsity()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new SparseLinearLayer<double>(100, 50, sparsity: -0.1));
        Assert.Throws<ArgumentException>(() => new SparseLinearLayer<double>(100, 50, sparsity: 1.0));
        Assert.Throws<ArgumentException>(() => new SparseLinearLayer<double>(100, 50, sparsity: 1.5));
    }

    [Fact]
    public void SparseLinearLayer_Forward_ProducesCorrectShape()
    {
        // Arrange
        var layer = new SparseLinearLayer<double>(20, 10, sparsity: 0.8);
        int batchSize = 5;
        var input = new Tensor<double>([batchSize, 20]);

        // Initialize with small values
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < 20; i++)
            {
                input[b, i] = 0.1 * (i + 1);
            }
        }

        // Act
        var output = layer.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(10, output.Shape[1]);
    }

    [Fact]
    public void SparseLinearLayer_ParameterCount_ReflectsSparsity()
    {
        // Arrange
        // 100 * 50 = 5000 weights, with 90% sparsity = 500 non-zeros + 50 biases = 550
        var layer = new SparseLinearLayer<double>(100, 50, sparsity: 0.9);

        // Act
        int paramCount = layer.ParameterCount;

        // Assert
        // Approximate check - should be much less than 5050 (dense case)
        Assert.True(paramCount < 1000, $"Parameter count {paramCount} should be less than 1000 for 90% sparsity");
        Assert.True(paramCount > 100, $"Parameter count {paramCount} should be greater than 100");
    }

    [Fact]
    public void SparseLinearLayer_GetSetParameters_RoundTrips()
    {
        // Arrange
        var layer = new SparseLinearLayer<double>(20, 10, sparsity: 0.8);
        var originalParams = layer.GetParameters();

        // Modify parameters
        var modifiedParams = new Vector<double>(originalParams.Length);
        for (int i = 0; i < modifiedParams.Length; i++)
        {
            modifiedParams[i] = originalParams[i] * 2.0;
        }
        layer.SetParameters(modifiedParams);

        // Act
        var retrievedParams = layer.GetParameters();

        // Assert
        Assert.Equal(modifiedParams.Length, retrievedParams.Length);
        for (int i = 0; i < modifiedParams.Length; i++)
        {
            Assert.Equal(modifiedParams[i], retrievedParams[i], precision: 10);
        }
    }

    [Fact]
    public void SparseLinearLayer_SupportsTraining_IsTrue()
    {
        // Arrange
        var layer = new SparseLinearLayer<double>(100, 50);

        // Assert
        Assert.True(layer.SupportsTraining);
    }

    [Fact]
    public void SparseLinearLayer_SupportsJitCompilation_IsFalse()
    {
        // Arrange
        var layer = new SparseLinearLayer<double>(100, 50);

        // Assert
        Assert.False(layer.SupportsJitCompilation);
    }

    #endregion

    #region Training Integration Tests

    [Fact]
    public void OctonionLinearLayer_ForwardBackward_ProducesGradients()
    {
        // Arrange
        var layer = new OctonionLinearLayer<double>(2, 1);
        var input = new Tensor<double>([1, 16]); // 2 octonions * 8 components
        for (int i = 0; i < 16; i++) input[0, i] = 0.1 * (i + 1);

        var outputGrad = new Tensor<double>([1, 8]); // 1 octonion * 8 components
        for (int i = 0; i < 8; i++) outputGrad[0, i] = 1.0;

        // Act
        layer.Forward(input);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.NotNull(inputGrad);
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void HyperbolicLinearLayer_ForwardBackward_ProducesGradients()
    {
        // Arrange
        var layer = new HyperbolicLinearLayer<double>(5, 3);
        var input = new Tensor<double>([2, 5]);
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < 5; i++)
                input[b, i] = 0.01 * (i + 1);

        var outputGrad = new Tensor<double>([2, 3]);
        for (int b = 0; b < 2; b++)
            for (int o = 0; o < 3; o++)
                outputGrad[b, o] = 1.0;

        // Act
        layer.Forward(input);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.NotNull(inputGrad);
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void SparseLinearLayer_ForwardBackward_ProducesGradients()
    {
        // Arrange
        var layer = new SparseLinearLayer<double>(10, 5, sparsity: 0.5);
        var input = new Tensor<double>([2, 10]);
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < 10; i++)
                input[b, i] = 0.1 * (i + 1);

        var outputGrad = new Tensor<double>([2, 5]);
        for (int b = 0; b < 2; b++)
            for (int o = 0; o < 5; o++)
                outputGrad[b, o] = 1.0;

        // Act
        layer.Forward(input);
        var inputGrad = layer.Backward(outputGrad);

        // Assert
        Assert.NotNull(inputGrad);
        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    #endregion
}
