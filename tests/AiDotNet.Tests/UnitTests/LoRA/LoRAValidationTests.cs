using AiDotNet.LinearAlgebra;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.LoRA;

/// <summary>
/// Validation tests for LoRA module to verify production bug fixes.
/// </summary>
public class LoRAValidationTests
{
    #region LoRALayer Input/Output Validation Tests

    [Fact]
    public void LoRALayer_InvalidInputSize_Zero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: 0, outputSize: 10, rank: 4));

        Assert.Contains("Input size", ex.Message);
    }

    [Fact]
    public void LoRALayer_InvalidInputSize_Negative_ThrowsArgumentOutOfRangeException()
    {
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: -5, outputSize: 10, rank: 4));

        Assert.Contains("Input size", ex.Message);
    }

    [Fact]
    public void LoRALayer_InvalidOutputSize_Zero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: 10, outputSize: 0, rank: 4));

        Assert.Contains("Output size", ex.Message);
    }

    [Fact]
    public void LoRALayer_InvalidOutputSize_Negative_ThrowsArgumentOutOfRangeException()
    {
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: 10, outputSize: -5, rank: 4));

        Assert.Contains("Output size", ex.Message);
    }

    [Fact]
    public void LoRALayer_InvalidRank_Zero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: 10, outputSize: 10, rank: 0));

        Assert.Contains("Rank", ex.Message);
    }

    [Fact]
    public void LoRALayer_InvalidRank_Negative_ThrowsArgumentOutOfRangeException()
    {
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: 10, outputSize: 10, rank: -4));

        Assert.Contains("Rank", ex.Message);
    }

    [Fact]
    public void LoRALayer_InvalidRank_ExceedsMinDimension_ThrowsArgumentOutOfRangeException()
    {
        // Rank cannot exceed min(inputSize, outputSize)
        // Arrange & Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LoRALayer<double>(inputSize: 5, outputSize: 10, rank: 6)); // rank > min(5, 10) = 5

        Assert.Contains("cannot exceed", ex.Message);
    }

    [Fact]
    public void LoRALayer_ValidParameters_CreatesSuccessfully()
    {
        // Arrange & Act
        var layer = new LoRALayer<double>(inputSize: 10, outputSize: 8, rank: 4);

        // Assert
        Assert.Equal(4, layer.Rank);
        Assert.True(layer.SupportsTraining);
    }

    #endregion

    #region AdaLoRAAdapter PruningInterval Validation Tests

    [Fact]
    public void AdaLoRAAdapter_InvalidPruningInterval_Zero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AdaLoRAAdapter<double>(baseLayer, maxRank: 4, alpha: 4, freezeBaseLayer: true,
                rankPruningThreshold: 0.1, minRank: 1, pruningInterval: 0));

        Assert.Contains("Pruning interval", ex.Message);
    }

    [Fact]
    public void AdaLoRAAdapter_InvalidPruningInterval_Negative_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act & Assert
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AdaLoRAAdapter<double>(baseLayer, maxRank: 4, alpha: 4, freezeBaseLayer: true,
                rankPruningThreshold: 0.1, minRank: 1, pruningInterval: -10));

        Assert.Contains("Pruning interval", ex.Message);
    }

    [Fact]
    public void AdaLoRAAdapter_ValidPruningInterval_CreatesSuccessfully()
    {
        // Arrange
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act
        var adapter = new AdaLoRAAdapter<double>(baseLayer, maxRank: 4, alpha: 4, freezeBaseLayer: true,
            rankPruningThreshold: 0.1, minRank: 1, pruningInterval: 100);

        // Assert
        Assert.NotNull(adapter);
    }

    #endregion

    #region DefaultLoRAConfiguration CreateAdapter Tests

    [Fact]
    public void DefaultLoRAConfiguration_StandardLoRAAdapter_CreatesSuccessfully()
    {
        // Arrange
        var config = new DefaultLoRAConfiguration<double>(rank: 4, alpha: 4, freezeBaseLayer: true);
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act
        var adaptedLayer = config.ApplyLoRA(baseLayer);

        // Assert
        Assert.IsType<StandardLoRAAdapter<double>>(adaptedLayer);
    }

    [Fact]
    public void DefaultLoRAConfiguration_DoRAAdapter_CreatesSuccessfully()
    {
        // Arrange - DoRA has matching constructor signature (ILayer<T>, int, double, bool)
        var doraAdapter = new DoRAAdapter<double>(new DenseLayer<double>(10, 5), rank: 4, alpha: 4, freezeBaseLayer: true);
        var config = new DefaultLoRAConfiguration<double>(rank: 4, alpha: 4, freezeBaseLayer: true, loraAdapter: doraAdapter);
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act
        var adaptedLayer = config.ApplyLoRA(baseLayer);

        // Assert
        Assert.IsType<DoRAAdapter<double>>(adaptedLayer);
    }

    [Fact]
    public void DefaultLoRAConfiguration_AdaLoRAAdapter_CreatesSuccessfully()
    {
        // Arrange - AdaLoRA has compatible constructor (4th param is bool, rest have defaults)
        var adaloraAdapter = new AdaLoRAAdapter<double>(new DenseLayer<double>(10, 5), maxRank: 4);
        var config = new DefaultLoRAConfiguration<double>(rank: 4, alpha: 4, freezeBaseLayer: true, loraAdapter: adaloraAdapter);
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act
        var adaptedLayer = config.ApplyLoRA(baseLayer);

        // Assert
        Assert.IsType<AdaLoRAAdapter<double>>(adaptedLayer);
    }

    [Fact]
    public void DefaultLoRAConfiguration_QLoRAAdapter_ThrowsInvalidOperationException()
    {
        // QLoRA has incompatible constructor signature (4th param is QuantizationType, not bool)
        // This should throw a helpful error message
        // Arrange
        var qloraAdapter = new QLoRAAdapter<double>(new DenseLayer<double>(10, 5), rank: 4);
        var config = new DefaultLoRAConfiguration<double>(rank: 4, alpha: 4, freezeBaseLayer: true, loraAdapter: qloraAdapter);
        var baseLayer = new DenseLayer<double>(10, 5);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => config.ApplyLoRA(baseLayer));
        Assert.Contains("compatible constructor", ex.Message);
    }

    [Fact]
    public void DefaultLoRAConfiguration_NonAdaptableLayers_PassThrough()
    {
        // Layers without trainable weights should pass through unchanged
        // Arrange
        var config = new DefaultLoRAConfiguration<double>(rank: 4);
        var poolingLayer = new MaxPoolingLayer<double>(inputShape: new[] { 28, 28, 1 }, poolSize: 2, stride: 2);

        // Act
        var result = config.ApplyLoRA(poolingLayer);

        // Assert - should be same instance, not wrapped
        Assert.Same(poolingLayer, result);
    }

    #endregion

    #region Matrix Indexing Bug Fix Verification Tests

    [Fact]
    public void LoRAAdapterBase_MergeWeights_ReturnsCorrectShape()
    {
        // Verify that MergeWeights() returns [inputSize, outputSize]
        // Arrange
        int inputSize = 10;
        int outputSize = 5;
        int rank = 4;
        var layer = new LoRALayer<double>(inputSize, outputSize, rank);

        // Act
        var merged = layer.MergeWeights();

        // Assert - shape should be [inputSize, outputSize]
        Assert.Equal(inputSize, merged.Rows);
        Assert.Equal(outputSize, merged.Columns);
    }

    [Fact]
    public void StandardLoRAAdapter_MergeToOriginalLayer_DoesNotThrow()
    {
        // This tests that the matrix indexing fix works correctly
        // Before the fix, this would throw IndexOutOfRangeException when outputSize > inputSize
        // Arrange
        int inputSize = 5; // Smaller than outputSize to catch the bug
        int outputSize = 10;
        int rank = 4;
        var baseLayer = new DenseLayer<double>(inputSize, outputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank: rank, alpha: rank, freezeBaseLayer: true);

        // Act & Assert - should not throw
        var merged = adapter.MergeToOriginalLayer();
        Assert.NotNull(merged);
    }

    [Fact]
    public void DoRAAdapter_MergeToOriginalLayer_DoesNotThrow()
    {
        // This tests that the matrix indexing fix works correctly
        // Arrange
        int inputSize = 5;
        int outputSize = 10;
        int rank = 4;
        var baseLayer = new DenseLayer<double>(inputSize, outputSize);
        var adapter = new DoRAAdapter<double>(baseLayer, rank: rank, alpha: rank, freezeBaseLayer: true);

        // Act & Assert - should not throw
        var merged = adapter.MergeToOriginalLayer();
        Assert.NotNull(merged);
    }

    [Fact]
    public void QLoRAAdapter_MergeToOriginalLayer_DoesNotThrow()
    {
        // This tests that the matrix indexing fix works correctly
        // Arrange
        int inputSize = 5;
        int outputSize = 10;
        int rank = 4;
        var baseLayer = new DenseLayer<double>(inputSize, outputSize);
        var adapter = new QLoRAAdapter<double>(baseLayer, rank: rank, alpha: rank, freezeBaseLayer: true);

        // Act & Assert - should not throw
        var merged = adapter.MergeToOriginalLayer();
        Assert.NotNull(merged);
    }

    [Fact]
    public void VeRAAdapter_MergeToOriginalLayer_DoesNotThrow()
    {
        // This tests that VeRA's matrix indexing is correct
        // Arrange
        int inputSize = 5;
        int outputSize = 10;
        int rank = 4;

        // Initialize shared matrices first (required for VeRA)
        VeRAAdapter<double>.InitializeSharedMatrices(inputSize, outputSize, rank, seed: 42);

        try
        {
            var baseLayer = new DenseLayer<double>(inputSize, outputSize);
            var adapter = new VeRAAdapter<double>(baseLayer, rank: rank, alpha: rank, freezeBaseLayer: true);

            // Act & Assert - should not throw
            var merged = adapter.MergeToOriginalLayer();
            Assert.NotNull(merged);
        }
        finally
        {
            // Clean up shared matrices
            VeRAAdapter<double>.ResetSharedMatrices();
        }
    }

    #endregion

    #region DoRAAdapter Forward Pass Bug Fix Verification Tests

    [Fact]
    public void DoRAAdapter_Forward_DoesNotThrowWithAsymmetricDimensions()
    {
        // This tests that the matrix indexing fix in Forward() works correctly
        // Before the fix, this would throw IndexOutOfRangeException when outputSize > inputSize
        // Arrange
        int inputSize = 5;
        int outputSize = 10;
        int rank = 4;
        int batchSize = 2;
        var baseLayer = new DenseLayer<double>(inputSize, outputSize);
        var adapter = new DoRAAdapter<double>(baseLayer, rank: rank, alpha: rank, freezeBaseLayer: true);

        // Create input tensor [batchSize, inputSize]
        var inputData = new double[batchSize * inputSize];
        for (int i = 0; i < inputData.Length; i++)
        {
            inputData[i] = 0.1 * i;
        }
        var input = new Tensor<double>(new[] { batchSize, inputSize }, new Vector<double>(inputData));

        // Act & Assert - should not throw
        var output = adapter.Forward(input);
        Assert.NotNull(output);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(outputSize, output.Shape[1]);
    }

    #endregion
}
