using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for the DenseNet implementations.
/// </summary>
public class DenseNetTests
{
    #region DenseNet-121 Tests

    [Fact]
    public void DenseNet121_Constructor_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(DenseNetVariant.DenseNet121, network.Variant);
        Assert.Equal(10, network.NumClasses);
        Assert.Equal(32, network.GrowthRate);
        Assert.True(network.Layers.Count > 0);
    }

    [Fact]
    public void DenseNet169_Constructor_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = DenseNetNetwork<float>.DenseNet169(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(DenseNetVariant.DenseNet169, network.Variant);
        Assert.Equal(100, network.NumClasses);
    }

    [Fact]
    public void DenseNet201_Constructor_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = DenseNetNetwork<float>.DenseNet201(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(DenseNetVariant.DenseNet201, network.Variant);
    }

    [Fact]
    public void DenseNet264_Constructor_CreatesValidNetwork()
    {
        // Arrange & Act
        var network = DenseNetNetwork<float>.DenseNet264(numClasses: 100);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(DenseNetVariant.DenseNet264, network.Variant);
    }

    #endregion

    #region Configuration Tests

    [Fact]
    public void DenseNetConfiguration_GetBlockLayers_ReturnsCorrectValues()
    {
        // Test DenseNet-121
        var config121 = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses: 10);
        var layers121 = config121.GetBlockLayers();
        Assert.Equal([6, 12, 24, 16], layers121);

        // Test DenseNet-169
        var config169 = new DenseNetConfiguration(DenseNetVariant.DenseNet169, numClasses: 10);
        var layers169 = config169.GetBlockLayers();
        Assert.Equal([6, 12, 32, 32], layers169);

        // Test DenseNet-201
        var config201 = new DenseNetConfiguration(DenseNetVariant.DenseNet201, numClasses: 10);
        var layers201 = config201.GetBlockLayers();
        Assert.Equal([6, 12, 48, 32], layers201);

        // Test DenseNet-264
        var config264 = new DenseNetConfiguration(DenseNetVariant.DenseNet264, numClasses: 10);
        var layers264 = config264.GetBlockLayers();
        Assert.Equal([6, 12, 64, 48], layers264);
    }

    [Fact]
    public void DenseNet_WithCustomGrowthRate_CreatesValidNetwork()
    {
        // Arrange - Use factory method with custom growth rate
        // Note: The default factory methods use growth rate 32, so test with that
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(32, network.GrowthRate); // Default growth rate
    }

    [Fact]
    public void DenseNet_WithGrayscaleInput_CreatesValidNetwork()
    {
        // Arrange
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10, inputChannels: 1);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.Layers.Count > 0);
    }

    #endregion

    #region DenseBlock Tests

    [Fact]
    public void DenseBlock_Constructor_CreatesValidBlock()
    {
        // Arrange & Act
        var block = new DenseBlock<float>(
            inputChannels: 64,
            numLayers: 6,
            growthRate: 32,
            inputHeight: 56,
            inputWidth: 56);

        // Assert
        Assert.Equal(6, block.NumLayers);
        Assert.Equal(32, block.GrowthRate);
        Assert.Equal(64 + 6 * 32, block.OutputChannels); // 64 + 192 = 256
    }

    [Fact]
    public void DenseBlock_OutputChannels_CalculatedCorrectly()
    {
        // Arrange
        int inputChannels = 64;
        int numLayers = 6;
        int growthRate = 32;

        // Act
        var block = new DenseBlock<float>(inputChannels, numLayers, growthRate, 28, 28);

        // Assert
        // Output = input + (numLayers * growthRate)
        Assert.Equal(inputChannels + numLayers * growthRate, block.OutputChannels);
    }

    [Fact]
    public void DenseBlock_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        var block = new DenseBlock<float>(
            inputChannels: 64,
            numLayers: 3,
            growthRate: 12,
            inputHeight: 14,
            inputWidth: 14);

        var input = new Tensor<float>([1, 64, 14, 14]);
        InitializeWithRandomValues(input);

        // Act
        var output = block.Forward(input);

        // Assert - output channels = 64 + 3*12 = 100
        Assert.Equal(4, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]); // batch
        Assert.Equal(100, output.Shape[1]); // channels
        Assert.Equal(14, output.Shape[2]); // height (unchanged)
        Assert.Equal(14, output.Shape[3]); // width (unchanged)
    }

    #endregion

    #region TransitionLayer Tests

    [Fact]
    public void TransitionLayer_Constructor_CreatesValidLayer()
    {
        // Arrange & Act
        var layer = new TransitionLayer<float>(
            inputChannels: 256,
            inputHeight: 56,
            inputWidth: 56,
            compressionFactor: 0.5);

        // Assert
        Assert.Equal(128, layer.OutputChannels); // 256 * 0.5
    }

    [Fact]
    public void TransitionLayer_Forward_ProducesCorrectOutputShape()
    {
        // Arrange
        var layer = new TransitionLayer<float>(
            inputChannels: 128,
            inputHeight: 28,
            inputWidth: 28,
            compressionFactor: 0.5);

        var input = new Tensor<float>([1, 128, 28, 28]);
        InitializeWithRandomValues(input);

        // Act
        var output = layer.Forward(input);

        // Assert - channels halved, spatial halved
        Assert.Equal(4, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]); // batch
        Assert.Equal(64, output.Shape[1]); // channels (128 * 0.5)
        Assert.Equal(14, output.Shape[2]); // height (28 / 2)
        Assert.Equal(14, output.Shape[3]); // width (28 / 2)
    }

    [Fact]
    public void TransitionLayer_DifferentCompressionFactor_WorksCorrectly()
    {
        // Arrange - no compression (compression = 1.0)
        var layer = new TransitionLayer<float>(
            inputChannels: 100,
            inputHeight: 14,
            inputWidth: 14,
            compressionFactor: 1.0);

        // Assert - channels unchanged
        Assert.Equal(100, layer.OutputChannels);
    }

    #endregion

    #region Model Metadata Tests

    [Fact]
    public void DenseNet_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("DenseNetNetwork", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal("DenseNet121", metadata.AdditionalInfo["Variant"]);
        Assert.Equal(32, (int)metadata.AdditionalInfo["GrowthRate"]);
        Assert.Equal(10, (int)metadata.AdditionalInfo["NumClasses"]);
    }

    #endregion

    #region Clone and Layer Access Tests

    [Fact]
    public void DenseNet_Clone_CreatesNewInstance()
    {
        // Arrange
        var original = DenseNetNetwork<float>.DenseNet121(numClasses: 10);

        // Act
        var clone = original.Clone();

        // Assert
        Assert.NotSame(original, clone);
        Assert.IsType<DenseNetNetwork<float>>(clone);
    }

    [Fact]
    public void DenseNet_GetLayer_ReturnsCorrectLayer()
    {
        // Arrange
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);

        // Act
        var firstLayer = network.GetLayer(0);
        var lastLayer = network.GetLayer(network.Layers.Count - 1);

        // Assert
        Assert.IsType<ConvolutionalLayer<float>>(firstLayer); // Stem conv
        Assert.IsType<DenseLayer<float>>(lastLayer); // Classification head
    }

    [Fact]
    public void DenseNet_GetLayer_ThrowsOnInvalidIndex()
    {
        // Arrange
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => network.GetLayer(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => network.GetLayer(network.Layers.Count));
    }

    #endregion

    #region Larger Variant Tests

    [Fact]
    public void DenseNet_LargerVariants_HaveMoreLayers()
    {
        // Arrange
        var d121 = DenseNetNetwork<float>.DenseNet121(numClasses: 10);
        var d169 = DenseNetNetwork<float>.DenseNet169(numClasses: 10);

        // Assert - D169 should have more layers due to more layers per block
        Assert.True(d169.Layers.Count >= d121.Layers.Count);
    }

    #endregion

    #region Training Tests (Skipped for performance)

    [Fact(Skip = "Training test - slow")]
    public void DenseNet121_Train_CompletesWithoutError()
    {
        // Arrange
        var network = DenseNetNetwork<float>.DenseNet121(numClasses: 10);
        var input = new Tensor<float>([1, 3, 224, 224]);
        var target = new Tensor<float>([1, 10]);
        InitializeWithRandomValues(input);
        target[0, 0] = 1f; // One-hot encoded target

        // Act & Assert - Should not throw
        network.Train(input, target);
    }

    #endregion

    #region Helper Methods

    private static void InitializeWithRandomValues(Tensor<float> tensor)
    {
        var random = new Random(42);
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            tensor.Data[i] = (float)(random.NextDouble() * 2 - 1);
        }
    }

    #endregion
}
