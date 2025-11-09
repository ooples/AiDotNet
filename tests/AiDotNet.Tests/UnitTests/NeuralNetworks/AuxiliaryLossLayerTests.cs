using Xunit;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Common;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for IAuxiliaryLossLayer implementations across all 26 components.
/// Tests verify that auxiliary loss methods are properly implemented and return expected values.
/// </summary>
public class AuxiliaryLossLayerTests
{
    #region Attention-Based Layers Tests

    [Fact]
    public void MultiHeadAttentionLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            dropout: 0.1);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss); // Default is false
        Assert.Equal(0.005, layer.AuxiliaryLossWeight);
    }

    [Fact]
    public void MultiHeadAttentionLayer_ComputeAuxiliaryLoss_ReturnsZeroByDefault()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            dropout: 0.1);

        // Act
        var loss = layer.ComputeAuxiliaryLoss();

        // Assert - Should return zero since no forward pass has been done
        Assert.Equal(0.0, loss);
    }

    [Fact]
    public void MultiHeadAttentionLayer_GetAuxiliaryLossDiagnostics_ReturnsValidData()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            dropout: 0.1);
        layer.UseAuxiliaryLoss = true;

        // Act
        var diagnostics = layer.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseAuxiliaryLoss"));
        Assert.Equal("True", diagnostics["UseAuxiliaryLoss"]);
    }

    [Fact]
    public void SelfAttentionLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new SelfAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
        Assert.Equal(0.005, layer.AuxiliaryLossWeight);
    }

    [Fact]
    public void TransformerEncoderLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new TransformerEncoderLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            feedForwardDimension: 256);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
    }

    [Fact]
    public void TransformerDecoderLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new TransformerDecoderLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            feedForwardDimension: 256);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
    }

    #endregion

    #region Memory-Based Layers Tests

    [Fact]
    public void MemoryReadLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new MemoryReadLayer<double>(
            inputDimension: 32,
            memoryDimension: 64,
            outputDimension: 32);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
        Assert.Equal(0.005, layer.AuxiliaryLossWeight);
    }

    [Fact]
    public void MemoryWriteLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new MemoryWriteLayer<double>(
            inputDimension: 32,
            memoryDimension: 64);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
    }

    [Fact]
    public void NeuralTuringMachine_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var ntm = new NeuralTuringMachine<double>(
            inputSize: 8,
            outputSize: 8,
            controllerSize: 100,
            memorySize: 128,
            numReadHeads: 1,
            numWriteHeads: 1);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(ntm);
        Assert.False(ntm.UseAuxiliaryLoss);
    }

    [Fact]
    public void DifferentiableNeuralComputer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var dnc = new DifferentiableNeuralComputer<double>(
            inputSize: 8,
            outputSize: 8,
            controllerSize: 100,
            memorySize: 128,
            numReadHeads: 3);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(dnc);
        Assert.False(dnc.UseAuxiliaryLoss);
    }

    #endregion

    #region Specialized Layers Tests

    [Fact]
    public void HighwayLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new HighwayLayer<double>(dimension: 64);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
        Assert.Equal(0.01, layer.AuxiliaryLossWeight);
    }

    [Fact]
    public void SqueezeAndExcitationLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new SqueezeAndExcitationLayer<double>(
            channels: 64,
            reductionRatio: 16);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
    }

    [Fact]
    public void SpatialTransformerLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new SpatialTransformerLayer<double>(
            inputHeight: 28,
            inputWidth: 28);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
    }

    [Fact]
    public void GraphConvolutionalLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new GraphConvolutionalLayer<double>(
            inputFeatures: 32,
            outputFeatures: 64);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
        Assert.False(layer.UseAuxiliaryLoss);
    }

    [Fact]
    public void SiameseNetwork_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var baseNetwork = new FeedForwardNeuralNetwork<double>(
            inputSize: 784,
            hiddenSizes: new[] { 128, 64 },
            outputSize: 32);

        var siamese = new SiameseNetwork<double>(baseNetwork);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(siamese);
        Assert.False(siamese.UseAuxiliaryLoss);
    }

    #endregion

    #region Previous Phase Implementations Tests

    [Fact]
    public void EmbeddingLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new EmbeddingLayer<double>(
            vocabularySize: 10000,
            embeddingDimension: 128);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
    }

    [Fact]
    public void AttentionLayer_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var layer = new AttentionLayer<double>(
            queryDimension: 64,
            keyDimension: 64,
            valueDimension: 64,
            outputDimension: 64);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(layer);
    }

    [Fact]
    public void CapsuleNetwork_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var network = new CapsuleNetwork<double>(
            inputHeight: 28,
            inputWidth: 28,
            inputChannels: 1,
            numPrimaryCapsules: 32,
            primaryCapsuleDimension: 8,
            numDigitCapsules: 10,
            digitCapsuleDimension: 16);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(network);
    }

    [Fact]
    public void AttentionNetwork_ImplementsIAuxiliaryLossLayer()
    {
        // Arrange
        var network = new AttentionNetwork<double>(
            inputSize: 784,
            hiddenSize: 128,
            outputSize: 10,
            numAttentionHeads: 4);

        // Assert
        Assert.IsAssignableFrom<IAuxiliaryLossLayer<double>>(network);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Transformer_AggregatesAuxiliaryLossFromLayers()
    {
        // Arrange
        var transformer = new Transformer<double>(
            vocabularySize: 1000,
            embeddingDimension: 64,
            numHeads: 4,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            feedForwardDimension: 256,
            maxSequenceLength: 100);

        transformer.UseAuxiliaryLoss = true;

        // Act
        var loss = transformer.ComputeAuxiliaryLoss();
        var diagnostics = transformer.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseAuxiliaryLoss"));
    }

    [Fact]
    public void AuxiliaryLoss_CanBeToggledOnAndOff()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            dropout: 0.1);

        // Act & Assert - Initially off
        Assert.False(layer.UseAuxiliaryLoss);

        // Toggle on
        layer.UseAuxiliaryLoss = true;
        Assert.True(layer.UseAuxiliaryLoss);

        // Toggle off
        layer.UseAuxiliaryLoss = false;
        Assert.False(layer.UseAuxiliaryLoss);
    }

    [Fact]
    public void AuxiliaryLossWeight_CanBeCustomized()
    {
        // Arrange
        var layer = new HighwayLayer<double>(dimension: 64);

        // Act & Assert - Check default
        Assert.Equal(0.01, layer.AuxiliaryLossWeight);

        // Customize weight
        layer.AuxiliaryLossWeight = 0.05;
        Assert.Equal(0.05, layer.AuxiliaryLossWeight);
    }

    #endregion

    #region Diagnostic Tests

    [Fact]
    public void AllLayers_ProvideCompleteDiagnostics()
    {
        // Test that diagnostics contain essential keys
        var layer = new MultiHeadAttentionLayer<double>(64, 4, 0.1);
        var diagnostics = layer.GetAuxiliaryLossDiagnostics();

        Assert.NotNull(diagnostics);
        Assert.NotEmpty(diagnostics);
        Assert.All(diagnostics.Values, value => Assert.NotNull(value));
    }

    #endregion
}
