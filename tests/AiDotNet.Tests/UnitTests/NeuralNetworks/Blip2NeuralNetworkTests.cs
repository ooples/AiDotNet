using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for BLIP-2 neural network constructor validation.
/// Note: Full integration tests require ONNX model files which are not included.
/// These tests verify parameter validation and error handling.
/// </summary>
public class Blip2NeuralNetworkTests
{
    #region ONNX Mode Constructor Tests

    [Fact]
    public void Constructor_Onnx_WithNullVisionEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                null!,
                "qformer.onnx",
                "llm.onnx",
                tokenizer));

        Assert.Contains("Vision encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithEmptyVisionEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                "",
                "qformer.onnx",
                "llm.onnx",
                tokenizer));

        Assert.Contains("Vision encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithNullQFormerPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                "vision.onnx",
                null!,
                "llm.onnx",
                tokenizer));

        Assert.Contains("Q-Former path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithEmptyQFormerPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                "vision.onnx",
                "",
                "llm.onnx",
                tokenizer));

        Assert.Contains("Q-Former path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithNullLlmPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                "vision.onnx",
                "qformer.onnx",
                null!,
                tokenizer));

        Assert.Contains("Language model path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithEmptyLlmPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                "vision.onnx",
                "qformer.onnx",
                "",
                tokenizer));

        Assert.Contains("Language model path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithNonExistentVisionEncoder_ThrowsFileNotFoundException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();
        var nonExistentPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString(), "vision.onnx");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                nonExistentPath,
                "qformer.onnx",
                "llm.onnx",
                tokenizer));
    }

    #endregion

    #region Native Mode Constructor Tests

    [Fact]
    public void Constructor_Native_WithNonDivisiblePatchSize_ThrowsArgumentException()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new Blip2NeuralNetwork<float>(
                architecture,
                imageSize: 224,
                patchSize: 15)); // 224 not divisible by 15

        Assert.Contains("divisible", exception.Message);
    }

    [Fact]
    public void Constructor_Native_WithValidParameters_SetsProperties()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act
        var network = new Blip2NeuralNetwork<float>(
            architecture,
            numQueryTokens: 32,
            languageModelBackbone: LanguageModelBackbone.OPT);

        // Assert
        Assert.Equal(32, network.NumQueryTokens);
        Assert.Equal(LanguageModelBackbone.OPT, network.LanguageModelBackbone);
    }

    [Fact]
    public void Constructor_Native_WithDefaultParameters_CreatesNetwork()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act
        var network = new Blip2NeuralNetwork<float>(architecture);

        // Assert
        Assert.NotNull(network);
        Assert.Equal(32, network.NumQueryTokens); // Default value
    }

    [Fact]
    public void Constructor_Native_WithCustomImageSize_CreatesNetwork()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act
        var network = new Blip2NeuralNetwork<float>(
            architecture,
            imageSize: 384,
            patchSize: 16); // 384 is divisible by 16

        // Assert
        Assert.NotNull(network);
    }

    [Fact]
    public void Constructor_Native_WithCustomQueryTokens_CreatesNetwork()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act
        var network = new Blip2NeuralNetwork<float>(
            architecture,
            numQueryTokens: 64);

        // Assert
        Assert.Equal(64, network.NumQueryTokens);
    }

    #endregion

    #region Interface Implementation Tests

    [Fact]
    public void Network_ImplementsIBlip2Model()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act
        var network = new Blip2NeuralNetwork<float>(architecture);

        // Assert
        Assert.IsAssignableFrom<Interfaces.IBlip2Model<float>>(network);
    }

    [Fact]
    public void Network_ImplementsIMultimodalEmbedding()
    {
        // Arrange
        var architecture = CreateBasicArchitecture();

        // Act
        var network = new Blip2NeuralNetwork<float>(architecture);

        // Assert
        Assert.IsAssignableFrom<Interfaces.IMultimodalEmbedding<float>>(network);
    }

    #endregion

    #region ModelMetadata Tests

    [Fact]
    public void GetModelMetadata_ReturnsCorrectModelType()
    {
        // Arrange - use small dimensions to avoid OOM during Serialize()
        var architecture = CreateBasicArchitecture();
        var network = CreateSmallBlip2Network(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.Equal(ModelType.Blip2, metadata.ModelType);
    }

    [Fact]
    public void GetModelMetadata_ContainsRequiredInfo()
    {
        // Arrange - use small dimensions to avoid OOM during Serialize()
        var architecture = CreateBasicArchitecture();
        var network = new Blip2NeuralNetwork<float>(
            architecture,
            imageSize: 28,
            patchSize: 14,
            vocabularySize: 64,
            embeddingDimension: 16,
            qformerHiddenDim: 32,
            visionHiddenDim: 32,
            lmHiddenDim: 32,
            numQformerLayers: 1,
            numQueryTokens: 32,
            numHeads: 2,
            numLmDecoderLayers: 1,
            languageModelBackbone: LanguageModelBackbone.FlanT5);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata.AdditionalInfo);
        Assert.Equal(32, metadata.AdditionalInfo["NumQueryTokens"]);
        Assert.Equal("FlanT5", metadata.AdditionalInfo["LanguageModelBackbone"]);
    }

    [Fact]
    public void GetModelMetadata_ContainsQFormerConfiguration()
    {
        // Arrange - use small dimensions to avoid OOM during Serialize()
        var architecture = CreateBasicArchitecture();
        var network = new Blip2NeuralNetwork<float>(
            architecture,
            imageSize: 28,
            patchSize: 14,
            vocabularySize: 64,
            embeddingDimension: 16,
            qformerHiddenDim: 48,
            visionHiddenDim: 32,
            lmHiddenDim: 32,
            numQformerLayers: 2,
            numHeads: 2,
            numLmDecoderLayers: 1);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata.AdditionalInfo);
        Assert.Equal(48, metadata.AdditionalInfo["QFormerHiddenDim"]);
        Assert.Equal(2, metadata.AdditionalInfo["NumQformerLayers"]);
    }

    #endregion

    #region Integration Tests (Skipped without models)

    /// <summary>
    /// Tests requiring actual ONNX model files are skipped.
    /// To run integration tests, provide paths to valid BLIP-2 ONNX models.
    /// </summary>
    [Fact(Skip = "Requires ONNX model files. Set environment variable BLIP2_MODELS_PATH to enable.")]
    public void Integration_WithValidModels_CreatesNetwork()
    {
        // This test would require actual ONNX models
        // Set BLIP2_MODELS_PATH environment variable to enable
    }

    #endregion

    #region Helper Methods

    private static NeuralNetworkArchitecture<float> CreateBasicArchitecture()
    {
        return new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 512,
            outputSize: 512
        );
    }

    private static Blip2NeuralNetwork<float> CreateSmallBlip2Network(
        NeuralNetworkArchitecture<float> architecture)
    {
        return new Blip2NeuralNetwork<float>(
            architecture,
            imageSize: 28,
            patchSize: 14,
            vocabularySize: 64,
            embeddingDimension: 16,
            qformerHiddenDim: 32,
            visionHiddenDim: 32,
            lmHiddenDim: 32,
            numQformerLayers: 1,
            numHeads: 2,
            numLmDecoderLayers: 1);
    }

    #endregion
}
