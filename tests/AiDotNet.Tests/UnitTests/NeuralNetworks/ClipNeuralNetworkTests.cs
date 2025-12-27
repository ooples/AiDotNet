using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for CLIP neural network constructor validation.
/// Note: Full integration tests require ONNX model files which are not included.
/// These tests verify parameter validation and error handling.
/// </summary>
public class ClipNeuralNetworkTests
{
    [Fact]
    public void Constructor_WithNullImageEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ClipNeuralNetwork<float>(architecture, null!, "text_encoder.onnx", tokenizer));

        Assert.Contains("Image encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_WithEmptyImageEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ClipNeuralNetwork<float>(architecture, "", "text_encoder.onnx", tokenizer));

        Assert.Contains("Image encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_WithNullTextEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ClipNeuralNetwork<float>(architecture, "image_encoder.onnx", null!, tokenizer));

        Assert.Contains("Text encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_WithEmptyTextEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new ClipNeuralNetwork<float>(architecture, "image_encoder.onnx", "", tokenizer));

        Assert.Contains("Text encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_WithNonExistentImageEncoder_ThrowsFileNotFoundException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();
        var nonExistentPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString(), "image.onnx");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            new ClipNeuralNetwork<float>(architecture, nonExistentPath, "text.onnx", tokenizer));
    }

    [Fact]
    public void Constructor_WithZeroEmbeddingDimension_PathValidationComesFirst()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Note: Path validation happens before dimension validation
        // So FileNotFoundException is thrown because the file doesn't exist
        Assert.Throws<FileNotFoundException>(() =>
            new ClipNeuralNetwork<float>(
                architecture,
                "image.onnx",
                "text.onnx",
                tokenizer,
                embeddingDimension: 0));
    }

    [Fact]
    public void Constructor_WithNegativeMaxSequenceLength_PathValidationComesFirst()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Note: Path validation happens before this validation
        Assert.Throws<FileNotFoundException>(() =>
            new ClipNeuralNetwork<float>(
                architecture,
                "image.onnx",
                "text.onnx",
                tokenizer,
                maxSequenceLength: -1));
    }

    [Fact]
    public void Constructor_WithNegativeImageSize_PathValidationComesFirst()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Note: Path validation happens before this validation
        Assert.Throws<FileNotFoundException>(() =>
            new ClipNeuralNetwork<float>(
                architecture,
                "image.onnx",
                "text.onnx",
                tokenizer,
                imageSize: -1));
    }

    /// <summary>
    /// Tests requiring actual ONNX model files are skipped.
    /// To run integration tests, provide paths to valid CLIP ONNX models.
    /// </summary>
    [Fact(Skip = "Requires ONNX model files. Set environment variable CLIP_MODELS_PATH to enable.")]
    public void Integration_WithValidModels_CreatesNetwork()
    {
        // This test would require actual ONNX models
        // Set CLIP_MODELS_PATH environment variable to enable
    }

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

    #endregion
}
