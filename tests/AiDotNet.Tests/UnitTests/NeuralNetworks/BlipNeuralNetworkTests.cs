using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for BLIP neural network constructor validation.
/// Note: Full integration tests require ONNX model files which are not included.
/// These tests verify parameter validation and error handling.
/// </summary>
public class BlipNeuralNetworkTests
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
            new BlipNeuralNetwork<float>(architecture, null!, "text_encoder.onnx", "decoder.onnx", tokenizer));

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
            new BlipNeuralNetwork<float>(architecture, "", "text_encoder.onnx", "decoder.onnx", tokenizer));

        Assert.Contains("Vision encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithNullTextEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new BlipNeuralNetwork<float>(architecture, "vision_encoder.onnx", null!, "decoder.onnx", tokenizer));

        Assert.Contains("Text encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithEmptyTextEncoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new BlipNeuralNetwork<float>(architecture, "vision_encoder.onnx", "", "decoder.onnx", tokenizer));

        Assert.Contains("Text encoder path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithNullDecoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new BlipNeuralNetwork<float>(architecture, "vision_encoder.onnx", "text_encoder.onnx", null!, tokenizer));

        Assert.Contains("Text decoder path", exception.Message);
    }

    [Fact]
    public void Constructor_Onnx_WithEmptyDecoderPath_ThrowsArgumentException()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var architecture = CreateBasicArchitecture();

        // Act & Assert
        var exception = Assert.Throws<ArgumentException>(() =>
            new BlipNeuralNetwork<float>(architecture, "vision_encoder.onnx", "text_encoder.onnx", "", tokenizer));

        Assert.Contains("Text decoder path", exception.Message);
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
            new BlipNeuralNetwork<float>(architecture, nonExistentPath, "text.onnx", "decoder.onnx", tokenizer));
    }

    #endregion

    #region Integration Tests (Skipped without models)

    /// <summary>
    /// Tests requiring actual ONNX model files are skipped.
    /// To run integration tests, provide paths to valid BLIP ONNX models.
    /// </summary>
    [Fact(Skip = "Requires ONNX model files. Set environment variable BLIP_MODELS_PATH to enable.")]
    public void Integration_WithValidModels_CreatesNetwork()
    {
        // This test would require actual ONNX models
        // Set BLIP_MODELS_PATH environment variable to enable
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

    #endregion
}
