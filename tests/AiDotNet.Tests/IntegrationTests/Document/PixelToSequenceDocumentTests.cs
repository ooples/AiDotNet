using AiDotNet.Document;
using AiDotNet.Document.PixelToSequence;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for pixel-to-sequence document models.
/// </summary>
public class PixelToSequenceDocumentTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 100);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    #region Donut Tests

    [Fact]
    public void Donut_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new Donut<double>(arch, imageHeight: 64, imageWidth: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void Donut_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new Donut<double>(arch, imageHeight: 64, imageWidth: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void Donut_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new Donut<double>(arch, imageHeight: 64, imageWidth: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("Donut", meta.Name);
    }

    #endregion

    #region Nougat Tests

    [Fact]
    public void Nougat_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new Nougat<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void Nougat_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new Nougat<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void Nougat_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new Nougat<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("Nougat", meta.Name);
    }

    #endregion

    #region Pix2Struct Tests

    [Fact]
    public void Pix2Struct_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new Pix2Struct<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void Pix2Struct_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new Pix2Struct<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void Pix2Struct_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new Pix2Struct<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("Pix2Struct", meta.Name);
    }

    #endregion

    #region Dessurt Tests

    [Fact]
    public void Dessurt_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new Dessurt<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void Dessurt_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new Dessurt<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void Dessurt_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new Dessurt<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("Dessurt", meta.Name);
    }

    #endregion

    #region MATCHA Tests

    [Fact]
    public void MATCHA_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new MATCHA<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void MATCHA_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new MATCHA<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void MATCHA_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new MATCHA<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("MATCHA", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllPixelToSequenceModels_RequiresOCR_IsFalse()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new Donut<double>(arch, imageHeight: 64, imageWidth: 64),
            new Nougat<double>(arch, imageSize: 64),
            new Pix2Struct<double>(arch, imageSize: 64),
            new Dessurt<double>(arch, imageSize: 64),
            new MATCHA<double>(arch, imageSize: 64),
        };

        foreach (var model in models)
        {
            // Pixel-to-sequence models process raw pixels, no OCR required
            Assert.False(model.RequiresOCR);
        }
    }

    #endregion
}
