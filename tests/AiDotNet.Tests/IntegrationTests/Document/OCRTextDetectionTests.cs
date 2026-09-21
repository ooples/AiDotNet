using AiDotNet.Document;
using AiDotNet.Document.OCR.TextDetection;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for OCR text detection models.
/// </summary>
public class OCRTextDetectionTests
{
    private static NeuralNetworkArchitecture<float> CreateArchitecture(int imageSize = DocumentTestScale.ImageSize)
    {
        return new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 2);
    }

    private static Tensor<float> CreateSmallImage(int channels = 3, int size = 64)
    {
        int totalSize = 1 * channels * size * size;
        var data = new Vector<float>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5f;
        return new Tensor<float>(new[] { 1, channels, size, size }, data);
    }

    #region CRAFT Tests

    [Fact(Timeout = 120000)]
    public async Task CRAFT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, upscaleChannels: DocumentTestScale.UpscaleChannels);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task CRAFT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, upscaleChannels: DocumentTestScale.UpscaleChannels);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task CRAFT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, upscaleChannels: DocumentTestScale.UpscaleChannels);
        var meta = model.GetModelMetadata();
        Assert.Equal("CRAFT", meta.Name);
        Assert.NotNull(meta.AdditionalInfo);
    }

    [Fact(Timeout = 120000)]
    public async Task CRAFT_GetModelSummary_ContainsModelName()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, upscaleChannels: DocumentTestScale.UpscaleChannels);
        var summary = model.GetModelSummary();
        Assert.Contains("CRAFT", summary);
    }

    #endregion

    #region DBNet Tests

    [Fact(Timeout = 120000)]
    public async Task DBNet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DBNet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DBNet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DBNet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task DBNet_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DBNet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels);
        var meta = model.GetModelMetadata();
        Assert.Equal("DBNet", meta.Name);
    }

    #endregion

    #region EAST Tests

    [Fact(Timeout = 120000)]
    public async Task EAST_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new EAST<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task EAST_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new EAST<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task EAST_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new EAST<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels);
        var meta = model.GetModelMetadata();
        Assert.Equal("EAST", meta.Name);
    }

    #endregion

    #region PSENet Tests

    [Fact(Timeout = 120000)]
    public async Task PSENet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new PSENet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PSENet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new PSENet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task PSENet_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new PSENet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels);
        var meta = model.GetModelMetadata();
        Assert.Equal("PSENet", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllTextDetectors_SupportedDocumentTypes_NotNone()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<float>[]
        {
            new CRAFT<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, upscaleChannels: DocumentTestScale.UpscaleChannels),
            new DBNet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels),
            new EAST<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels),
            new PSENet<float>(arch, imageSize: DocumentTestScale.ImageSize, backboneChannels: DocumentTestScale.BackboneChannels, featureChannels: DocumentTestScale.FeatureChannels),
        };

        foreach (var model in models)
        {
            Assert.NotEqual(DocumentType.None, model.SupportedDocumentTypes);
        }
    }

    #endregion
}
