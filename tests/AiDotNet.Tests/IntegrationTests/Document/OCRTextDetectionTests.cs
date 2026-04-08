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
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 2);
    }

    private static Tensor<double> CreateSmallImage(int channels = 3, int size = 64)
    {
        int totalSize = 1 * channels * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, channels, size, size }, data);
    }

    #region CRAFT Tests

    [Fact(Timeout = 120000)]
    public async Task CRAFT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task CRAFT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<double>(arch, imageSize: 64);
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
        var model = new CRAFT<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("CRAFT", meta.Name);
        Assert.NotNull(meta.AdditionalInfo);
    }

    [Fact(Timeout = 120000)]
    public async Task CRAFT_GetModelSummary_ContainsModelName()
    {
        var arch = CreateArchitecture();
        var model = new CRAFT<double>(arch, imageSize: 64);
        var summary = model.GetModelSummary();
        Assert.Contains("CRAFT", summary);
    }

    #endregion

    #region DBNet Tests

    [Fact(Timeout = 120000)]
    public async Task DBNet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DBNet<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DBNet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DBNet<double>(arch, imageSize: 64);
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
        var model = new DBNet<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DBNet", meta.Name);
    }

    #endregion

    #region EAST Tests

    [Fact(Timeout = 120000)]
    public async Task EAST_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new EAST<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task EAST_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new EAST<double>(arch, imageSize: 64);
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
        var model = new EAST<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("EAST", meta.Name);
    }

    #endregion

    #region PSENet Tests

    [Fact(Timeout = 120000)]
    public async Task PSENet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new PSENet<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PSENet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new PSENet<double>(arch, imageSize: 64);
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
        var model = new PSENet<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("PSENet", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllTextDetectors_SupportedDocumentTypes_NotNone()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new CRAFT<double>(arch, imageSize: 64),
            new DBNet<double>(arch, imageSize: 64),
            new EAST<double>(arch, imageSize: 64),
            new PSENet<double>(arch, imageSize: 64),
        };

        foreach (var model in models)
        {
            Assert.NotEqual(DocumentType.None, model.SupportedDocumentTypes);
        }
    }

    #endregion
}
