using AiDotNet.Document;
using AiDotNet.Document.LayoutAware;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for layout-aware document models.
/// </summary>
public class LayoutAwareDocumentTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 7);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    #region LayoutLM Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLM", meta.Name);
    }

    #endregion

    #region LayoutLMv2 Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv2", meta.Name);
    }

    #endregion

    #region LayoutLMv3 Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv3", meta.Name);
    }

    #endregion

    #region LayoutXLM Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutXLM", meta.Name);
    }

    #endregion

    #region DocFormer Tests

    [Fact(Timeout = 120000)]
    public async Task DocFormer_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DocFormer_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task DocFormer_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocFormer", meta.Name);
    }

    #endregion

    #region DiT Tests

    [Fact(Timeout = 120000)]
    public async Task DiT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DiT", meta.Name);
    }

    #endregion

    #region LiLT Tests

    [Fact(Timeout = 120000)]
    public async Task LiLT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LiLT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LiLT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LiLT", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllLayoutAwareModels_RequiresOCR_IsTrue()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new LayoutLM<double>(arch),
            new LayoutLMv2<double>(arch),
            new LayoutLMv3<double>(arch),
            new LayoutXLM<double>(arch),
            new DocFormer<double>(arch),
            new LiLT<double>(arch),
        };

        foreach (var model in models)
        {
            // Layout-aware models require OCR to provide text and bounding boxes
            Assert.True(model.RequiresOCR);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_RequiresOCR_IsFalse()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        // DiT is vision-only, does not require OCR
        Assert.False(model.RequiresOCR);
    }

    #endregion
}
