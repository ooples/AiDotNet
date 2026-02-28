using AiDotNet.Document;
using AiDotNet.Document.LayoutAware;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

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

    [Fact]
    public void LayoutLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void LayoutLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void LayoutLM_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLM", meta.Name);
    }

    #endregion

    #region LayoutLMv2 Tests

    [Fact]
    public void LayoutLMv2_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void LayoutLMv2_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void LayoutLMv2_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv2", meta.Name);
    }

    #endregion

    #region LayoutLMv3 Tests

    [Fact]
    public void LayoutLMv3_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void LayoutLMv3_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void LayoutLMv3_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv3", meta.Name);
    }

    #endregion

    #region LayoutXLM Tests

    [Fact]
    public void LayoutXLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void LayoutXLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void LayoutXLM_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutXLM", meta.Name);
    }

    #endregion

    #region DocFormer Tests

    [Fact]
    public void DocFormer_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void DocFormer_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void DocFormer_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocFormer", meta.Name);
    }

    #endregion

    #region DiT Tests

    [Fact]
    public void DiT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void DiT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void DiT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DiT", meta.Name);
    }

    #endregion

    #region LiLT Tests

    [Fact]
    public void LiLT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void LiLT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void LiLT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LiLT", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllLayoutAwareModels_RequiresOCR_IsTrue()
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

    [Fact]
    public void DiT_RequiresOCR_IsFalse()
    {
        var arch = CreateArchitecture();
        var model = new DiT<double>(arch, imageSize: 64);
        // DiT is vision-only, does not require OCR
        Assert.False(model.RequiresOCR);
    }

    #endregion
}
