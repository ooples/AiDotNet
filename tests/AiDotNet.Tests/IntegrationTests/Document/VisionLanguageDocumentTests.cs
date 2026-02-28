using AiDotNet.Document;
using AiDotNet.Document.VisionLanguage;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for vision-language document models.
/// </summary>
public class VisionLanguageDocumentTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 16);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    #region DocOwl Tests

    [Fact]
    public void DocOwl_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void DocOwl_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void DocOwl_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocOwl", meta.Name);
    }

    #endregion

    #region InfographicVQA Tests

    [Fact]
    public void InfographicVQA_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void InfographicVQA_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void InfographicVQA_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("InfographicVQA", meta.Name);
    }

    #endregion

    #region UDOP Tests

    [Fact]
    public void UDOP_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact]
    public void UDOP_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void UDOP_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("UDOP", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllVisionLanguageModels_SupportsTraining_InNativeMode()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new DocOwl<double>(arch, imageSize: 64),
            new InfographicVQA<double>(arch, imageSize: 64),
            new UDOP<double>(arch, imageSize: 64),
        };

        foreach (var model in models)
        {
            Assert.True(model.SupportsTraining);
        }
    }

    #endregion
}
