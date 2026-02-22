using AiDotNet.Document;
using AiDotNet.Document.OCR.TextRecognition;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for OCR text recognition models.
/// </summary>
public class OCRTextRecognitionTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32,
            inputWidth: 128,
            inputDepth: 3,
            outputSize: 62);
    }

    private static Tensor<double> CreateSmallImage(int height = 32, int width = 128)
    {
        int totalSize = 1 * 3 * height * width;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, height, width }, data);
    }

    #region CRNN Tests

    [Fact]
    public void CRNN_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new CRNN<double>(arch, imageWidth: 128);
        Assert.NotNull(model);
    }

    [Fact]
    public void CRNN_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new CRNN<double>(arch, imageWidth: 128);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void CRNN_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new CRNN<double>(arch, imageWidth: 128);
        var meta = model.GetModelMetadata();
        Assert.Equal("CRNN", meta.Name);
    }

    #endregion

    #region TrOCR Tests

    [Fact]
    public void TrOCR_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new TrOCR<double>(arch, imageHeight: 32, imageWidth: 128);
        Assert.NotNull(model);
    }

    [Fact]
    public void TrOCR_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new TrOCR<double>(arch, imageHeight: 32, imageWidth: 128);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void TrOCR_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new TrOCR<double>(arch, imageHeight: 32, imageWidth: 128);
        var meta = model.GetModelMetadata();
        Assert.Equal("TrOCR", meta.Name);
    }

    #endregion

    #region SVTR Tests

    [Fact]
    public void SVTR_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new SVTR<double>(arch, imageWidth: 128, imageHeight: 32);
        Assert.NotNull(model);
    }

    [Fact]
    public void SVTR_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new SVTR<double>(arch, imageWidth: 128, imageHeight: 32);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void SVTR_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new SVTR<double>(arch, imageWidth: 128, imageHeight: 32);
        var meta = model.GetModelMetadata();
        Assert.Equal("SVTR", meta.Name);
    }

    #endregion

    #region ABINet Tests

    [Fact]
    public void ABINet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new ABINet<double>(arch, imageWidth: 128, imageHeight: 32);
        Assert.NotNull(model);
    }

    [Fact]
    public void ABINet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new ABINet<double>(arch, imageWidth: 128, imageHeight: 32);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void ABINet_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new ABINet<double>(arch, imageWidth: 128, imageHeight: 32);
        var meta = model.GetModelMetadata();
        Assert.Equal("ABINet", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllTextRecognizers_RequiresOCR_IsFalse()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new CRNN<double>(arch, imageWidth: 128),
            new TrOCR<double>(arch, imageHeight: 32, imageWidth: 128),
            new SVTR<double>(arch, imageWidth: 128, imageHeight: 32),
            new ABINet<double>(arch, imageWidth: 128, imageHeight: 32),
        };

        foreach (var model in models)
        {
            // Text recognizers are OCR components themselves, they don't require OCR
            Assert.False(model.RequiresOCR);
        }
    }

    #endregion
}
