using AiDotNet.Document;
using AiDotNet.Document.OCR.TextRecognition;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

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

    private static TrOCR<double> CreateSmallTrOCR(NeuralNetworkArchitecture<double> architecture)
    {
        // Keep the production constructor's TrOCR-base paper defaults intact. These integration
        // invariants exercise native construction, prediction, metadata, and OCR capabilities, so
        // use the model's public topology controls to avoid allocating the full 50,265 x 768
        // double-precision embedding table repeatedly on a 16 GiB CI runner.
        return new TrOCR<double>(
            architecture,
            imageHeight: 32,
            imageWidth: 128,
            maxSequenceLength: 16,
            encoderHiddenDim: 32,
            decoderHiddenDim: 32,
            numEncoderLayers: 1,
            numDecoderLayers: 1,
            numEncoderHeads: 4,
            numDecoderHeads: 4,
            patchSize: 16,
            vocabSize: 128);
    }

    #region CRNN Tests

    [Fact(Timeout = 120000)]
    public async Task CRNN_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new CRNN<double>(arch, imageWidth: 128);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task CRNN_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new CRNN<double>(arch, imageWidth: 128);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task CRNN_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new CRNN<double>(arch, imageWidth: 128);
        var meta = model.GetModelMetadata();
        Assert.Equal("CRNN", meta.Name);
    }

    #endregion

    #region TrOCR Tests

    [Fact(Timeout = 120000)]
    public async Task TrOCR_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = CreateSmallTrOCR(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task TrOCR_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = CreateSmallTrOCR(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task TrOCR_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = CreateSmallTrOCR(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("TrOCR", meta.Name);
    }

    #endregion

    #region SVTR Tests

    [Fact(Timeout = 120000)]
    public async Task SVTR_NativeConstruction_Succeeds()
    {
        await Task.Yield();
        var arch = CreateArchitecture();
        using var model = new SVTR<double>(arch);
        Assert.True(model.ParameterCount > 0,
            "SVTR construction must create a trainable architecture.");
    }

    [Fact(Timeout = 120000)]
    public async Task SVTR_Predict_ReturnsOutput()
    {
        await Task.Yield();
        var arch = CreateArchitecture();
        using var model = new SVTR<double>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task SVTR_GetModelMetadata_ReturnsValidData()
    {
        await Task.Yield();
        var arch = CreateArchitecture();
        using var model = new SVTR<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("SVTR", meta.Name);
    }

    #endregion

    #region ABINet Tests

    [Fact(Timeout = 120000)]
    public async Task ABINet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new ABINet<double>(arch, imageWidth: 128, imageHeight: 32);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task ABINet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new ABINet<double>(arch, imageWidth: 128, imageHeight: 32);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task ABINet_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new ABINet<double>(arch, imageWidth: 128, imageHeight: 32);
        var meta = model.GetModelMetadata();
        Assert.Equal("ABINet", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllTextRecognizers_RequiresOCR_IsFalse()
    {
        await Task.Yield();
        // Each model owns pooled buffers, so it is declared under its own using scope before the
        // array is built: if a later constructor throws, the array assignment never completes and a
        // finally-based cleanup would never run, leaking every model already constructed.
        using var crnn = new CRNN<double>(CreateArchitecture(), imageWidth: 128);
        using var trOcr = CreateSmallTrOCR(CreateArchitecture());
        using var svtr = new SVTR<double>(CreateArchitecture());
        using var abiNet = new ABINet<double>(CreateArchitecture(), imageWidth: 128, imageHeight: 32);

        var models = new DocumentNeuralNetworkBase<double>[] { crnn, trOcr, svtr, abiNet };

        foreach (var model in models)
        {
            // Text recognizers are OCR components themselves, they don't require OCR
            Assert.False(model.RequiresOCR);
        }
    }

    #endregion
}
