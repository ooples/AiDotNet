using AiDotNet.Document;
using AiDotNet.Document.Options;
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

    #region PSENet Tests

    [Fact(Timeout = 120000)]
    public async Task PSENet_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new PSENet<double>(arch, options: new AiDotNet.Document.Options.PSENetOptions { ImageSize = 64 });
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PSENet_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new PSENet<double>(arch, options: new AiDotNet.Document.Options.PSENetOptions { ImageSize = 64 });
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task PSENet_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new PSENet<double>(arch, options: new AiDotNet.Document.Options.PSENetOptions { ImageSize = 64 });
        var meta = model.GetModelMetadata();
        Assert.Equal("PSENet", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllTextDetectors_SupportedDocumentTypes_NotNone()
    {
        // Each model owns pooled buffers, so it is declared under its own using scope before the
        // array is built: if a later constructor throws, the array assignment never completes and a
        // finally-based cleanup would never run, leaking every model already constructed.
        using var pseNet = new PSENet<double>(CreateArchitecture(), options: new AiDotNet.Document.Options.PSENetOptions { ImageSize = 64 });

        var models = new DocumentNeuralNetworkBase<double>[] { pseNet };

        foreach (var model in models)
        {
            Assert.NotEqual(DocumentType.None, model.SupportedDocumentTypes);
        }
    }

    #endregion
}
