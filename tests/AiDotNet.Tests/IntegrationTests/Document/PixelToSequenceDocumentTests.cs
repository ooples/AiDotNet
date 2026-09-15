using AiDotNet.Document;
using AiDotNet.Document.PixelToSequence;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for pixel-to-sequence document models.
/// </summary>
/// <remarks>
/// <para>
/// Every model is built at a TEST SCALE: the paper's layer types, encoder/decoder split and patch
/// geometry, with small widths, depths and vocabularies. These tests check construction, the
/// forward contract and metadata; none of them depends on paper-scale capacity.
/// </para>
/// <para>
/// The previous fixtures shrank only the image and kept every other paper default, in double
/// precision: MATCHA at 1536-wide 18+18 layers with a 50,265-token vocabulary is ~1.3 B parameters
/// (~10.7 GB of weights), Dessurt ~0.5 B (~3.9 GB). Their Predict tests pushed the xUnit host to a
/// measured 26.35 GB peak on the Integration D shard, and on the 16 GB GitHub runner the job was
/// killed ("The runner has received a shutdown signal") while MATCHA_Predict was running
/// (Build &amp; SonarCloud run 34561803988, job 103164792718).
/// </para>
/// </remarks>
public class PixelToSequenceDocumentTests
{
    private const int ImageSize = 64;

    // Shared test-scale geometry. Heads divide every width; the vocabulary covers the token range the
    // models' text decoders map to characters (ids up to 214).
    private const int TestWidth = 64;
    private const int TestHeads = 4;
    private const int TestLayers = 2;
    private const int TestVocab = 256;
    private const int TestSequence = 32;

    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = ImageSize)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 100);
    }

    private static Tensor<double> CreateSmallImage(int size = ImageSize)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    // Swin encoder: exactly four stages (LayerHelper enforces it) with a patch merge between each, so
    // widths run 32/64/128/256 and the 64x64 image's 16x16 patch grid runs 16/8/4/2 - a window of 2
    // tiles every stage. The decoder reads the final 256-wide (embedDim * 8) encoder output.
    private static Donut<double> CreateDonut() => new(
        CreateArchitecture(), imageHeight: ImageSize, imageWidth: ImageSize,
        maxGenerationLength: TestSequence, embedDim: 32, depths: new[] { 1, 1, 1, 1 },
        numHeads: new[] { 1, 2, 4, 8 }, windowSize: 2, patchSize: 4, decoderHiddenDim: TestWidth,
        numDecoderLayers: TestLayers, decoderHeads: TestHeads, vocabSize: TestVocab);

    private static Nougat<double> CreateNougat() => new(
        CreateArchitecture(), imageSize: ImageSize, patchSize: 16, maxSequenceLength: TestSequence,
        hiddenDim: TestWidth, numEncoderLayers: TestLayers, numDecoderLayers: TestLayers,
        numHeads: TestHeads, vocabSize: TestVocab);

    private static Pix2Struct<double> CreatePix2Struct() => new(
        CreateArchitecture(), imageSize: ImageSize, patchSize: 16, maxPatches: 64,
        maxSequenceLength: TestSequence, hiddenDim: TestWidth, numEncoderLayers: TestLayers,
        numDecoderLayers: TestLayers, numHeads: TestHeads, vocabSize: TestVocab);

    private static Dessurt<double> CreateDessurt() => new(
        CreateArchitecture(), imageSize: ImageSize, maxSequenceLength: TestSequence,
        encoderDim: TestWidth, decoderDim: TestWidth, encoderLayers: TestLayers,
        decoderLayers: TestLayers, numHeads: TestHeads, vocabSize: TestVocab);

    private static MATCHA<double> CreateMatcha() => new(
        CreateArchitecture(), imageSize: ImageSize, maxSequenceLength: TestSequence,
        encoderDim: TestWidth, decoderDim: TestWidth, encoderLayers: TestLayers,
        decoderLayers: TestLayers, numHeads: TestHeads, vocabSize: TestVocab, maxPatchesPerImage: 64);

    private static void AssertPredictReturnsOutput(DocumentNeuralNetworkBase<double> model)
    {
        var output = model.Predict(CreateSmallImage());
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    #region Donut Tests

    [Fact(Timeout = 120000)]
    public async Task Donut_NativeConstruction_Succeeds()
    {
        Assert.NotNull(CreateDonut());
    }

    [Fact(Timeout = 120000)]
    public async Task Donut_Predict_ReturnsOutput()
    {
        AssertPredictReturnsOutput(CreateDonut());
    }

    [Fact(Timeout = 120000)]
    public async Task Donut_GetModelMetadata_ReturnsValidData()
    {
        Assert.Equal("Donut", CreateDonut().GetModelMetadata().Name);
    }

    #endregion

    #region Nougat Tests

    [Fact(Timeout = 120000)]
    public async Task Nougat_NativeConstruction_Succeeds()
    {
        Assert.NotNull(CreateNougat());
    }

    [Fact(Timeout = 120000)]
    public async Task Nougat_Predict_ReturnsOutput()
    {
        AssertPredictReturnsOutput(CreateNougat());
    }

    [Fact(Timeout = 120000)]
    public async Task Nougat_GetModelMetadata_ReturnsValidData()
    {
        Assert.Equal("Nougat", CreateNougat().GetModelMetadata().Name);
    }

    #endregion

    #region Pix2Struct Tests

    [Fact(Timeout = 120000)]
    public async Task Pix2Struct_NativeConstruction_Succeeds()
    {
        Assert.NotNull(CreatePix2Struct());
    }

    [Fact(Timeout = 120000)]
    public async Task Pix2Struct_Predict_ReturnsOutput()
    {
        AssertPredictReturnsOutput(CreatePix2Struct());
    }

    [Fact(Timeout = 120000)]
    public async Task Pix2Struct_GetModelMetadata_ReturnsValidData()
    {
        Assert.Equal("Pix2Struct", CreatePix2Struct().GetModelMetadata().Name);
    }

    #endregion

    #region Dessurt Tests

    [Fact(Timeout = 120000)]
    public async Task Dessurt_NativeConstruction_Succeeds()
    {
        Assert.NotNull(CreateDessurt());
    }

    [Fact(Timeout = 120000)]
    public async Task Dessurt_Predict_ReturnsOutput()
    {
        AssertPredictReturnsOutput(CreateDessurt());
    }

    [Fact(Timeout = 120000)]
    public async Task Dessurt_GetModelMetadata_ReturnsValidData()
    {
        Assert.Equal("Dessurt", CreateDessurt().GetModelMetadata().Name);
    }

    #endregion

    #region MATCHA Tests

    [Fact(Timeout = 120000)]
    public async Task MATCHA_NativeConstruction_Succeeds()
    {
        Assert.NotNull(CreateMatcha());
    }

    [Fact(Timeout = 120000)]
    public async Task MATCHA_Predict_ReturnsOutput()
    {
        AssertPredictReturnsOutput(CreateMatcha());
    }

    [Fact(Timeout = 120000)]
    public async Task MATCHA_GetModelMetadata_ReturnsValidData()
    {
        Assert.Equal("MATCHA", CreateMatcha().GetModelMetadata().Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllPixelToSequenceModels_RequiresOCR_IsFalse()
    {
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            CreateDonut(),
            CreateNougat(),
            CreatePix2Struct(),
            CreateDessurt(),
            CreateMatcha(),
        };

        foreach (var model in models)
        {
            // Pixel-to-sequence models process raw pixels, no OCR required
            Assert.False(model.RequiresOCR);
        }
    }

    #endregion
}
