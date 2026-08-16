using AiDotNet.Document;
using AiDotNet.Document.VisionLanguage;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task DocOwl_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DocOwl_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    private static DocOwl<double> CreateSmallDocOwl()
        => new DocOwl<double>(CreateArchitecture(imageSize: 64), imageSize: 64,
            maxSequenceLength: 64, visionDim: 32, languageDim: 32,
            visionLayers: 1, languageLayers: 1, numHeads: 4, vocabSize: 64);

    private static Tensor<double> CreateTokenIds(int count, int offset = 0, int vocab = 64)
    {
        var data = new Vector<double>(count);
        for (int i = 0; i < count; i++) data[i] = (i + offset) % vocab;
        return new Tensor<double>(new[] { count }, data);
    }

    /// <summary>
    /// DocOwl's text decoder had no token embedding, so only projected visual tokens ever reached it
    /// and _languageEmbeddings sat dead. Different text with the same image must now move the output.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DocOwl_TextTokens_ChangeTheOutput()
    {
        await Task.Yield();
        var model = CreateSmallDocOwl();
        model.SetTrainingMode(false);
        var image = CreateSmallImage(64);

        var imageOnly = model.Predict(image);
        // Genuinely different sequences. An earlier version varied only the vocab bound, which for
        // six tokens produced the SAME ids both times and made the assertion unfalsifiable.
        var withTextA = model.Predict(image, CreateTokenIds(6, offset: 0));
        var withTextB = model.Predict(image, CreateTokenIds(6, offset: 17));

        Assert.NotNull(imageOnly);
        Assert.True(Differs(withTextA, withTextB),
            "Two different token sequences over the same image gave identical output, so the text " +
            "tokens are not reaching DocOwl's decoder.");
    }

    /// <summary>
    /// The point of routing the second modality through the ordinary forward rather than a
    /// NoGradScope side door: training must actually see it. Every per-model EncodeMultimodal in
    /// this family opens a NoGradScope, which is why LiLT's layout stream could never be trained.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DocOwl_TrainingWithText_ChangesParameters()
    {
        await Task.Yield();
        var model = CreateSmallDocOwl();
        var image = CreateSmallImage(64);
        var tokens = CreateTokenIds(6);

        model.SetTrainingMode(true);

        // Run one forward BEFORE snapshotting. The token table allocates lazily, so a count taken
        // first is short by its size and the comparison below would report a length change rather
        // than the weight movement it is actually testing.
        var target = model.Predict(image, tokens);
        var before = model.GetParameters();
        var shape = new int[target.Rank];
        for (int i = 0; i < target.Rank; i++) shape[i] = target.Shape[i];

        var shifted = new Tensor<double>(shape);
        for (int i = 0; i < shifted.Length; i++)
            shifted.Data.Span[i] = target.Data.Span[i] + 0.5;

        model.Train(image, tokens, shifted);

        var after = model.GetParameters();
        Assert.Equal(before.Length, after.Length);

        bool moved = false;
        for (int i = 0; i < before.Length && !moved; i++)
        {
            if (System.Math.Abs(before[i] - after[i]) > 1e-12) moved = true;
        }

        Assert.True(moved,
            "Training through the auxiliary-input overload left every parameter untouched, so the " +
            "text path is not on the gradient tape.");
    }

    private static bool Differs(Tensor<double> a, Tensor<double> b)
    {
        if (a.Length != b.Length) return true;
        for (int i = 0; i < a.Length; i++)
        {
            if (System.Math.Abs(a.Data.Span[i] - b.Data.Span[i]) > 1e-12) return true;
        }

        return false;
    }

    [Fact(Timeout = 120000)]
    public async Task DocOwl_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocOwl", meta.Name);
    }

    #endregion

    #region InfographicVQA Tests

    [Fact(Timeout = 120000)]
    public async Task InfographicVQA_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task InfographicVQA_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task InfographicVQA_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("InfographicVQA", meta.Name);
    }

    #endregion

    #region UDOP Tests

    [Fact(Timeout = 120000)]
    public async Task UDOP_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UDOP_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task UDOP_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("UDOP", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllVisionLanguageModels_SupportsTraining_InNativeMode()
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
