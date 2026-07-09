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
    private static NeuralNetworkArchitecture<float> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 7);
    }

    private static Tensor<float> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<float>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5f;
        return new Tensor<float>(new[] { 1, 3, size, size }, data);
    }

    #region LayoutLM Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLM<float>(arch);
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
        var model = new LayoutLM<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLM", meta.Name);
    }

    #endregion

    #region LayoutLMv2 Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv2<float>(arch);
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
        var model = new LayoutLMv2<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv2", meta.Name);
    }

    #endregion

    #region LayoutLMv3 Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutLMv3<float>(arch);
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
        var model = new LayoutLMv3<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv3", meta.Name);
    }

    #endregion

    #region LayoutXLM Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutXLM<float>(arch);
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
        var model = new LayoutXLM<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutXLM", meta.Name);
    }

    #endregion

    #region DocFormer Tests

    [Fact(Timeout = 120000)]
    public async Task DocFormer_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DocFormer_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocFormer<float>(arch);
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
        var model = new DocFormer<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocFormer", meta.Name);
    }

    #endregion

    #region DiT Tests

    [Fact(Timeout = 120000)]
    public async Task DiT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DiT<float>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DiT<float>(arch, imageSize: 64);
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
        var model = new DiT<float>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DiT", meta.Name);
    }

    #endregion

    #region LiLT Tests

    [Fact(Timeout = 120000)]
    public async Task LiLT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LiLT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LiLT<float>(arch);
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
        var model = new LiLT<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LiLT", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllLayoutAwareModels_RequiresOCR_IsTrue()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<float>[]
        {
            new LayoutLM<float>(arch),
            new LayoutLMv2<float>(arch),
            new LayoutLMv3<float>(arch),
            new LayoutXLM<float>(arch),
            new DocFormer<float>(arch),
            new LiLT<float>(arch),
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
        var model = new DiT<float>(arch, imageSize: 64);
        // DiT is vision-only, does not require OCR
        Assert.False(model.RequiresOCR);
    }

    #endregion

    #region LiLT BiACM (paper-faithful dual-stream)
    // LiLT (Wang et al. 2022) couples a text stream and a layout stream through BiACM: the two streams
    // SHARE attention scores (text += layout; layout += detach(text)). These tests prove the layout
    // stream actually influences the text-stream output (coupling is live) and that both streams stay
    // finite, with token-aligned integer token IDs so the [seq, seq] score matrices are addable.

    private static Tensor<float> CreateIntTokenVector(int count, int vocab = 50)
    {
        var data = new Vector<float>(count);
        for (int i = 0; i < count; i++) data[i] = i % vocab;
        return new Tensor<float>(new[] { count }, data);
    }

    private static Tensor<float> CreateBoxFeatures(int count, int boxDim = 6)
    {
        var t = new Tensor<float>(new[] { count, boxDim });
        for (int i = 0; i < t.Length; i++) t[i] = 0.1f * ((i % 7) + 1);
        return t;
    }

    private static LiLT<float> CreateSmallLiLT()
        => new LiLT<float>(CreateArchitecture(imageSize: 32), numClasses: 4, maxSequenceLength: 64,
            hiddenDim: 64, numLayers: 2, numHeads: 4, vocabSize: 100);

    [Fact(Timeout = 120000)]
    public async Task LiLT_BiACM_LayoutStreamInfluencesTextOutput()
    {
        await Task.Yield();
        var model = CreateSmallLiLT();
        model.SetTrainingMode(false);
        var tokens = CreateIntTokenVector(8);

        var textOnly = model.EncodeDualStream(tokens, layoutBoxes: null);
        var fused = model.EncodeDualStream(tokens, CreateBoxFeatures(8));

        // Same shape, both finite.
        Assert.Equal(textOnly.Length, fused.Length);
        var f = fused.ToArray();
        for (int i = 0; i < f.Length; i++)
            Assert.True(!float.IsNaN(f[i]) && !float.IsInfinity(f[i]), $"BiACM output[{i}] = {f[i]} not finite.");

        // BiACM coupling must be LIVE: adding the layout stream changes the text-stream output.
        double l2 = 0;
        var t = textOnly.ToArray();
        for (int i = 0; i < f.Length; i++) { double d = f[i] - t[i]; l2 += d * d; }
        Assert.True(System.Math.Sqrt(l2) > 1e-6,
            "Layout stream did not influence the text output — BiACM score sharing is not active.");
    }

    [Fact(Timeout = 120000)]
    public async Task LiLT_TextOnly_IsFiniteAndDegradesGracefully()
    {
        await Task.Yield();
        var model = CreateSmallLiLT();
        model.SetTrainingMode(false);

        var output = model.EncodeDualStream(CreateIntTokenVector(8), layoutBoxes: null);
        Assert.True(output.Length > 0);
        var d = output.ToArray();
        for (int i = 0; i < d.Length; i++)
            Assert.True(!float.IsNaN(d[i]) && !float.IsInfinity(d[i]), $"text-only output[{i}] not finite.");
    }

    #endregion

    #region Multimodal Fusion Regression
    // The LayoutLMv2 / LayoutXLM two-stream fusion (Xu et al. 2021, §3.1) must join the visual
    // token sequence and the text token sequence along the SEQUENCE axis, preserving the batch
    // dimension. The prior implementation concatenated on axis 0 with unequal-rank streams — the
    // visual backbone emits [B, Lvis, D] while the text embedding emits [Ltext, D] — which spuriously
    // grew the BATCH dimension ([B+Ltext, Lvis, D]) and left an uninitialized tail in the rented
    // output buffer. That tail was harmless only when the tensor pool happened to be clean; after a
    // sibling training step dirtied the pool it surfaced as intermittent NaN. These deterministic
    // tests catch both defects without needing pool contamination: the batch dimension must stay 1
    // and the fused sequence length must scale with the number of text tokens.

    private static Tensor<float> CreateTokenIds(int count, int vocab = 100)
    {
        var data = new Vector<float>(count);
        for (int i = 0; i < count; i++) data[i] = i % vocab;   // integer IDs -> embedding lookup
        return new Tensor<float>(new[] { count }, data);
    }

    private static void AssertAllFinite(Tensor<float> t, string context)
    {
        var d = t.ToArray();
        Assert.True(d.Length > 0, $"{context}: output must be non-empty.");
        for (int i = 0; i < d.Length; i++)
            Assert.True(!float.IsNaN(d[i]) && !float.IsInfinity(d[i]),
                $"{context}: output[{i}] = {d[i]} is not finite.");
    }

    private static LayoutXLM<float> CreateSmallLayoutXLM()
        => new LayoutXLM<float>(CreateArchitecture(imageSize: 32), numClasses: 7, imageSize: 32,
            maxSequenceLength: 64, hiddenDim: 64, numLayers: 2, numHeads: 4, vocabSize: 100,
            visualBackboneChannels: 32);

    private static LayoutLMv2<float> CreateSmallLayoutLMv2()
        => new LayoutLMv2<float>(CreateArchitecture(imageSize: 32), numClasses: 7, imageSize: 32,
            maxSequenceLength: 64, hiddenDim: 64, numLayers: 2, numHeads: 4, vocabSize: 100,
            visualBackboneChannels: 32);

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_EncodeMultimodal_GrowsSequenceNotBatch_AndIsFinite()
    {
        await Task.Yield();
        var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);
        var image = CreateSmallImage(32);

        var fused = model.EncodeMultimodal(CreateTokenIds(16), image);

        Assert.Equal(3, fused.Rank);
        Assert.Equal(1, fused.Shape[0]);          // batch preserved — pre-fix this was 1 + Ltext
        Assert.True(fused.Shape[1] > 1, $"fused sequence length {fused.Shape[1]} must span multiple tokens.");
        AssertAllFinite(fused, "LayoutXLM full fusion");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_EncodeMultimodal_MoreTextTokens_LengthensSequence()
    {
        await Task.Yield();
        var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);
        var image = CreateSmallImage(32);

        var fused8 = model.EncodeMultimodal(CreateTokenIds(8), image);
        var fused24 = model.EncodeMultimodal(CreateTokenIds(24), image);

        // 16 extra text tokens must extend the joint SEQUENCE by 16 while the batch stays 1.
        // Pre-fix (concat on axis 0) the extra tokens grew the batch dimension instead.
        Assert.Equal(1, fused8.Shape[0]);
        Assert.Equal(1, fused24.Shape[0]);
        Assert.Equal(16, fused24.Shape[1] - fused8.Shape[1]);
        AssertAllFinite(fused24, "LayoutXLM fusion (24 tokens)");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_EncodeMultimodal_GrowsSequenceNotBatch_AndIsFinite()
    {
        await Task.Yield();
        var model = CreateSmallLayoutLMv2();
        model.SetTrainingMode(false);
        var image = CreateSmallImage(32);

        var fused = model.EncodeMultimodal(CreateTokenIds(16), image);

        Assert.Equal(3, fused.Rank);
        Assert.Equal(1, fused.Shape[0]);
        Assert.True(fused.Shape[1] > 1, $"fused sequence length {fused.Shape[1]} must span multiple tokens.");
        AssertAllFinite(fused, "LayoutLMv2 full fusion");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_EncodeMultimodal_MoreTextTokens_LengthensSequence()
    {
        await Task.Yield();
        var model = CreateSmallLayoutLMv2();
        model.SetTrainingMode(false);
        var image = CreateSmallImage(32);

        var fused8 = model.EncodeMultimodal(CreateTokenIds(8), image);
        var fused24 = model.EncodeMultimodal(CreateTokenIds(24), image);

        Assert.Equal(1, fused8.Shape[0]);
        Assert.Equal(1, fused24.Shape[0]);
        Assert.Equal(16, fused24.Shape[1] - fused8.Shape[1]);
        AssertAllFinite(fused24, "LayoutLMv2 fusion (24 tokens)");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_EncodeMultimodal_FiniteAfterPoolChurn()
    {
        // End-to-end guard for the original symptom: a sibling model's forward/backward churns the
        // thread-local tensor pool, then a fresh model's fusion must still produce finite output.
        // Pre-fix (uninitialized concat tail) this leaked stale pool data as NaN.
        await Task.Yield();
        var churn = CreateSmallLayoutXLM();
        churn.SetTrainingMode(false);
        var churnImage = CreateSmallImage(32);
        for (int i = 0; i < 5; i++)
            churn.EncodeMultimodal(CreateTokenIds(16 + i), churnImage);

        var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);
        var fused = model.EncodeMultimodal(CreateTokenIds(16), CreateSmallImage(32));

        AssertAllFinite(fused, "LayoutXLM fusion after pool churn");
        Assert.Equal(1, fused.Shape[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_SingleModality_ImageOnlyAndTextOnly_AreFinite()
    {
        // Modality-robustness edge cases: the fusion path degrades gracefully to each single stream.
        await Task.Yield();
        var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);

        var imageOnly = model.Predict(CreateSmallImage(32));   // routes through the visual stream
        AssertAllFinite(imageOnly, "LayoutXLM image-only");

        var textOnly = model.Predict(CreateTokenIds(16));      // routes through the text stream
        AssertAllFinite(textOnly, "LayoutXLM text-only");
    }

    #endregion
}
