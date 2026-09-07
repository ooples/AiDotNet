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

    // LayoutLM v1 (Xu et al. 2020, KDD) is a TEXT + 2D-layout model: it consumes a rank-1 sequence of
    // token IDs (looked up by its front EmbeddingLayer), NOT a document image — the image-region stream
    // first appears in LayoutLMv2/v3. So the v1 forward is exercised with a token-ID sequence; the
    // image-consuming models below (LayoutLMv2/v3, DiT, ...) keep CreateSmallImage.
    private static Tensor<float> CreateTokenSequence(int length = 16)
    {
        var data = new Vector<float>(length);
        for (int i = 0; i < length; i++)
            data[i] = i % 50; // token IDs well within any BERT-scale vocab
        return new Tensor<float>(new[] { length }, data);
    }

    #region LayoutLM Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLM<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLM<float>(arch);
        using var input = CreateTokenSequence();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLM_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLM<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLM", meta.Name);
    }

    #endregion

    #region LayoutLMv2 Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLMv2<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLMv2<float>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLMv2<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv2", meta.Name);
    }

    #endregion

    #region LayoutLMv3 Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLMv3<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLMv3<float>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv3_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutLMv3<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutLMv3", meta.Name);
    }

    #endregion

    #region LayoutXLM Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutXLM<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutXLM<float>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutXLM<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutXLM", meta.Name);
    }

    #endregion

    #region DocFormer Tests

    [Fact(Timeout = 120000)]
    public async Task DocFormer_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new DocFormer<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DocFormer_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new DocFormer<float>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task DocFormer_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new DocFormer<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocFormer", meta.Name);
    }

    #endregion

    #region DiT Tests

    [Fact(Timeout = 120000)]
    public async Task DiT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new DiT<float>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new DiT<float>(arch, imageSize: 64);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new DiT<float>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DiT", meta.Name);
    }

    #endregion

    #region LiLT Tests

    [Fact(Timeout = 120000)]
    public async Task LiLT_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new LiLT<float>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LiLT_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new LiLT<float>(arch);
        // LiLT (Wang et al., ACL 2022) is text + layout with NO vision stream — decoupling the two is
        // the paper's whole point, which is what lets one layout encoder pair with any language's text
        // encoder. Handing it a document IMAGE fed float pixels straight into its front EmbeddingLayer,
        // which correctly refused them ("requires token indices, but element 0 is 0.0655..."). Same
        // fixture mistake already corrected for LayoutLM above.
        using var input = CreateTokenSequence();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LiLT_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new LiLT<float>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LiLT", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllLayoutAwareModels_RequiresOCR_IsTrue()
    {
        var models = new DocumentNeuralNetworkBase<float>[]
        {
            new LayoutLM<float>(CreateArchitecture()),
            new LayoutLMv2<float>(CreateArchitecture()),
            new LayoutLMv3<float>(CreateArchitecture()),
            new LayoutXLM<float>(CreateArchitecture()),
            new DocFormer<float>(CreateArchitecture()),
            new LiLT<float>(CreateArchitecture()),
        };

        try
        {
            foreach (var model in models)
            {
                // Layout-aware models require OCR to provide text and bounding boxes
                Assert.True(model.RequiresOCR);
            }
        }
        finally
        {
            foreach (var model in models)
            {
                model.Dispose();
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DiT_RequiresOCR_IsFalse()
    {
        var arch = CreateArchitecture();
        using var model = new DiT<float>(arch, imageSize: 64);
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
        using var model = CreateSmallLiLT();
        model.SetTrainingMode(false);
        using var tokens = CreateIntTokenVector(8);
        using var boxes = CreateBoxFeatures(8);

        using var textOnly = model.EncodeDualStream(tokens, layoutBoxes: null);
        using var fused = model.EncodeDualStream(tokens, boxes);

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
        using var model = CreateSmallLiLT();
        model.SetTrainingMode(false);

        using var tokens = CreateIntTokenVector(8);
        using var output = model.EncodeDualStream(tokens, layoutBoxes: null);
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
        using var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);
        using var image = CreateSmallImage(32);
        using var tokens = CreateTokenIds(16);

        using var fused = model.EncodeMultimodal(tokens, image);

        Assert.Equal(3, fused.Rank);
        Assert.Equal(1, fused.Shape[0]);          // batch preserved — pre-fix this was 1 + Ltext
        Assert.True(fused.Shape[1] > 1, $"fused sequence length {fused.Shape[1]} must span multiple tokens.");
        AssertAllFinite(fused, "LayoutXLM full fusion");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_EncodeMultimodal_MoreTextTokens_LengthensSequence()
    {
        await Task.Yield();
        using var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);
        using var image = CreateSmallImage(32);
        using var tokens8 = CreateTokenIds(8);
        using var tokens24 = CreateTokenIds(24);

        using var fused8 = model.EncodeMultimodal(tokens8, image);
        using var fused24 = model.EncodeMultimodal(tokens24, image);

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
        using var model = CreateSmallLayoutLMv2();
        model.SetTrainingMode(false);
        using var image = CreateSmallImage(32);
        using var tokens = CreateTokenIds(16);

        using var fused = model.EncodeMultimodal(tokens, image);

        Assert.Equal(3, fused.Rank);
        Assert.Equal(1, fused.Shape[0]);
        Assert.True(fused.Shape[1] > 1, $"fused sequence length {fused.Shape[1]} must span multiple tokens.");
        AssertAllFinite(fused, "LayoutLMv2 full fusion");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutLMv2_EncodeMultimodal_MoreTextTokens_LengthensSequence()
    {
        await Task.Yield();
        using var model = CreateSmallLayoutLMv2();
        model.SetTrainingMode(false);
        using var image = CreateSmallImage(32);
        using var tokens8 = CreateTokenIds(8);
        using var tokens24 = CreateTokenIds(24);

        using var fused8 = model.EncodeMultimodal(tokens8, image);
        using var fused24 = model.EncodeMultimodal(tokens24, image);

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
        using var churn = CreateSmallLayoutXLM();
        churn.SetTrainingMode(false);
        using var churnImage = CreateSmallImage(32);
        for (int i = 0; i < 5; i++)
        {
            using var churnTokens = CreateTokenIds(16 + i);
            using var churnOutput = churn.EncodeMultimodal(churnTokens, churnImage);
        }

        using var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);
        using var tokens = CreateTokenIds(16);
        using var image = CreateSmallImage(32);
        using var fused = model.EncodeMultimodal(tokens, image);

        AssertAllFinite(fused, "LayoutXLM fusion after pool churn");
        Assert.Equal(1, fused.Shape[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutXLM_SingleModality_ImageOnlyAndTextOnly_AreFinite()
    {
        // Modality-robustness edge cases: the fusion path degrades gracefully to each single stream.
        await Task.Yield();
        using var model = CreateSmallLayoutXLM();
        model.SetTrainingMode(false);

        using var image = CreateSmallImage(32);
        using var imageOnly = model.Predict(image);   // routes through the visual stream
        AssertAllFinite(imageOnly, "LayoutXLM image-only");

        using var tokens = CreateTokenIds(16);
        using var textOnly = model.Predict(tokens);      // routes through the text stream
        AssertAllFinite(textOnly, "LayoutXLM text-only");
    }

    #endregion

    #region LiLT layout-stream reachability

    /// <summary>
    /// LiLT's layout stream used to be unreachable from Predict and from Train alike: both Forward
    /// and ForwardForTraining passed null for boxes, and the one entry point that accepted them
    /// (EncodeDualStream) opens a NoGradScope, so the layout half could never be trained at all.
    /// A packed row must now reach the stream through the ordinary Predict path.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task LiLT_PackedInput_ReachesTheLayoutStream()
    {
        await Task.Yield();
        using var model = new LiLT<float>(CreateArchitecture());
        model.SetTrainingMode(false);

        using var leftTokens = CreatePackedTokens(8, x0: 0, y0: 0);
        using var rightTokens = CreatePackedTokens(8, x0: 300, y0: 400);
        using var left = model.Predict(leftTokens);
        using var right = model.Predict(rightTokens);

        AssertAllFinite(left, "LiLT packed left");
        AssertAllFinite(right, "LiLT packed right");

        bool differs = false;
        for (int i = 0; i < left.Length && i < right.Length && !differs; i++)
        {
            if (System.Math.Abs(left.Data.Span[i] - right.Data.Span[i]) > 1e-9f)
                differs = true;
        }

        Assert.True(differs,
            "Identical tokens with different boxes gave identical output, so the packed row is not " +
            "reaching LiLT's layout stream.");
    }

    /// <summary>
    /// Text-only input keeps working unchanged — a caller with no OCR boxes still gets the
    /// text-only BiACM path rather than an error.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task LiLT_TokensOnly_StillRunsTextOnly()
    {
        await Task.Yield();
        using var model = new LiLT<float>(CreateArchitecture());
        model.SetTrainingMode(false);

        using var tokens = CreateTokenIds(16);
        using var output = model.Predict(tokens);
        AssertAllFinite(output, "LiLT text-only");
    }

    #endregion

    #region DocFormer Tests

    private static DocFormer<float> CreateSmallDocFormer()
        => new DocFormer<float>(CreateArchitecture(imageSize: 32), numClasses: 7, imageSize: 32,
            maxSequenceLength: 64, hiddenDim: 64, numLayers: 2, numHeads: 4, vocabSize: 100);

    /// <summary>
    /// DocFormer routes by input rank, and its text stream is now ONE LayoutEmbeddingLayer where it
    /// used to be an EmbeddingLayer + PositionalEncodingLayer pair. That collapse moved every index
    /// after it, so both branches are exercised here: the shared stack starts one slot earlier and a
    /// mistake would either run the text embedding over image features or skip the first shared layer.
    /// The generated DocFormerTests cannot catch it — all 26 of them fail in their warm-up Predict on
    /// a pre-existing fixture problem (continuous floats fed to a rank-1 token input), so they would
    /// stay red either way and prove nothing about the routing.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DocFormer_BothModalities_RouteAndStayFinite()
    {
        await Task.Yield();
        using var model = CreateSmallDocFormer();
        model.SetTrainingMode(false);

        using var image = CreateSmallImage(32);
        using var imageOnly = model.Predict(image);   // visual backbone, then the shared stack
        AssertAllFinite(imageOnly, "DocFormer image-only");

        using var tokens = CreateTokenIds(16);
        using var textOnly = model.Predict(tokens);      // layout embedding, then the shared stack
        AssertAllFinite(textOnly, "DocFormer text-only");
    }

    /// <summary>
    /// The point of the change: DocFormer's spatial tables were model fields that nothing read, so
    /// two tokens printed in different places produced identical vectors. Feeding the same token IDs
    /// with different boxes must now move the output.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DocFormer_BoundingBoxes_ChangeTheOutput()
    {
        await Task.Yield();
        using var model = CreateSmallDocFormer();
        model.SetTrainingMode(false);

        using var topLeftTokens = CreatePackedTokens(8, x0: 0, y0: 0);
        using var bottomRightTokens = CreatePackedTokens(8, x0: 300, y0: 400);
        using var topLeft = model.Predict(topLeftTokens);
        using var bottomRight = model.Predict(bottomRightTokens);

        AssertAllFinite(topLeft, "DocFormer packed top-left");
        AssertAllFinite(bottomRight, "DocFormer packed bottom-right");

        bool differs = false;
        for (int i = 0; i < topLeft.Length && i < bottomRight.Length && !differs; i++)
        {
            if (System.Math.Abs(topLeft.Data.Span[i] - bottomRight.Data.Span[i]) > 1e-9f)
                differs = true;
        }

        Assert.True(differs,
            "Identical tokens at different page positions gave identical output, so DocFormer's " +
            "spatial embeddings still are not reaching the forward pass.");
    }

    /// <summary>
    /// Builds LayoutEmbeddingLayer's packed row: [seq, 5] of (tokenId, x0, y0, x1, y1).
    /// </summary>
    private static Tensor<float> CreatePackedTokens(int count, int x0, int y0, int vocab = 100)
    {
        var data = new Vector<float>(count * 5);
        for (int i = 0; i < count; i++)
        {
            int b = i * 5;
            data[b] = i % vocab;
            data[b + 1] = x0 + i;         // boxes march along the line
            data[b + 2] = y0;
            data[b + 3] = x0 + i + 10;
            data[b + 4] = y0 + 12;
        }

        return new Tensor<float>(new[] { count, 5 }, data);
    }

    #endregion
}
