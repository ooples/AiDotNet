using AiDotNet.Document;
using AiDotNet.Document.GraphBased;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for graph-based document models.
/// </summary>
public class GraphBasedDocumentTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 9);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    // A rank-2 [N, F] node-feature matrix (one row per document node/segment).
    private static Tensor<double> CreateNodeFeatures(int numNodes, int featureDim)
    {
        var data = new Vector<double>(numNodes * featureDim);
        for (int i = 0; i < data.Length; i++)
            data[i] = 0.01 * ((i % 11) + 1);
        return new Tensor<double>(new[] { numNodes, featureDim }, data);
    }

    #region DocGCN Tests

    [Fact(Timeout = 120000)]
    public async Task DocGCN_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new DocGCN<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DocGCN_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new DocGCN<double>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task DocGCN_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new DocGCN<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocGCN", meta.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task DocGCN_FusedMultimodal_CombinesHeterogeneousNodes()
    {
        await Task.Yield();
        using var model = new DocGCN<double>(CreateArchitecture());
        using var textNodes = CreateNodeFeatures(6, 128);
        using var visualNodes = CreateNodeFeatures(10, 128);

        using var textOnly = model.PredictMultimodal(textNodes, null);        // graceful degradation
        using var visualOnly = model.PredictMultimodal(null, visualNodes);    // graceful degradation
        using var fused = model.PredictMultimodal(textNodes, visualNodes);    // heterogeneous joint graph

        AssertAllFinite(textOnly, "DocGCN text-only");
        AssertAllFinite(visualOnly, "DocGCN visual-only");
        AssertAllFinite(fused, "DocGCN fused");

        Assert.Equal(textNodes.Shape[0] + visualNodes.Shape[0], fused.Shape[0]);
        Assert.True(fused.Shape[0] > textOnly.Shape[0] && fused.Shape[0] > visualOnly.Shape[0],
            "Fused heterogeneous node count should exceed each single modality.");
    }

    #endregion

    #region PICK Tests

    // PICK takes packed OCR text segments [N, 4 + T]: a box (x0, y0, x1, y1) in page pixels, then T token ids.
    private static PICK<double> CreateSmallPICK()
        => new PICK<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputHeight: 3, inputWidth: 9, outputSize: 5),
            new AiDotNet.Document.Options.PICKOptions
            {
                NumEntityTypes = 5,
                ImageSize = 64,
                MaxSequenceLength = 32,
                HiddenDim = 32,
                NumHeads = 2,
                NumEncoderLayers = 1,
                FeedForwardDim = 64,
                VocabSize = 50,
                ImageFeatureDim = 16,
                ImageEncoderDepths = new[] { 1, 1, 1, 1 },
                GraphLearningDim = 8,
                NumGcnLayers = 2,
                LstmHiddenDim = 16,
                LstmLayers = 1
            });

    private static Tensor<double> CreateSegments()
    {
        // Three segments of up to 5 tokens; the third is shorter (trailing padding).
        double[][] rows =
        {
            new double[] { 4, 4, 30, 12, 3, 7, 11, 2, 9 },
            new double[] { 34, 4, 60, 12, 5, 5, 8, 0, 0 },
            new double[] { 4, 40, 44, 52, 12, 6, 0, 0, 0 },
        };
        var t = new Tensor<double>(new[] { rows.Length, rows[0].Length });
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++) t[i, j] = rows[i][j];
        return t;
    }

    private static Tensor<double> CreatePage(int size = 64)
    {
        var page = new Tensor<double>(new[] { 3, size, size });
        for (int i = 0; i < page.Length; i++) page[i] = ((i * 37) % 101) / 101.0;
        return page;
    }

    [Fact(Timeout = 120000)]
    public async Task PICK_NativeConstruction_Succeeds()
    {
        await Task.Yield();
        using var model = new PICK<double>(CreateArchitecture());
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task PICK_Predict_TagsEveryTokenSlotWithOneCrfTag()
    {
        await Task.Yield();
        using var model = CreateSmallPICK();
        using var segments = CreateSegments();
        using var output = model.PredictDocument(segments);

        Assert.Equal(new[] { 3 * 5, 5 }, output.Shape.ToArray());
        int[] lengths = { 5, 3, 2 };
        for (int i = 0; i < 3; i++)
            for (int t = 0; t < 5; t++)
            {
                double rowSum = 0;
                for (int c = 0; c < 5; c++)
                {
                    double v = output[(i * 5) + t, c];
                    Assert.True(v == 0 || v == 1, $"slot ({i},{t}) is not one-hot: {v}");
                    rowSum += v;
                }
                // A real token carries exactly one Viterbi tag; a padding slot carries none.
                Assert.Equal(t < lengths[i] ? 1.0 : 0.0, rowSum);
            }
    }

    [Fact(Timeout = 120000)]
    public async Task PICK_GetModelMetadata_ReturnsValidData()
    {
        await Task.Yield();
        using var model = new PICK<double>(CreateArchitecture());
        var meta = model.GetModelMetadata();
        Assert.Equal("PICK", meta.Name);
    }

    [Fact(Timeout = 120000)]
    public async Task PICK_TrainDocument_WithPage_DecreasesTheCrfAndGraphObjective()
    {
        await Task.Yield();
        using var model = CreateSmallPICK();
        using var segments = CreateSegments();
        using var page = CreatePage();
        using var tags = new Tensor<double>(new[] { 15 });
        for (int s = 0; s < 15; s++) tags[s] = (s % 4) + 1;

        model.TrainDocument(segments, page, tags);
        double first = model.GetLastLoss();
        for (int step = 0; step < 14; step++) model.TrainDocument(segments, page, tags);
        double last = model.GetLastLoss();

        Assert.False(double.IsNaN(last) || double.IsInfinity(last), $"loss is {last}");
        Assert.True(last < first, $"15 steps on one document did not lower the CRF + graph loss: {first} -> {last}");
    }

    [Fact(Timeout = 120000)]
    public async Task PICK_PageImage_ChangesTheTagEmissions()
    {
        await Task.Yield();
        using var model = CreateSmallPICK();
        using var segments = CreateSegments();
        using var page = CreatePage();
        using var tags = new Tensor<double>(new[] { 15 });
        // Train with the page so the image branch's BatchNorm and weights are live, then compare the
        // training-path emissions with and without it: the image embedding must reach the tagger.
        for (int step = 0; step < 3; step++) model.TrainDocument(segments, page, tags);
        using var withPage = model.EmissionsForTest(segments, page);
        using var withoutPage = model.EmissionsForTest(segments, null);
        double diff = 0;
        for (int i = 0; i < withPage.Length; i++) diff += Math.Abs(withPage[i] - withoutPage[i]);
        Assert.True(diff > 1e-8, "The page image did not change PICK's emissions: the image branch is disconnected.");
    }

    [Fact(Timeout = 120000)]
    public async Task PICK_Graph_CouplesSegments()
    {
        await Task.Yield();
        using var model = CreateSmallPICK();
        using var segments = CreateSegments();
        using var baseline = model.EmissionsForTest(segments, null);
        // Change ONLY the third segment's tokens; the first segment's emissions must move through the graph.
        using var altered = CreateSegments();
        altered[2, 4] = 40;
        altered[2, 5] = 41;
        using var changed = model.EmissionsForTest(altered, null);
        double diff = 0;
        for (int c = 0; c < baseline.Shape[1]; c++) diff += Math.Abs(baseline[0, c] - changed[0, c]);
        Assert.True(diff > 1e-8, "Segment 0's emission ignored segment 2: the graph does not couple segments.");
    }

    #endregion

    #region TRIE Tests

    [Fact(Timeout = 120000)]
    public async Task TRIE_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new TRIE<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task TRIE_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new TRIE<double>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task TRIE_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new TRIE<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("TRIE", meta.Name);
    }

    // ===== Modality-robust fusion (task #48): text-only, image-only, and fused inference. =====

    private static Tensor<double> CreateTextTokens(int numTokens = 8, int featureDim = 128)
    {
        var data = new Vector<double>(numTokens * featureDim);
        for (int i = 0; i < data.Length; i++)
            data[i] = 0.01 * ((i % 7) + 1);
        return new Tensor<double>(new[] { numTokens, featureDim }, data);
    }

    private static void AssertAllFinite(Tensor<double> t, string ctx)
    {
        for (int i = 0; i < t.Length; i++)
        {
            Assert.False(double.IsNaN(t[i]), $"{ctx}: output[{i}] is NaN");
            Assert.False(double.IsInfinity(t[i]), $"{ctx}: output[{i}] is Infinity");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task TRIE_TextOnly_ProducesFiniteOutput()
    {
        await Task.Yield();
        using var model = new TRIE<double>(CreateArchitecture());
        using var tokens = CreateTextTokens();
        using var output = model.Predict(tokens);   // rank-2 -> text stream (graceful single-modality)
        Assert.True(output.Length > 0);
        AssertAllFinite(output, "TRIE text-only");
    }

    [Fact(Timeout = 120000)]
    public async Task TRIE_FusedMultimodal_CombinesTextAndVisualNodes()
    {
        await Task.Yield();
        using var model = new TRIE<double>(CreateArchitecture());
        using var tokens = CreateTextTokens();
        using var image = CreateSmallImage();

        using var textOnly = model.PredictMultimodal(tokens, null);   // graceful degradation
        using var imageOnly = model.PredictMultimodal(null, image);   // graceful degradation
        using var fused = model.PredictMultimodal(tokens, image);     // joint reasoning

        AssertAllFinite(textOnly, "TRIE fused-text-only");
        AssertAllFinite(imageOnly, "TRIE fused-image-only");
        AssertAllFinite(fused, "TRIE fused");

        // The fused node set is the union of text nodes and visual nodes, so it must have strictly
        // more nodes than either single modality — proof the streams were actually concatenated.
        Assert.True(fused.Shape[0] > textOnly.Shape[0],
            $"Fused node count ({fused.Shape[0]}) should exceed text-only ({textOnly.Shape[0]}).");
        Assert.True(fused.Shape[0] > imageOnly.Shape[0],
            $"Fused node count ({fused.Shape[0]}) should exceed image-only ({imageOnly.Shape[0]}).");
    }

    #endregion

    #region LayoutGraph Tests

    [Fact(Timeout = 120000)]
    public async Task LayoutGraph_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutGraph<double>(arch);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutGraph_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutGraph<double>(arch);
        using var input = CreateSmallImage();
        using var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task LayoutGraph_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        using var model = new LayoutGraph<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutGraph", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllGraphBasedModels_SupportsTraining_InNativeMode()
    {
        // Each model owns pooled buffers, so it is declared under its own using scope before the
        // array is built: if a later constructor throws, the array assignment never completes and a
        // finally-based cleanup would never run, leaking every model already constructed.
        using var docGcn = new DocGCN<double>(CreateArchitecture());
        using var pick = new PICK<double>(CreateArchitecture());
        using var trie = new TRIE<double>(CreateArchitecture());
        using var layoutGraph = new LayoutGraph<double>(CreateArchitecture());

        var models = new DocumentNeuralNetworkBase<double>[] { docGcn, pick, trie, layoutGraph };

        foreach (var model in models)
        {
            // All native mode models support training
            Assert.True(model.SupportsTraining);
        }
    }

    #endregion
}
