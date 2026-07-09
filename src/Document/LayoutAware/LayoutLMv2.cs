using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.LayoutAware;

/// <summary>
/// LayoutLMv2 neural network for document understanding with visual features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutLMv2 extends LayoutLM by adding visual features from a CNN backbone,
/// enabling the model to understand documents through text, layout, AND image features.
/// </para>
/// <para>
/// <b>For Beginners:</b> LayoutLMv2 improves on v1 by also looking at the actual image:
/// 1. Text content (what the words say)
/// 2. Layout structure (where words are positioned)
/// 3. Visual appearance (what the document looks like)
///
/// Key improvements over v1:
/// - Visual backbone (ResNeXt-FPN) for image features
/// - Spatial-aware self-attention mechanism
/// - Pre-training on both text-layout and image-text-layout alignment
///
/// Example usage:
/// <code>
/// var model = new LayoutLMv2&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding" (ACL 2021)
/// https://arxiv.org/abs/2012.14740
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding", "https://doi.org/10.48550/arXiv.2012.14740", Year = 2021, Authors = "Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou")]
public class LayoutLMv2<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>
{
    private readonly LayoutLMv2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private readonly int _numClasses;
    private readonly int _visualBackboneChannels;

    // Native mode layers
    private readonly List<ILayer<T>> _visualBackboneLayers = [];
    private readonly List<ILayer<T>> _textEmbeddingLayers = [];
    private readonly List<ILayer<T>> _transformerLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    // Learnable embeddings
    private Tensor<T>? _positionEmbeddings;
    private Tensor<T>? _spatialPositionEmbeddings;
    private Tensor<T>? _visualPositionEmbeddings;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public IReadOnlyList<LayoutElementType> SupportedElementTypes { get; } =
    [
        LayoutElementType.Text,
        LayoutElementType.Title,
        LayoutElementType.List,
        LayoutElementType.Table,
        LayoutElementType.Figure,
        LayoutElementType.Caption,
        LayoutElementType.Header,
        LayoutElementType.Footer,
        LayoutElementType.FormField
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a LayoutLMv2 model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="visualBackboneChannels">Visual backbone output channels (default: 256).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    public LayoutLMv2(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numClasses = 7,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int visualBackboneChannels = 256,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LayoutLMv2Options? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutLMv2Options();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _visualBackboneChannels = visualBackboneChannels;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a LayoutLMv2 model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text processing (optional).</param>
    /// <param name="numClasses">Number of output classes (default: 7).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="visualBackboneChannels">Visual backbone output channels (default: 256).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (LayoutLMv2-Base from ACL 2021):</b>
    /// - Text encoder: BERT-base architecture
    /// - Visual backbone: ResNeXt-101 FPN
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// - Image size: 224×224
    /// </para>
    /// </remarks>
    public LayoutLMv2(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numClasses = 7,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int visualBackboneChannels = 256,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LayoutLMv2Options? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutLMv2Options();
        Options = _options;

        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _visualBackboneChannels = visualBackboneChannels;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

        InitializeLayers();
        InitializeEmbeddings();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultLayoutLMv2Layers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            visualBackboneChannels: _visualBackboneChannels,
            numClasses: _numClasses));

        DistributeLayers();
    }

    // CreateDefaultLayoutLMv2Layers emits the layers in three role groups, in this order:
    //   [0 .. VisualBackboneLayerCount)                        = ResNeXt-FPN visual backbone
    //                                                            (conv/BN/pool stack + a final C->hidden projection Dense),
    //   [.. + TextEmbeddingLayerCount)                          = word-embedding / position / layernorm / dropout,
    //   [rest]                                                  = multimodal transformer encoder + classification head.
    private const int VisualBackboneLayerCount = 12;
    private const int TextEmbeddingLayerCount = 4;

    // Re-links the per-role forward-path sublists to the CURRENT Layers. The two-stream forward reads
    // these lists, not Layers directly, so they must be rebuilt whenever Layers is replaced — including
    // after deserialization (the base clears Layers and adds freshly-deserialized layers).
    private void DistributeLayers()
    {
        _visualBackboneLayers.Clear();
        _textEmbeddingLayers.Clear();
        _transformerLayers.Clear();
        for (int i = 0; i < Layers.Count; i++)
        {
            if (i < VisualBackboneLayerCount)
                _visualBackboneLayers.Add(Layers[i]);
            else if (i < VisualBackboneLayerCount + TextEmbeddingLayerCount)
                _textEmbeddingLayers.Add(Layers[i]);
            else
                _transformerLayers.Add(Layers[i]);
        }
    }

    /// <summary>
    /// Runs LayoutLMv2's real two-stream forward and fuses whatever modalities are present.
    /// </summary>
    /// <remarks>
    /// Reference LayoutLMv2 (Xu et al. 2021) REQUIRES both a document image AND text tokens and crashes
    /// otherwise. This implementation is modality-robust: it runs the visual backbone and/or the text
    /// embedding stream depending on which inputs are supplied, concatenates the resulting token
    /// sequences, and runs the shared multimodal transformer + head — so it also handles OCR-text-only
    /// or image-only documents (graceful degradation), which the reference model cannot.
    /// </remarks>
    private Tensor<T> RunMultimodal(Tensor<T>? textTokens, Tensor<T>? documentImage)
    {
        Tensor<T>? textSeq = textTokens is not null ? RunTextStream(textTokens) : null;
        Tensor<T>? visualSeq = documentImage is not null ? RunVisualStream(documentImage) : null;

        Tensor<T> seq;
        if (textSeq is not null && visualSeq is not null)
        {
            // Fuse the way LayoutLMv2 (Xu et al. 2021, §3.1) does: stack the visual and text token
            // sequences along the SEQUENCE axis (visual first, then text) into one joint sequence for
            // the shared multimodal transformer. Normalize both streams to a batched [B, L, D] first —
            // the visual backbone emits [B, Lvis, D] while the text stream can emit an unbatched
            // [Ltext, D] (and a continuous-valued token tensor projects to [1, D]) — so the
            // concatenation matches on batch and hidden and only grows the sequence axis. The prior
            // axis-(Rank-2) concat on unequal-rank streams was invalid and only appeared to work when
            // the output buffer's unwritten tail happened to be zero.
            var vis = AlignToBatchedSequence(visualSeq);
            var txt = AlignToBatchedSequence(textSeq);
            seq = Engine.TensorConcatenate([vis, txt], axis: 1);
        }
        else
        {
            seq = textSeq ?? visualSeq
                ?? throw new ArgumentException(
                    "LayoutLMv2 requires at least one modality: text token IDs (rank <= 2) or a document image (rank >= 3).");
        }

        foreach (var layer in _transformerLayers)
            seq = layer.Forward(seq);
        return seq;
    }

    // Normalizes a token sequence to a batched [B, L, D] layout so the two fusion streams concatenate
    // cleanly on the sequence axis. A [L, D] stream (unbatched, e.g. the text embedding on a rank-1
    // token vector) gains a leading batch of 1; a continuous [1, D] projection becomes a single-token
    // [1, 1, D]; an already-batched [B, L, D] passes through unchanged.
    private Tensor<T> AlignToBatchedSequence(Tensor<T> t)
    {
        if (t.Rank == 3) return t;
        if (t.Rank == 2) return Engine.Reshape(t, new[] { 1, t.Shape[0], t.Shape[1] });
        throw new ArgumentException($"Fusion stream must be rank 2 or 3, got rank {t.Rank}.");
    }

    // Text stream: token IDs -> word embedding -> position -> layernorm -> dropout => [seq, hidden].
    private Tensor<T> RunTextStream(Tensor<T> textTokens)
    {
        var x = textTokens;
        foreach (var layer in _textEmbeddingLayers)
            x = layer.Forward(x);
        return x;
    }

    // Visual stream: image -> conv/BN/pool backbone -> flatten spatial grid to a token sequence ->
    // channel projection => [numPatches, hidden]. The final backbone layer is the C->hidden projection,
    // applied AFTER the spatial->token reshape so it maps channels (not width) to the hidden dim.
    private Tensor<T> RunVisualStream(Tensor<T> documentImage)
    {
        var x = documentImage;
        int projIndex = _visualBackboneLayers.Count - 1;
        for (int i = 0; i < projIndex; i++)
            x = _visualBackboneLayers[i].Forward(x);

        x = FlattenSpatialToTokens(x);
        if (projIndex >= 0 && projIndex < _visualBackboneLayers.Count)
            x = _visualBackboneLayers[projIndex].Forward(x);
        return x;
    }

    // [C, H, W] -> [H*W, C]; [B, C, H, W] -> [B, H*W, C]. Puts channels last so each spatial location
    // becomes a token whose feature vector the projection Dense maps to the hidden dim.
    private Tensor<T> FlattenSpatialToTokens(Tensor<T> feat)
    {
        if (feat.Rank == 4)
        {
            int b = feat.Shape[0], c = feat.Shape[1], n = feat.Shape[2] * feat.Shape[3];
            return Engine.TensorPermute(Engine.Reshape(feat, new[] { b, c, n }), new[] { 0, 2, 1 });
        }
        if (feat.Rank == 3)
        {
            int c = feat.Shape[0], n = feat.Shape[1] * feat.Shape[2];
            return Engine.TensorPermute(Engine.Reshape(feat, new[] { c, n }), new[] { 1, 0 });
        }
        return feat;
    }

    /// <summary>
    /// Full text+image fusion entry (industry-standard LayoutLMv2): encodes BOTH a token-ID sequence
    /// and a document image and fuses them through the multimodal transformer.
    /// </summary>
    public Tensor<T> EncodeMultimodal(Tensor<T> textTokens, Tensor<T> documentImage)
    {
        // Inference entry: mirror Predict()/PredictCore by suppressing gradient-tape recording
        // (PyTorch torch.no_grad() semantics). RunMultimodal issues raw Engine.Reshape/Permute/
        // Concatenate ops that would otherwise record onto the shared autodiff tape; if a prior
        // training pass left that singleton tape non-empty, replaying it here poisons the fusion
        // forward with stale/NaN buffers. NoGradScope makes this direct call as tape-clean as
        // the Predict()-wrapped image-only path. ForwardForTraining keeps recording (no scope).
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var image = PreprocessDocument(documentImage);
        return RunMultimodal(textTokens, image);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var prepared = PreprocessDocument(input);
        var (tokens, image) = prepared.Rank <= 2 ? (prepared, (Tensor<T>?)null) : ((Tensor<T>?)null, prepared);
        return RunMultimodal(tokens, image);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        // The base walks Layers linearly feeding the raw input, which crashes the visual conv backbone
        // on a token-only input (rank-1). Replay the real two-stream forward, capturing each layer.
        var activations = new Dictionary<string, Tensor<T>>();
        if (!_useNativeMode)
            return activations;

        var prepared = PreprocessDocument(input);
        var (tokens, image) = prepared.Rank <= 2 ? (prepared, (Tensor<T>?)null) : ((Tensor<T>?)null, prepared);

        int idx = 0;
        Tensor<T>? textSeq = null, visualSeq = null;

        if (tokens is not null)
        {
            var x = tokens;
            foreach (var layer in _textEmbeddingLayers)
            {
                x = layer.Forward(x);
                activations[$"Layer_{idx++}_{layer.GetType().Name}"] = x.Clone();
            }
            textSeq = x;
        }

        if (image is not null)
        {
            var x = image;
            int projIndex = _visualBackboneLayers.Count - 1;
            for (int i = 0; i < projIndex; i++)
            {
                x = _visualBackboneLayers[i].Forward(x);
                activations[$"Layer_{idx++}_{_visualBackboneLayers[i].GetType().Name}"] = x.Clone();
            }
            x = FlattenSpatialToTokens(x);
            if (projIndex >= 0 && projIndex < _visualBackboneLayers.Count)
            {
                x = _visualBackboneLayers[projIndex].Forward(x);
                activations[$"Layer_{idx++}_{_visualBackboneLayers[projIndex].GetType().Name}"] = x.Clone();
            }
            visualSeq = x;
        }

        Tensor<T> seq;
        if (textSeq is not null && visualSeq is not null)
            // Same sequence-axis fusion as RunMultimodal: normalize both streams to [B, L, D] and
            // concatenate along the sequence axis (concatenating on axis 0 with unequal-rank streams
            // grows the batch dimension and leaves an uninitialized output tail).
            seq = Engine.TensorConcatenate(
                [AlignToBatchedSequence(visualSeq), AlignToBatchedSequence(textSeq)], axis: 1);
        else
            seq = textSeq ?? visualSeq ?? new Tensor<T>(new[] { 1, _hiddenDim });

        foreach (var layer in _transformerLayers)
        {
            seq = layer.Forward(seq);
            activations[$"Layer_{idx++}_{layer.GetType().Name}"] = seq.Clone();
        }
        return activations;
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _positionEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        _spatialPositionEmbeddings = Tensor<T>.CreateDefault([1024, _hiddenDim], NumOps.Zero);
        _visualPositionEmbeddings = Tensor<T>.CreateDefault([(ImageSize / 16) * (ImageSize / 16), _hiddenDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_positionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_spatialPositionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_visualPositionEmbeddings, random, 0.02);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data.Span[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
    }

    #endregion

    #region ILayoutDetector Implementation

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage)
    {
        return DetectLayout(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var regions = ParseLayoutOutput(output, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<LayoutRegion<T>> ParseLayoutOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<LayoutRegion<T>>();
        int numDetections = output.Shape[0];
        int numClasses = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;

        for (int i = 0; i < numDetections; i++)
        {
            double maxConf = 0;
            int maxClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                double conf = NumOps.ToDouble(output[i, c]);
                if (conf > maxConf) { maxConf = conf; maxClass = c; }
            }

            if (maxConf >= threshold && maxClass > 0)
            {
                regions.Add(new LayoutRegion<T>
                {
                    ElementType = (LayoutElementType)Math.Min(maxClass, (int)LayoutElementType.Other),
                    Confidence = NumOps.FromDouble(maxConf),
                    ConfidenceValue = maxConf,
                    Index = i,
                    BoundingBox = Vector<T>.Empty()
                });
            }
        }

        return regions;
    }

    #endregion

    #region IDocumentQA Implementation

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question)
    {
        return AnswerQuestion(documentImage, question, 64, 0.0);
    }

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var (answer, confidence) = ExtractAnswer(output, maxAnswerLength);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
            Question = question,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <summary>
    /// Extracts answer from model output using extractive QA approach.
    /// </summary>
    private (string answer, double confidence) ExtractAnswer(Tensor<T> output, int maxAnswerLength)
    {
        int seqLen = output.Shape[0];
        int hiddenDim = output.Shape.Length > 1 ? output.Shape[1] : _hiddenDim;

        double bestStartScore = double.MinValue;
        double bestEndScore = double.MinValue;
        int bestStart = 0;
        int bestEnd = 0;

        for (int i = 0; i < seqLen; i++)
        {
            double startScore = NumOps.ToDouble(output[i, 0]);
            if (startScore > bestStartScore)
            {
                bestStartScore = startScore;
                bestStart = i;
            }
        }

        int endSearchLimit = Math.Min(seqLen, bestStart + maxAnswerLength);
        for (int i = bestStart; i < endSearchLimit; i++)
        {
            double endScore = NumOps.ToDouble(output[i, Math.Min(1, hiddenDim - 1)]);
            if (endScore > bestEndScore)
            {
                bestEndScore = endScore;
                bestEnd = i;
            }
        }

        var tokens = new List<int>();
        for (int i = bestStart; i <= bestEnd && i < seqLen; i++)
        {
            double maxVal = double.MinValue;
            int maxIdx = 0;
            for (int j = 0; j < Math.Min(hiddenDim, _vocabSize); j++)
            {
                double val = NumOps.ToDouble(output[i, j]);
                if (val > maxVal) { maxVal = val; maxIdx = j; }
            }
            if (maxIdx > 0) tokens.Add(maxIdx);
        }

        string answer = DecodeTokensToText(tokens);
        double confidence = Math.Max(0, Math.Min(1, (bestStartScore + bestEndScore) / 2.0));

        return (string.IsNullOrEmpty(answer) ? "[No answer found]" : answer, confidence);
    }

    /// <summary>
    /// Decodes token IDs to text using BERT-style vocabulary.
    /// </summary>
    private static string DecodeTokensToText(List<int> tokens)
    {
        if (tokens.Count == 0) return string.Empty;

        var sb = new System.Text.StringBuilder();
        foreach (int token in tokens)
        {
            char c = token switch
            {
                >= 1000 and <= 1031 => (char)(token - 1000 + 32),
                >= 1032 and <= 1057 => (char)(token - 1032 + 65),
                >= 1058 and <= 1083 => (char)(token - 1058 + 97),
                >= 103 and <= 125 => (char)(token - 103 + 48),
                >= 126 and <= 151 => (char)(token - 126 + 65),
                >= 152 and <= 177 => (char)(token - 152 + 97),
                _ => (char)((token % 95) + 32)
            };
            sb.Append(c);
        }

        return sb.ToString();
    }

    /// <inheritdoc/>
    public IEnumerable<DocumentQAResult<T>> AnswerQuestions(Tensor<T> documentImage, IEnumerable<string> questions)
    {
        foreach (var q in questions)
            yield return AnswerQuestion(documentImage, q);
    }

    /// <inheritdoc/>
    public Dictionary<string, DocumentQAResult<T>> ExtractFields(Tensor<T> documentImage, IEnumerable<string> fieldPrompts)
    {
        var results = new Dictionary<string, DocumentQAResult<T>>();
        foreach (var field in fieldPrompts)
            results[field] = AnswerQuestion(documentImage, $"What is the {field}?");
        return results;
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("LayoutLMv2 Model Summary");
        sb.AppendLine("========================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: BERT + ResNeXt-FPN visual backbone");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Visual Backbone Channels: {_visualBackboneChannels}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Uses Visual Features: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies LayoutLMv2's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// LayoutLMv2 (Microsoft paper) uses ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image._shape);
        double[] means = [0.485, 0.456, 0.406];
        double[] stds = [0.229, 0.224, 0.225];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 0.5;
                double std = c < stds.Length ? stds[c] : 0.5;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        normalized.Data.Span[idx] = NumOps.FromDouble((NumOps.ToDouble(image.Data.Span[idx]) - mean) / std);
                    }
                }
            }
        }
        return normalized;
    }

    /// <summary>
    /// Applies LayoutLMv2's industry-standard postprocessing: pass-through (multimodal outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "LayoutLMv2",
            Description = "LayoutLMv2 with text, layout, and visual features (ACL 2021)",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "image_size", ImageSize },
                { "visual_backbone_channels", _visualBackboneChannels },
                { "num_classes", _numClasses },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = SafeSerialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_vocabSize);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_visualBackboneChannels);
        writer.Write(_numClasses);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int visualChannels = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;

        // Re-link the two-stream forward sublists to the layers the base just deserialized (the forward
        // reads _visualBackboneLayers/_textEmbeddingLayers/_transformerLayers, not Layers directly).
        if (Layers.Count > 0)
            DistributeLayers();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LayoutLMv2<T>(Architecture, _tokenizer, _numClasses, ImageSize, MaxSequenceLength,
            _hiddenDim, _numLayers, _numHeads, _vocabSize, _visualBackboneChannels);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        if (!_useNativeMode)
            return RunOnnxInference(preprocessed);

        // Route through the real two-stream forward (RunMultimodal) instead of the base linear walk,
        // which would feed the visual conv backbone the text tokens (or vice versa). A single input is
        // disambiguated by rank: rank <= 2 = token IDs (text-only), rank >= 3 = document image.
        var (tokens, image) = preprocessed.Rank <= 2 ? (preprocessed, (Tensor<T>?)null) : ((Tensor<T>?)null, preprocessed);
        return RunMultimodal(tokens, image);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        // TrainWithTape runs the full forward (ForwardForTraining -> the two-stream RunMultimodal),
        // backprop through the autodiff tape, and the optimizer parameter update. The previous code
        // ALSO ran a manual UpdateParameters(CollectGradients()) afterwards — a redundant second
        // gradient-descent step whose hand-collected gradient vector didn't match GetParameters'
        // length (Expected N params, got N+12288), crashing every training step. Use the tape path only.
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.00005);
        
        currentParams = Engine.Subtract(currentParams, Engine.Multiply(gradients, lr));
        SetParameters(currentParams);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
