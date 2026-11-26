using AiDotNet.LearningRateSchedulers;
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
using AiDotNet.Models.Options;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.LayoutAware;

/// <summary>
/// LiLT (Language-Independent Layout Transformer) for document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LiLT separates the text and layout modalities during pre-training, enabling
/// the layout model to be combined with ANY pre-trained text model at fine-tuning
/// time, providing true language independence.
/// </para>
/// <para>
/// <b>For Beginners:</b> LiLT is designed for maximum flexibility:
/// 1. Layout understanding is learned separately from text
/// 2. Can plug in ANY language model (BERT, RoBERTa, XLM-R, etc.)
/// 3. Supports any language without retraining the layout part
///
/// Key features:
/// - BiACM (Bi-directional Attention Complementation Mechanism)
/// - Separate text and layout streams
/// - Works with any pre-trained text encoder
/// - Language-agnostic layout understanding
///
/// Example usage:
/// <code>
/// var model = new LiLT&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "LiLT: A Simple yet Effective Language-Independent Layout Transformer" (ACL 2022)
/// https://arxiv.org/abs/2202.13669
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding", "https://doi.org/10.48550/arXiv.2202.13669", Year = 2022, Authors = "Jiapeng Wang, Lianwen Jin, Kai Ding")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-5, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0.01, ReferenceBatchSize = 96,
                Source = "Wang et al. 2022, Sec. 4.1: LiLT BASE is pre-trained with Adam at a learning "
                        + "rate of 2e-5, a weight decay of 1e-2 and betas (0.9, 0.999), at a batch size "
                        + "of 96 for 5 epochs.")]
public partial class LiLT<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>
{
    private readonly LiLTOptions _options;

    #region Fields

    private bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _hiddenDim;
    private int _numLayers;
    private int _numHeads;
    private int _vocabSize;
    private int _numClasses;
    private string _textBackbone;

    // Native mode layers - separate streams
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private readonly List<ILayer<T>> _layoutEncoderLayers = [];
    private readonly List<ILayer<T>> _biACMLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    // Learnable embeddings
    // The text-position, layout-position and spatial coordinate tables used to be model fields
    // here. They are now inside the two LayoutEmbeddingLayers at the front of the stack, where the
    // forward pass reads them -- see LayerHelper.CreateDefaultLiLTLayers.

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the text backbone model name.
    /// </summary>
    public string TextBackbone => _textBackbone;

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
    /// Creates a LiLT model using a pre-trained ONNX model for inference.
    /// </summary>
    public LiLT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        LiLTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new LiLTOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _numLayers = _options.NumLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _textBackbone = _options.TextBackbone;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 1e-4 });

        MaxSequenceLength = _options.MaxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a LiLT model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (LiLT-Base from ACL 2022):</b>
    /// - Text encoder: Pluggable (default: BERT-base)
    /// - Layout encoder: Separate transformer
    /// - BiACM: Bi-directional attention between streams
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// </para>
    /// </remarks>
    public LiLT(
        NeuralNetworkArchitecture<T> architecture,
        LiLTOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new LiLTOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _numClasses = _options.NumClasses;
        _hiddenDim = _options.HiddenDim;
        _numLayers = _options.NumLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _textBackbone = _options.TextBackbone;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 1e-4 });

        MaxSequenceLength = _options.MaxSequenceLength;

        _tokenizer = tokenizer ?? CreateTokenizerForBackbone(_options.TextBackbone, _options.VocabSize);

        InitializeLayers();
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultLiLTLayers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            numClasses: _numClasses,
            maxPosition2D: 1024));
    }

    private static ITokenizer CreateTokenizerForBackbone(string textBackbone, int vocabSize)
    {
        if (string.IsNullOrWhiteSpace(textBackbone))
        {
            return LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
        }

        if (TryParseBackbone(textBackbone, out var backbone))
        {
            return LanguageModelTokenizerFactory.CreateForBackbone(backbone);
        }

        string normalized = NormalizeTextBackbone(textBackbone);
        try
        {
            return AutoTokenizer.FromPretrained(normalized);
        }
        catch (Exception)
        {
            return WordPieceTokenizer.Train(GetDefaultTokenizerCorpus(), vocabSize, SpecialTokens.Bert());
        }
    }

    private static bool TryParseBackbone(string textBackbone, out LanguageModelBackbone backbone)
    {
        return Enum.TryParse(textBackbone, true, out backbone);
    }

    private static string NormalizeTextBackbone(string textBackbone)
    {
        if (string.Equals(textBackbone, "bert-base", StringComparison.OrdinalIgnoreCase))
        {
            return "bert-base-uncased";
        }

        if (string.Equals(textBackbone, "bert-large", StringComparison.OrdinalIgnoreCase))
        {
            return "bert-large-uncased";
        }

        return textBackbone;
    }

    private static IEnumerable<string> GetDefaultTokenizerCorpus()
    {
        return new[]
        {
            "a photo of a document",
            "invoice total amount",
            "table row column header",
            "page number and date",
            "signature and stamp",
            "summary section",
            "this is a test",
            "layout understanding",
            "document classification"
        };
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

        var regions = ParseLayoutOutput(output, documentImage, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<LayoutRegion<T>> ParseLayoutOutput(Tensor<T> output, Tensor<T> documentImage, double threshold)
    {
        var regions = new List<LayoutRegion<T>>();
        if (output.Shape.Length < 1)
        {
            return regions;
        }

        var layoutOutput = NormalizeLayoutOutput(output);
        if (layoutOutput.Shape.Length < 2)
        {
            return regions;
        }

        int numDetections = layoutOutput.Shape[0];
        int numValues = layoutOutput.Shape[1];

        if (numDetections <= 0 || numValues <= 0)
        {
            return regions;
        }

        int numClasses = Math.Min(_numClasses, numValues);
        bool hasBbox = false;
        if (numValues > _numClasses && numValues >= 5)
        {
            numClasses = Math.Min(_numClasses, numValues - 4);
            hasBbox = numValues - 4 > 0;
        }

        GetImageDimensions(documentImage, out int imageWidth, out int imageHeight);
        if (numClasses <= 0)
        {
            return regions;
        }

        var (maxClass, maxConf) = DocumentTensorOps.RowMax(Engine, layoutOutput, numClasses);
        var boxes = hasBbox
            ? ExtractBoundingBoxes(layoutOutput, numValues, imageWidth, imageHeight)
            : EstimateGridBoundingBoxes(numDetections, imageWidth, imageHeight);

        // A detection is kept when its best class is not the background, reaches the threshold and its box
        // has a positive area.
        var x1 = Column(boxes, 0);
        var y1 = Column(boxes, 1);
        var x2 = Column(boxes, 2);
        var y2 = Column(boxes, 3);
        var keep = Engine.TensorMultiply(
            Engine.TensorMultiply(
                Engine.TensorGreaterThan(DocumentTensorOps.IntTensor<T>(maxClass), NumOps.Zero),
                DocumentTensorOps.AtLeast(Engine, maxConf, NumOps.FromDouble(threshold))),
            Engine.TensorMultiply(Engine.TensorGreaterThan(x2, x1), Engine.TensorGreaterThan(y2, y1)));
        int[] kept = DocumentTensorOps.Hits(Engine, keep);
        T[] confidence = DocumentTensorOps.Pick(Engine, maxConf, kept);
        T[] corners = boxes.ToArray();

        regions.AddRange(kept.Select((i, k) => new LayoutRegion<T>
        {
            ElementType = MapElementType(maxClass[i]),
            Confidence = confidence[k],
            Index = i,
            BoundingBox = new Vector<T>(new[] { corners[4 * i], corners[(4 * i) + 1], corners[(4 * i) + 2], corners[(4 * i) + 3] })
        }));

        return regions;
    }

    /// <summary>Column <paramref name="c"/> of a <c>[N, 4]</c> box tensor as a 1-D tensor.</summary>
    private Tensor<T> Column(Tensor<T> boxes, int c)
        => Engine.Reshape(Engine.TensorSlice(boxes, new[] { 0, c }, new[] { boxes.Shape[0], 1 }), new[] { boxes.Shape[0] });

    private Tensor<T> NormalizeLayoutOutput(Tensor<T> output)
    {
        if (output.Shape.Length == 2)
        {
            return output;
        }

        if (output.Shape.Length == 3 && output.Shape[0] == 1)
        {
            return output.Reshape([output.Shape[1], output.Shape[2]]);
        }

        if (output.Shape.Length == 1)
        {
            int expectedWithBbox = _numClasses + 4;
            if (expectedWithBbox > 0 && output.Length % expectedWithBbox == 0)
            {
                return output.Reshape([output.Length / expectedWithBbox, expectedWithBbox]);
            }

            if (_numClasses > 0 && output.Length % _numClasses == 0)
            {
                return output.Reshape([output.Length / _numClasses, _numClasses]);
            }

            return output.Reshape([1, output.Length]);
        }

        int lastDim = output.Shape[^1];
        if (lastDim <= 0 || output.Length % lastDim != 0)
        {
            return output.Reshape([1, output.Length]);
        }

        int numDetections = output.Length / lastDim;
        return output.Reshape([numDetections, lastDim]);
    }

    private LayoutElementType MapElementType(int classIndex)
    {
        if (classIndex <= 0)
        {
            return LayoutElementType.Other;
        }

        int supportedIndex = classIndex - 1;
        if (supportedIndex >= 0 && supportedIndex < SupportedElementTypes.Count)
        {
            return SupportedElementTypes[supportedIndex];
        }

        return LayoutElementType.Other;
    }

    /// <summary>
    /// The <c>[N, 4]</c> pixel boxes in the last four columns of the detections: read as corners, or as
    /// centre and size when the corners do not span a positive area, scaled to the page when every value lies
    /// in [-0.5, 1.5] (normalized coordinates), and clamped to the page.
    /// </summary>
    private Tensor<T> ExtractBoundingBoxes(Tensor<T> output, int numValues, int imageWidth, int imageHeight)
    {
        int n = output.Shape[0];
        var raw = Engine.TensorSlice(output, new[] { 0, numValues - 4 }, new[] { n, 4 });
        var one = NumOps.One;
        var below = Engine.TensorLessThan(raw, NumOps.FromDouble(-0.5));
        var above = Engine.TensorGreaterThan(raw, NumOps.FromDouble(1.5));
        var outside = Engine.ReduceSum(Engine.TensorAdd(below, above), new[] { 1 }, false);
        var normalized = Engine.TensorEquals(outside, NumOps.Zero);

        var b0 = Column(raw, 0);
        var b1 = Column(raw, 1);
        var b2 = Column(raw, 2);
        var b3 = Column(raw, 3);
        var spans = Engine.TensorMultiply(Engine.TensorGreaterThan(b2, b0), Engine.TensorGreaterThan(b3, b1));
        var half = NumOps.FromDouble(0.5);
        var halfWidth = Engine.TensorMultiplyScalar(Engine.TensorAbs(b2), half);
        var halfHeight = Engine.TensorMultiplyScalar(Engine.TensorAbs(b3), half);
        var x1 = Engine.TensorWhere(spans, b0, Engine.TensorSubtract(b0, halfWidth));
        var y1 = Engine.TensorWhere(spans, b1, Engine.TensorSubtract(b1, halfHeight));
        var x2 = Engine.TensorWhere(spans, b2, Engine.TensorAdd(b0, halfWidth));
        var y2 = Engine.TensorWhere(spans, b3, Engine.TensorAdd(b1, halfHeight));

        T width = NumOps.FromDouble(imageWidth);
        T height = NumOps.FromDouble(imageHeight);
        var ones = Tensor<T>.CreateDefault(new[] { n }, one);
        var xScale = Engine.TensorWhere(normalized, Tensor<T>.CreateDefault(new[] { n }, width), ones);
        var yScale = Engine.TensorWhere(normalized, Tensor<T>.CreateDefault(new[] { n }, height), ones);
        var columns = new[]
        {
            Engine.TensorClamp(Engine.TensorMultiply(x1, xScale), NumOps.Zero, width),
            Engine.TensorClamp(Engine.TensorMultiply(y1, yScale), NumOps.Zero, height),
            Engine.TensorClamp(Engine.TensorMultiply(x2, xScale), NumOps.Zero, width),
            Engine.TensorClamp(Engine.TensorMultiply(y2, yScale), NumOps.Zero, height)
        };
        return Engine.TensorConcatenate(columns.Select(c => Engine.Reshape(c, new[] { n, 1 })).ToArray(), 1);
    }

    /// <summary>
    /// The <c>[N, 4]</c> boxes of the cells of a square grid over the page, one per detection in row-major order.
    /// </summary>
    private Tensor<T> EstimateGridBoundingBoxes(int numDetections, int imageWidth, int imageHeight)
    {
        int gridSize = (int)Math.Ceiling(Math.Sqrt(numDetections));
        if (gridSize <= 0)
        {
            gridSize = 1;
        }

        int cellWidth = Math.Max(1, imageWidth / gridSize);
        int cellHeight = Math.Max(1, imageHeight / gridSize);
        int[] corners = Enumerable.Range(0, numDetections).SelectMany(index =>
        {
            int x1 = (index % gridSize) * cellWidth;
            int y1 = (index / gridSize) * cellHeight;
            return new[] { x1, y1, Math.Min(imageWidth, x1 + cellWidth), Math.Min(imageHeight, y1 + cellHeight) };
        }).ToArray();
        return Engine.Reshape(DocumentTensorOps.IntTensor<T>(corners), new[] { numDetections, 4 });
    }

    private void GetImageDimensions(Tensor<T> image, out int width, out int height)
    {
        if (image.Rank == 4)
        {
            height = image.Shape[2];
            width = image.Shape[3];
        }
        else if (image.Rank == 3)
        {
            height = image.Shape[1];
            width = image.Shape[2];
        }
        else
        {
            height = ImageSize;
            width = ImageSize;
        }

        if (height <= 0 || width <= 0)
        {
            height = ImageSize;
            width = ImageSize;
        }
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

        // Extract answer using extractive QA approach
        var (answer, confidence) = ExtractAnswer(output, maxAnswerLength, temperature);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = confidence,
            Question = question,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <summary>
    /// Extracts answer from model output using a token-probability decoding pass.
    /// </summary>
    private (string answer, T confidence) ExtractAnswer(Tensor<T> output, int maxAnswerLength, double temperature)
    {
        var logits = NormalizeAnswerLogits(output);
        if (logits.Shape.Length < 2)
        {
            return ("[No answer found]", NumOps.Zero);
        }

        int seqLen = logits.Shape[0];
        int vocabSize = logits.Shape[1];
        if (_vocabSize > 0)
        {
            vocabSize = Math.Min(vocabSize, _vocabSize);
        }

        if (seqLen <= 0 || vocabSize <= 0)
        {
            return ("[No answer found]", NumOps.Zero);
        }

        if (maxAnswerLength <= 0)
        {
            return ("[No answer found]", NumOps.Zero);
        }

        int maxLen = Math.Min(seqLen, maxAnswerLength);
        var tokens = new List<int>(maxLen);
        var answerSteps = new List<int>(maxLen);

        int eosId = GetSpecialTokenId(_tokenizer.SpecialTokens.EosToken);
        int sepId = GetSpecialTokenId(_tokenizer.SpecialTokens.SepToken);
        int padId = GetSpecialTokenId(_tokenizer.SpecialTokens.PadToken);
        int clsId = GetSpecialTokenId(_tokenizer.SpecialTokens.ClsToken);

        var steps = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { maxLen, vocabSize });
        var (stepTokens, probabilities) = temperature > 0.0
            ? DocumentTensorOps.SampleWithProbabilities(Engine, steps, NumOps.FromDouble(temperature), RandomHelper.Shared.Next())
            : DocumentTensorOps.GreedyWithProbabilities(Engine, steps);

        for (int i = 0; i < maxLen; i++)
        {
            int tokenId = stepTokens[i];
            if ((eosId >= 0 && tokenId == eosId) || (sepId >= 0 && tokenId == sepId))
            {
                break;
            }

            if ((padId >= 0 && tokenId == padId) || (clsId >= 0 && tokenId == clsId))
            {
                continue;
            }

            tokens.Add(tokenId);
            answerSteps.Add(i);
        }

        if (tokens.Count == 0)
        {
            return ("[No answer found]", NumOps.Zero);
        }

        string answer = _tokenizer.Decode(tokens, skipSpecialTokens: true).Trim();
        if (string.IsNullOrWhiteSpace(answer))
        {
            return ("[No answer found]", NumOps.Zero);
        }

        return (answer, DocumentTensorOps.Mean(Engine, DocumentTensorOps.Pick(Engine, probabilities, answerSteps.ToArray())));
    }

    private Tensor<T> NormalizeAnswerLogits(Tensor<T> output)
    {
        if (output.Shape.Length == 2)
        {
            return output;
        }

        if (output.Shape.Length == 3 && output.Shape[0] == 1)
        {
            return output.Reshape([output.Shape[1], output.Shape[2]]);
        }

        if (output.Shape.Length == 1)
        {
            if (_vocabSize > 0 && output.Length % _vocabSize == 0)
            {
                return output.Reshape([output.Length / _vocabSize, _vocabSize]);
            }

            return output.Reshape([1, output.Length]);
        }

        int lastDim = output.Shape[^1];
        if (lastDim <= 0 || output.Length % lastDim != 0)
        {
            return output.Reshape([1, output.Length]);
        }

        int seqLen = output.Length / lastDim;
        return output.Reshape([seqLen, lastDim]);
    }

    private int GetSpecialTokenId(string token)
    {
        if (string.IsNullOrWhiteSpace(token))
        {
            return -1;
        }

        var vocabulary = _tokenizer.Vocabulary;
        if (!vocabulary.ContainsToken(token))
        {
            return -1;
        }

        return vocabulary.GetTokenId(token);
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
        sb.AppendLine("LiLT Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Dual-stream with BiACM");
        sb.AppendLine($"Text Backbone: {_textBackbone}");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Language Independent: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies LiLT's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// LiLT (Language-independent Layout Transformer) uses ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
        => DocumentTensorOps.NormalizeChannels(Engine, EnsureBatchDimension(rawImage),
            DocumentTensorOps.ImageNetMean, DocumentTensorOps.ImageNetStd);

    /// <summary>
    /// Applies LiLT's industry-standard postprocessing: pass-through (layout-aware outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "LiLT",
            Description = "LiLT for language-independent layout understanding (ACL 2022)",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "max_sequence_length", MaxSequenceLength },
                { "num_classes", _numClasses },
                { "text_backbone", _textBackbone },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    // ── Faithful LiLT dual-stream forward with BiACM (Wang et al. 2022, §3.2) ──────────────────────────
    // Layer layout from CreateDefaultLiLTLayers: [0] = text LayoutEmbeddingLayer (word + learned 1D position,
    // no layout terms), [1] = LiltLayoutEmbeddingLayer (the reference coordinate tables + LayerNorm), then
    // numLayers blocks of 16 layers:
    //   text:   [q, k, v, o, attnLN, ffn1, ffn2, ffnLN]   width hidden
    //   layout: [q, k, v, o, attnLN, ffn1, ffn2, ffnLN]   width hidden / channel_shrink_ratio (reference LiltLayer)
    // and finally [Dropout, Dense(head)] on the text flow.
    private const int LiLTPrefixLayers = 2;
    private const int LiLTBlockLayers = 16;
    private const int LiLTChannelShrinkRatio = 4;
    private int HeadDim => _hiddenDim / _numHeads;
    private int LayoutDim => _hiddenDim / LiLTChannelShrinkRatio;
    private int LayoutHeadDim => LayoutDim / _numHeads;

    // BiACM is LiLT's contribution: the two flows SHARE attention scores (reference LiltSelfAttention). Each
    // flow's raw scores are scaled by its own head width. The text flow uses text + layout, and the layout
    // flow uses layout + DETACH(text), as the paper describes: the reusable text encoder is not perturbed
    // by the layout branch's gradient.
    private Tensor<T> RunDualStream(Tensor<T> textTokens, Tensor<T>? layoutBoxes)
    {
        var text = Layers[0].Forward(textTokens);

        // No boxes means text-only operation (graceful degradation the reference model lacks). BiACM then
        // reduces to a standard text self-attention transformer.
        Tensor<T>? layout = layoutBoxes is null ? null : Layers[1].Forward(layoutBoxes);

        int numBlocks = (Layers.Count - LiLTPrefixLayers - 2) / LiLTBlockLayers;
        for (int b = 0; b < numBlocks; b++)
            (text, layout) = BiACMBlock(text, layout, LiLTPrefixLayers + b * LiLTBlockLayers);

        // The classification head reads the TEXT flow (reference LiltForTokenClassification).
        int headStart = LiLTPrefixLayers + numBlocks * LiLTBlockLayers;
        var output = text;
        for (int i = headStart; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    private (Tensor<T> text, Tensor<T>? layout) BiACMBlock(Tensor<T> text, Tensor<T>? layout, int baseIdx)
    {
        int seqT = text.Shape[0];
        var st = Scores(Layers[baseIdx].Forward(text), Layers[baseIdx + 1].Forward(text), seqT, HeadDim);
        var vt = SplitHeads(Layers[baseIdx + 2].Forward(text), seqT, HeadDim);

        Tensor<T>? sl = null, vl = null;
        int seqL = 0, lb = baseIdx + 8;
        if (layout is not null)
        {
            seqL = layout.Shape[0];
            sl = Scores(Layers[lb].Forward(layout), Layers[lb + 1].Forward(layout), seqL, LayoutHeadDim);
            vl = SplitHeads(Layers[lb + 2].Forward(layout), seqL, LayoutHeadDim);
        }

        // BiACM needs token-aligned flows (one box per token) so the [heads, seq, seq] score matrices are
        // addable, which is the paper's assumption. With mismatched lengths the flows run independently.
        bool coupled = sl is not null && seqL == seqT;
        var textScores = sl is not null && seqL == seqT ? Engine.TensorAdd(st, sl) : st;
        var textContext = Layers[baseIdx + 3].Forward(MergeHeads(Engine.BatchMatMul(Engine.Softmax(textScores, axis: -1), vt), seqT, _hiddenDim));
        text = Layers[baseIdx + 4].Forward(Engine.TensorAdd(text, textContext));
        text = Layers[baseIdx + 7].Forward(Engine.TensorAdd(text, Layers[baseIdx + 6].Forward(Layers[baseIdx + 5].Forward(text))));

        if (layout is not null && sl is not null && vl is not null)
        {
            var layoutScores = coupled ? Engine.TensorAdd(sl, Engine.StopGradient(st)) : sl;
            var layoutContext = Layers[lb + 3].Forward(MergeHeads(Engine.BatchMatMul(Engine.Softmax(layoutScores, axis: -1), vl), seqL, LayoutDim));
            layout = Layers[lb + 4].Forward(Engine.TensorAdd(layout, layoutContext));
            layout = Layers[lb + 7].Forward(Engine.TensorAdd(layout, Layers[lb + 6].Forward(Layers[lb + 5].Forward(layout))));
        }
        return (text, layout);
    }

    // Scaled per-head scores q k^T / sqrt(headDim), [heads, seq, seq], from already-projected q and k.
    private Tensor<T> Scores(Tensor<T> query, Tensor<T> key, int seq, int headDim)
    {
        var q = SplitHeads(query, seq, headDim);
        var k = SplitHeads(key, seq, headDim);
        return Engine.TensorMultiplyScalar(Engine.BatchMatMul(q, Engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(headDim)));
    }

    private Tensor<T> SplitHeads(Tensor<T> x, int seq, int headDim)
        => Engine.TensorPermute(Engine.Reshape(x, new[] { seq, _numHeads, headDim }), new[] { 1, 0, 2 });

    private Tensor<T> MergeHeads(Tensor<T> x, int seq, int width)
        => Engine.Reshape(Engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { seq, width });

    /// <summary>
    /// Full dual-stream fusion entry (industry-standard LiLT): encodes text token IDs AND their layout
    /// bounding boxes through the coupled BiACM transformer. <paramref name="layoutBoxes"/> is the
    /// per-token box feature tensor the layout Dense consumes; pass null for text-only operation.
    /// </summary>
    public Tensor<T> EncodeDualStream(Tensor<T> textTokens, Tensor<T>? layoutBoxes)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return RunDualStream(textTokens, layoutBoxes);
    }

    // LiLT is a text + layout model: a rank-1/2 token-ID input drives the BiACM dual-stream (text-only
    // when no boxes accompany it). A higher-rank input (e.g. a raw image handed to Predict) isn't LiLT's
    // modality, so defer to the base sequential walk rather than forcing it through the token path.
    /// <inheritdoc/>
    protected override Tensor<T> Forward(Tensor<T> input)
        => RouteDualStream(input, trainingPath: false);

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => RouteDualStream(input, trainingPath: true);

    /// <summary>
    /// Sends an input through the BiACM dual stream, unpacking bounding boxes when they are present.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both forwards used to pass <c>null</c> for boxes. The layout stream was therefore unreachable
    /// from <c>Predict</c> and from <c>Train</c> alike -- LiLT could only ever run, and only ever
    /// learn, as a plain text transformer, which is the one thing the paper is not about. The only
    /// entry point that accepted boxes, <see cref="EncodeDualStream"/>, opens a <c>NoGradScope</c>
    /// and so could never train the layout side either.
    /// </para>
    /// <para>
    /// A packed <c>[seq, 5]</c> row -- <c>(tokenId, x0, y0, x1, y1)</c>, the convention the rest of
    /// this family uses -- is split here and both halves reach the streams, on the training path as
    /// well as the inference one. A bare token sequence still runs text-only, so callers with no OCR
    /// boxes are unaffected. A higher-rank input is not LiLT's modality and defers to the base walk.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Reports per-stage activations by running LiLT's actual dual stream.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The base implementation walks <c>Layers</c> as a chain, feeding each layer the previous one's
    /// output. LiLT is not a chain: <c>Layers[0]</c> embeds text, <c>Layers[1..2]</c> embed layout
    /// from bounding boxes, and the blocks after them consume BOTH. Walking it linearly handed the
    /// layout embedding a text activation, which the old <c>Dense(box-&gt;hidden)</c> silently accepted
    /// and multiplied -- so the reported "activations" were arithmetic on mismatched tensors that
    /// corresponded to nothing the model computes. The coordinate lookup that replaced the Dense
    /// rejects it outright, which is the correct response to being handed the wrong stream.
    /// </para>
    /// <para>
    /// Running the real dual stream reports what the model actually produces, and names the two
    /// streams so a caller can tell them apart.
    /// </para>
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || input.Rank > 2)
        {
            return base.GetNamedLayerActivations(input);
        }

        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

        Tensor<T> tokens = input;
        Tensor<T>? boxes = null;
        if (TrySplitPackedLayout(input, out var splitTokens, out var splitBoxes))
        {
            tokens = splitTokens;
            boxes = splitBoxes;
        }

        var activations = new Dictionary<string, Tensor<T>>
        {
            ["text_embedding"] = Layers[0].Forward(tokens),
        };

        if (boxes is not null)
        {
            activations["layout_embedding"] = Layers[1].Forward(boxes);
        }

        activations["output"] = RunDualStream(tokens, boxes);
        return activations;
    }


    private Tensor<T> RouteDualStream(Tensor<T> input, bool trainingPath)
    {
        if (_useNativeMode && TrySplitPackedLayout(input, out var tokens, out var boxes))
        {
            return RunDualStream(tokens, boxes);
        }

        if (_useNativeMode && input.Rank <= 2)
        {
            return RunDualStream(input, null);
        }

        return trainingPath ? base.ForwardForTraining(input) : base.Forward(input);
    }

    /// <summary>
    /// Splits a packed <c>[seq, 5]</c> or <c>[batch, seq, 5]</c> tensor into token IDs and the
    /// <c>[.., 4]</c> box tensor the layout stream consumes. False when the input is not packed.
    /// </summary>
    private bool TrySplitPackedLayout(Tensor<T> input, out Tensor<T> tokens, out Tensor<T>? boxes)
    {
        tokens = input;
        boxes = null;

        int rank = input.Rank;
        if (rank < 2 || input.Shape[rank - 1] != NeuralNetworks.Layers.LayoutEmbeddingLayer<T>.PackedRowWidth)
            return false;

        int width = NeuralNetworks.Layers.LayoutEmbeddingLayer<T>.PackedRowWidth;
        int rows = input.Length / width;
        var leading = input.Shape.ToArray().Take(rank - 1).ToArray();
        var packed = Engine.Reshape(input, new[] { rows, width });
        tokens = Engine.Reshape(Engine.TensorSlice(packed, new[] { 0, 0 }, new[] { rows, 1 }), leading).Clone();
        boxes = Engine.Reshape(Engine.TensorSlice(packed, new[] { 0, 1 }, new[] { rows, LayoutBoxWidth }),
            leading.Concat(new[] { LayoutBoxWidth }).ToArray()).Clone();
        return true;
    }

    /// <summary>Coordinates per box: x0, y0, x1, y1.</summary>
    private const int LayoutBoxWidth = 4;

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.


    /// <summary>
    /// Parameters cannot be written while the model is backed by a loaded ONNX graph: the weights
    /// belong to that graph, not to this instance.
    /// </summary>
    /// <remarks>
    /// Replaces a hand-written throw that used to sit inside UpdateParameters. The base checks this
    /// on every mutating entry point rather than the one member the throw happened to guard, and
    /// reading -- ParameterCount and GetParameters -- stays available either way.
    /// </remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
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
