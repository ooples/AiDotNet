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
public class LiLT<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>
{
    private readonly LiLTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
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
    private Tensor<T>? _textPositionEmbeddings;
    private Tensor<T>? _layoutPositionEmbeddings;
    private Tensor<T>? _spatialEmbeddings;

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
        int numClasses = 7,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        string textBackbone = "bert-base",
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LiLTOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
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
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _textBackbone = textBackbone;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        MaxSequenceLength = maxSequenceLength;

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
        ITokenizer? tokenizer = null,
        int numClasses = 7,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        string textBackbone = "bert-base",
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LiLTOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new LiLTOptions();
        Options = _options;

        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _textBackbone = textBackbone;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        MaxSequenceLength = maxSequenceLength;

        _tokenizer = tokenizer ?? CreateTokenizerForBackbone(textBackbone, vocabSize);

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

        Layers.AddRange(LayerHelper<T>.CreateDefaultLiLTLayers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            layoutDim: _hiddenDim,
            vocabSize: _vocabSize,
            numClasses: _numClasses));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _textPositionEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        _layoutPositionEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        _spatialEmbeddings = Tensor<T>.CreateDefault([1024, _hiddenDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_textPositionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_layoutPositionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_spatialEmbeddings, random, 0.02);
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

        for (int i = 0; i < numDetections; i++)
        {
            double maxConf = double.MinValue;
            int maxClass = -1;
            int offset = i * numValues;

            for (int c = 0; c < numClasses; c++)
            {
                double conf = NumOps.ToDouble(layoutOutput.Data.Span[offset + c]);
                if (conf > maxConf)
                {
                    maxConf = conf;
                    maxClass = c;
                }
            }

            if (maxClass <= 0 || maxConf < threshold)
            {
                continue;
            }

            var elementType = MapElementType(maxClass);
            var bbox = hasBbox
                ? ExtractBoundingBox(layoutOutput, i, numValues, imageWidth, imageHeight)
                : EstimateGridBoundingBox(i, numDetections, imageWidth, imageHeight);

            if (bbox.Length == 4)
            {
                double x1 = NumOps.ToDouble(bbox[0]);
                double y1 = NumOps.ToDouble(bbox[1]);
                double x2 = NumOps.ToDouble(bbox[2]);
                double y2 = NumOps.ToDouble(bbox[3]);

                if (x2 <= x1 || y2 <= y1)
                {
                    continue;
                }
            }

            regions.Add(new LayoutRegion<T>
            {
                ElementType = elementType,
                Confidence = NumOps.FromDouble(maxConf),
                ConfidenceValue = maxConf,
                Index = i,
                BoundingBox = bbox
            });
        }

        return regions;
    }

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

    private Vector<T> ExtractBoundingBox(Tensor<T> output, int detectionIndex, int numValues, int imageWidth, int imageHeight)
    {
        int bboxOffset = detectionIndex * numValues + numValues - 4;
        double b0 = NumOps.ToDouble(output.Data.Span[bboxOffset]);
        double b1 = NumOps.ToDouble(output.Data.Span[bboxOffset + 1]);
        double b2 = NumOps.ToDouble(output.Data.Span[bboxOffset + 2]);
        double b3 = NumOps.ToDouble(output.Data.Span[bboxOffset + 3]);

        bool looksNormalized = b0 >= -0.5 && b0 <= 1.5 &&
                               b1 >= -0.5 && b1 <= 1.5 &&
                               b2 >= -0.5 && b2 <= 1.5 &&
                               b3 >= -0.5 && b3 <= 1.5;

        double x1 = b0;
        double y1 = b1;
        double x2 = b2;
        double y2 = b3;

        if (x2 <= x1 || y2 <= y1)
        {
            double cx = x1;
            double cy = y1;
            double w = Math.Abs(x2);
            double h = Math.Abs(y2);
            x1 = cx - w / 2.0;
            y1 = cy - h / 2.0;
            x2 = cx + w / 2.0;
            y2 = cy + h / 2.0;
        }

        if (looksNormalized)
        {
            x1 *= imageWidth;
            x2 *= imageWidth;
            y1 *= imageHeight;
            y2 *= imageHeight;
        }

        x1 = Clamp(x1, 0, imageWidth);
        x2 = Clamp(x2, 0, imageWidth);
        y1 = Clamp(y1, 0, imageHeight);
        y2 = Clamp(y2, 0, imageHeight);

        return new Vector<T>([
            NumOps.FromDouble(x1),
            NumOps.FromDouble(y1),
            NumOps.FromDouble(x2),
            NumOps.FromDouble(y2)
        ]);
    }

    private Vector<T> EstimateGridBoundingBox(int index, int numDetections, int imageWidth, int imageHeight)
    {
        int gridSize = (int)Math.Ceiling(Math.Sqrt(numDetections));
        if (gridSize <= 0)
        {
            gridSize = 1;
        }

        int cellWidth = Math.Max(1, imageWidth / gridSize);
        int cellHeight = Math.Max(1, imageHeight / gridSize);
        int row = index / gridSize;
        int col = index % gridSize;

        double x1 = col * cellWidth;
        double y1 = row * cellHeight;
        double x2 = Math.Min(imageWidth, x1 + cellWidth);
        double y2 = Math.Min(imageHeight, y1 + cellHeight);

        return new Vector<T>([
            NumOps.FromDouble(x1),
            NumOps.FromDouble(y1),
            NumOps.FromDouble(x2),
            NumOps.FromDouble(y2)
        ]);
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

    private static double Clamp(double value, double min, double max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
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
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
            Question = question,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <summary>
    /// Extracts answer from model output using a token-probability decoding pass.
    /// </summary>
    private (string answer, double confidence) ExtractAnswer(Tensor<T> output, int maxAnswerLength, double temperature)
    {
        var logits = NormalizeAnswerLogits(output);
        if (logits.Shape.Length < 2)
        {
            return ("[No answer found]", 0.0);
        }

        int seqLen = logits.Shape[0];
        int vocabSize = logits.Shape[1];
        if (_vocabSize > 0)
        {
            vocabSize = Math.Min(vocabSize, _vocabSize);
        }

        if (seqLen <= 0 || vocabSize <= 0)
        {
            return ("[No answer found]", 0.0);
        }

        if (maxAnswerLength <= 0)
        {
            return ("[No answer found]", 0.0);
        }

        int maxLen = Math.Min(seqLen, maxAnswerLength);
        var tokens = new List<int>(maxLen);
        double confidenceSum = 0.0;
        int confidenceCount = 0;

        int eosId = GetSpecialTokenId(_tokenizer.SpecialTokens.EosToken);
        int sepId = GetSpecialTokenId(_tokenizer.SpecialTokens.SepToken);
        int padId = GetSpecialTokenId(_tokenizer.SpecialTokens.PadToken);
        int clsId = GetSpecialTokenId(_tokenizer.SpecialTokens.ClsToken);

        var random = RandomHelper.Shared;
        bool sampleTokens = temperature > 0.0;

        for (int i = 0; i < maxLen; i++)
        {
            int offset = i * logits.Shape[1];
            int tokenId;
            double tokenProb;

            if (sampleTokens)
            {
                tokenId = SampleToken(logits, offset, vocabSize, temperature, random, out tokenProb);
            }
            else
            {
                tokenId = SelectGreedyToken(logits, offset, vocabSize, out tokenProb);
            }

            if ((eosId >= 0 && tokenId == eosId) || (sepId >= 0 && tokenId == sepId))
            {
                break;
            }

            if ((padId >= 0 && tokenId == padId) || (clsId >= 0 && tokenId == clsId))
            {
                continue;
            }

            tokens.Add(tokenId);
            confidenceSum += tokenProb;
            confidenceCount++;
        }

        if (tokens.Count == 0)
        {
            return ("[No answer found]", 0.0);
        }

        string answer = _tokenizer.Decode(tokens, skipSpecialTokens: true).Trim();
        if (string.IsNullOrWhiteSpace(answer))
        {
            return ("[No answer found]", 0.0);
        }

        double confidence = confidenceCount > 0 ? confidenceSum / confidenceCount : 0.0;
        return (answer, confidence);
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

    private int SelectGreedyToken(Tensor<T> logits, int offset, int vocabSize, out double probability)
    {
        double maxVal = double.MinValue;
        int maxIdx = 0;

        for (int v = 0; v < vocabSize; v++)
        {
            double val = NumOps.ToDouble(logits.Data.Span[offset + v]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = v;
            }
        }

        double sumExp = 0.0;
        for (int v = 0; v < vocabSize; v++)
        {
            double val = NumOps.ToDouble(logits.Data.Span[offset + v]);
            sumExp += Math.Exp(val - maxVal);
        }

        probability = sumExp > 0 ? 1.0 / sumExp : 0.0;
        return maxIdx;
    }

    private int SampleToken(Tensor<T> logits, int offset, int vocabSize, double temperature, Random random, out double probability)
    {
        double maxVal = double.MinValue;
        for (int v = 0; v < vocabSize; v++)
        {
            double scaled = NumOps.ToDouble(logits.Data.Span[offset + v]) / temperature;
            if (scaled > maxVal)
            {
                maxVal = scaled;
            }
        }

        double sumExp = 0.0;
        for (int v = 0; v < vocabSize; v++)
        {
            double scaled = NumOps.ToDouble(logits.Data.Span[offset + v]) / temperature;
            sumExp += Math.Exp(scaled - maxVal);
        }

        if (sumExp <= 0.0)
        {
            probability = 0.0;
            return 0;
        }

        double roll = random.NextDouble() * sumExp;
        double cumulative = 0.0;
        for (int v = 0; v < vocabSize; v++)
        {
            double scaled = NumOps.ToDouble(logits.Data.Span[offset + v]) / temperature;
            double expVal = Math.Exp(scaled - maxVal);
            cumulative += expVal;
            if (cumulative >= roll)
            {
                probability = expVal / sumExp;
                return v;
            }
        }

        probability = 0.0;
        return vocabSize - 1;
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
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);
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
            ModelType = ModelType.NeuralNetwork,
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
        writer.Write(MaxSequenceLength);
        writer.Write(_numClasses);
        writer.Write(_textBackbone);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        string textBackbone = reader.ReadString();
        bool useNativeMode = reader.ReadBoolean();

        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _numClasses = numClasses;
        _textBackbone = textBackbone;
        _useNativeMode = useNativeMode;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException(
                "Deep copy is not supported for ONNX LiLT instances. Create a new instance with model paths instead.");
        }
        return new LiLT<T>(Architecture, _tokenizer, _numClasses, MaxSequenceLength,
            _hiddenDim, _numLayers, _numHeads, _vocabSize, _textBackbone, _optimizer, LossFunction);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        var output = Predict(input);
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var gradient = Tensor<T>.FromVector(
            LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector()));

        for (int i = Layers.Count - 1; i >= 0; i--)
            gradient = Layers[i].Backward(gradient);

        UpdateParameters(CollectGradients());
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Parameter updates not supported in ONNX mode.");

        var currentParams = GetParameters();
        T lr = NumOps.FromDouble(0.00005);
        for (int i = 0; i < currentParams.Length; i++)
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(lr, gradients[i]));
        SetParameters(currentParams);
    }

    private Vector<T> CollectGradients()
    {
        var grads = new List<T>();
        foreach (var layer in Layers)
            grads.AddRange(layer.GetParameterGradients());
        return new Vector<T>([.. grads]);
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
