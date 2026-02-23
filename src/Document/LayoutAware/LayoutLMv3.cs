using AiDotNet.ActivationFunctions;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Document.LayoutAware;

/// <summary>
/// LayoutLMv3 neural network for document understanding with unified text and image pre-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutLMv3 is the third generation of the LayoutLM series from Microsoft Research,
/// featuring unified multimodal pre-training with masked image modeling and masked language
/// modeling on the same architecture.
/// </para>
/// <para>
/// <b>For Beginners:</b> LayoutLMv3 understands documents by learning from:
/// 1. The text content (what the words say)
/// 2. The visual appearance (what the document looks like)
/// 3. The layout structure (where elements are positioned)
///
/// This makes it excellent for:
/// - Extracting information from forms and receipts
/// - Understanding document structure
/// - Answering questions about document content
/// - Classifying document types
///
/// Example usage (ONNX mode - for inference with pre-trained models):
/// <code>
/// var model = new LayoutLMv3&lt;float&gt;(architecture, "model.onnx", tokenizer);
/// var layout = model.DetectLayout(documentImage);
/// </code>
///
/// Example usage (Native mode - for training):
/// <code>
/// var model = new LayoutLMv3&lt;float&gt;(architecture);
/// model.Train(trainingData, labels);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"
/// https://arxiv.org/abs/2204.08387
/// </para>
/// </remarks>
public class LayoutLMv3<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>
{
    private readonly LayoutLMv3Options _options;

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
    private readonly int _patchSize;

    // Native mode layers
    private readonly List<ILayer<T>> _textEmbeddingLayers = [];
    private readonly List<ILayer<T>> _imageEmbeddingLayers = [];
    private readonly List<ILayer<T>> _transformerLayers = [];
    private readonly List<ILayer<T>> _classificationLayers = [];

    // Learnable embeddings
    private Tensor<T>? _position1DEmbeddings;
    private Tensor<T>? _position2DXEmbeddings;
    private Tensor<T>? _position2DYEmbeddings;
    private Tensor<T>? _segmentEmbeddings;

    // Gradient storage
    private Tensor<T>? _position1DEmbeddingsGradients;
    private Tensor<T>? _position2DXEmbeddingsGradients;
    private Tensor<T>? _position2DYEmbeddingsGradients;

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
        LayoutElementType.PageNumber,
        LayoutElementType.FormField
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a LayoutLMv3 model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="numClasses">Number of output classes for classification tasks.</param>
    /// <param name="imageSize">Expected input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265 for RoBERTa).</param>
    /// <param name="optimizer">Optimizer for training (optional, Adam used if null).</param>
    /// <param name="lossFunction">Loss function (optional, CrossEntropy used if null).</param>
    /// <exception cref="ArgumentNullException">Thrown if onnxModelPath or tokenizer is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file doesn't exist.</exception>
    public LayoutLMv3(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numClasses = 17,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 50265,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LayoutLMv3Options? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutLMv3Options();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));

        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _patchSize = 16; // Default patch size for LayoutLMv3
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        Guard.Positive(imageSize, nameof(imageSize));
        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a LayoutLMv3 model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text processing (optional, default created if null).</param>
    /// <param name="numClasses">Number of output classes for classification tasks.</param>
    /// <param name="imageSize">Expected input image size (default: 224 from paper).</param>
    /// <param name="patchSize">Vision transformer patch size (default: 16 from paper).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension size (default: 768 for LayoutLMv3-Base).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12 for Base).</param>
    /// <param name="numHeads">Number of attention heads (default: 12 for Base).</param>
    /// <param name="vocabSize">Vocabulary size (default: 50265 for RoBERTa).</param>
    /// <param name="optimizer">Optimizer for training (optional, Adam used if null).</param>
    /// <param name="lossFunction">Loss function (optional, CrossEntropy used if null).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (LayoutLMv3-Base from ICCV 2022 paper):</b>
    /// - Hidden dimension: 768
    /// - Transformer layers: 12
    /// - Attention heads: 12
    /// - Image size: 224×224
    /// - Patch size: 16
    /// - Max sequence length: 512
    /// </para>
    /// </remarks>
    public LayoutLMv3(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numClasses = 17,
        int imageSize = 224,
        int patchSize = 16,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 50265,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LayoutLMv3Options? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutLMv3Options();
        Options = _options;

        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _patchSize = patchSize;

        Guard.Positive(imageSize, nameof(imageSize));
        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.RoBERTa);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
        InitializeEmbeddings();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // In ONNX mode, layers are handled by ONNX runtime
        if (!_useNativeMode)
        {
            return;
        }

        // Check if user provided custom layers
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            PopulateLayerGroups();
            return;
        }

        // Use LayerHelper to create default LayoutLMv3 layers
        Layers.AddRange(LayerHelper<T>.CreateDefaultLayoutLMv3Layers(
            Architecture,
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            patchSize: _patchSize,
            numClasses: _numClasses));
        PopulateLayerGroups();
    }

    private void PopulateLayerGroups()
    {
        _textEmbeddingLayers.Clear();
        _imageEmbeddingLayers.Clear();
        _transformerLayers.Clear();

        bool reachedHead = false;
        foreach (var layer in Layers)
        {
            if (IsClassificationHeadLayer(layer))
            {
                reachedHead = true;
            }

            if (reachedHead)
            {
                continue;
            }

            if (layer is EmbeddingLayer<T>)
            {
                _textEmbeddingLayers.Add(layer);
            }
            else if (layer is PatchEmbeddingLayer<T>)
            {
                _imageEmbeddingLayers.Add(layer);
            }
            else
            {
                _transformerLayers.Add(layer);
            }
        }
    }

    private static bool IsClassificationHeadLayer(ILayer<T> layer)
    {
        return layer is DenseLayer<T>;
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // 1D position embeddings for sequence position
        _position1DEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_position1DEmbeddings, random, 0.02);

        // 2D position embeddings for bounding box coordinates (normalized 0-1000)
        _position2DXEmbeddings = Tensor<T>.CreateDefault([1001, _hiddenDim], NumOps.Zero);
        _position2DYEmbeddings = Tensor<T>.CreateDefault([1001, _hiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_position2DXEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_position2DYEmbeddings, random, 0.02);

        // Segment embeddings (text vs image)
        _segmentEmbeddings = Tensor<T>.CreateDefault([2, _hiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_segmentEmbeddings, random, 0.02);

        // Initialize gradient tensors
        _position1DEmbeddingsGradients = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        _position2DXEmbeddingsGradients = Tensor<T>.CreateDefault([1001, _hiddenDim], NumOps.Zero);
        _position2DYEmbeddingsGradients = Tensor<T>.CreateDefault([1001, _hiddenDim], NumOps.Zero);
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

        var result = _useNativeMode
            ? DetectLayoutNative(documentImage, confidenceThreshold)
            : DetectLayoutOnnx(documentImage, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = result.Regions,
            ReadingOrder = result.ReadingOrder,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private DocumentLayoutResult<T> DetectLayoutNative(Tensor<T> image, double threshold)
    {
        var input = PreprocessDocument(image);
        var output = Forward(input);
        return ParseLayoutOutput(output, threshold);
    }

    private DocumentLayoutResult<T> DetectLayoutOnnx(Tensor<T> image, double threshold)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var input = PreprocessDocument(image);
        var output = RunOnnxInference(input);
        return ParseLayoutOutput(output, threshold);
    }

    private DocumentLayoutResult<T> ParseLayoutOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<LayoutRegion<T>>();

        // Parse output tensor into layout regions
        // Expected format: [numDetections, numClasses + 4] where last 4 are bbox coords
        // Or: [numDetections, hiddenDim] where we extract class from first numClasses and bbox from last 4

        int numDetections = output.Shape[0];
        int numValues = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;
        int numClasses = Math.Max(0, Math.Min(numValues - 4, _numClasses));
        bool hasBbox = numValues >= 4;

        if (numClasses == 0)
        {
            return new DocumentLayoutResult<T>
            {
                Regions = regions
            };
        }

        for (int i = 0; i < numDetections; i++)
        {
            // Find the class with highest confidence
            double maxConf = 0;
            int maxClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                double conf = NumOps.ToDouble(output.Data.Span[i * numValues + c]);
                if (conf > maxConf)
                {
                    maxConf = conf;
                    maxClass = c;
                }
            }

            if (maxConf >= threshold)
            {
                // Extract bounding box from last 4 values (normalized coordinates)
                Vector<T> bbox;
                if (hasBbox && numValues >= 4)
                {
                    int bboxOffset = i * numValues + numValues - 4;
                    double x1 = NumOps.ToDouble(output.Data.Span[bboxOffset]) * ImageSize;
                    double y1 = NumOps.ToDouble(output.Data.Span[bboxOffset + 1]) * ImageSize;
                    double x2 = NumOps.ToDouble(output.Data.Span[bboxOffset + 2]) * ImageSize;
                    double y2 = NumOps.ToDouble(output.Data.Span[bboxOffset + 3]) * ImageSize;

                    bbox = new Vector<T>([
                        NumOps.FromDouble(Math.Max(0, x1)),
                        NumOps.FromDouble(Math.Max(0, y1)),
                        NumOps.FromDouble(Math.Min(ImageSize, x2)),
                        NumOps.FromDouble(Math.Min(ImageSize, y2))
                    ]);
                }
                else
                {
                    // Estimate bbox from detection index (grid-based fallback)
                    int gridSize = (int)Math.Sqrt(numDetections);
                    int cellSize = ImageSize / Math.Max(1, gridSize);
                    int row = i / gridSize;
                    int col = i % gridSize;

                    bbox = new Vector<T>([
                        NumOps.FromDouble(col * cellSize),
                        NumOps.FromDouble(row * cellSize),
                        NumOps.FromDouble((col + 1) * cellSize),
                        NumOps.FromDouble((row + 1) * cellSize)
                    ]);
                }

                regions.Add(new LayoutRegion<T>
                {
                    ElementType = (LayoutElementType)Math.Min(maxClass, (int)LayoutElementType.Other),
                    Confidence = NumOps.FromDouble(maxConf),
                    ConfidenceValue = maxConf,
                    Index = i,
                    BoundingBox = bbox
                });
            }
        }

        return new DocumentLayoutResult<T>
        {
            Regions = regions
        };
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

        var result = _useNativeMode
            ? AnswerQuestionNative(documentImage, question, maxAnswerLength, temperature)
            : AnswerQuestionOnnx(documentImage, question, maxAnswerLength, temperature);

        return new DocumentQAResult<T>
        {
            Answer = result.Answer,
            Confidence = result.Confidence,
            ConfidenceValue = result.ConfidenceValue,
            Evidence = result.Evidence,
            AlternativeAnswers = result.AlternativeAnswers,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds,
            Question = question
        };
    }

    private DocumentQAResult<T> AnswerQuestionNative(Tensor<T> image, string question, int maxLength, double temp)
    {
        // Process image
        var imageFeatures = PreprocessDocument(image);

        // Combine and run through model
        var output = Forward(imageFeatures);

        var (answer, confidence) = ExtractAnswer(output, maxLength, temp);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence
        };
    }

    private DocumentQAResult<T> AnswerQuestionOnnx(Tensor<T> image, string question, int maxLength, double temp)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var imageFeatures = PreprocessDocument(image);
        var output = RunOnnxInference(imageFeatures);

        var (answer, confidence) = ExtractAnswer(output, maxLength, temp);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence
        };
    }

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

        if (seqLen <= 0 || vocabSize <= 0 || maxAnswerLength <= 0)
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
        ValidateImageShape(documentImage);

        foreach (var question in questions)
        {
            yield return AnswerQuestion(documentImage, question);
        }
    }

    /// <inheritdoc/>
    public Dictionary<string, DocumentQAResult<T>> ExtractFields(Tensor<T> documentImage, IEnumerable<string> fieldPrompts)
    {
        var results = new Dictionary<string, DocumentQAResult<T>>();

        foreach (var field in fieldPrompts)
        {
            var question = $"What is the {field}?";
            results[field] = AnswerQuestion(documentImage, question);
        }

        return results;
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);

        var input = PreprocessDocument(documentImage);

        if (_useNativeMode)
        {
            // Run through embedding and transformer layers only (not classification head)
            var output = input;
            if (_textEmbeddingLayers.Count == 0
                && _imageEmbeddingLayers.Count == 0
                && _transformerLayers.Count == 0)
            {
                foreach (var layer in Layers)
                {
                    if (IsClassificationHeadLayer(layer))
                    {
                        break;
                    }
                    output = layer.Forward(output);
                }
                return output;
            }

            foreach (var layer in _textEmbeddingLayers) output = layer.Forward(output);
            foreach (var layer in _imageEmbeddingLayers) output = layer.Forward(output);
            foreach (var layer in _transformerLayers) output = layer.Forward(output);
            return output;
        }
        else
        {
            return RunOnnxInference(input);
        }
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
        sb.AppendLine("LayoutLMv3 Model Summary");
        sb.AppendLine("========================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Number of Attention Heads: {_numHeads}");
        sb.AppendLine($"Vocabulary Size: {_vocabSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        sb.AppendLine($"Supported Document Types: {SupportedDocumentTypes}");
        sb.AppendLine($"Requires OCR: {RequiresOCR}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies LayoutLMv3's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// LayoutLMv3 (Microsoft paper) uses ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    /// The unified architecture for multimodal document understanding.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Resize to model's expected ImageSize if needed (bilinear interpolation)
        if (height != ImageSize || width != ImageSize)
        {
            image = ResizeBilinear(image, batchSize, channels, height, width, ImageSize, ImageSize);
            height = ImageSize;
            width = ImageSize;
        }

        // ImageNet normalization: (x - mean) / std
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
                        double value = NumOps.ToDouble(image.Data.Span[idx]);
                        normalized.Data.Span[idx] = NumOps.FromDouble((value - mean) / std);
                    }
                }
            }
        }

        return normalized;
    }

    private Tensor<T> ResizeBilinear(Tensor<T> image, int batchSize, int channels,
        int srcH, int srcW, int dstH, int dstW)
    {
        var resized = new Tensor<T>([batchSize, channels, dstH, dstW]);

        double scaleH = (double)srcH / dstH;
        double scaleW = (double)srcW / dstW;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int srcBaseIdx = b * channels * srcH * srcW + c * srcH * srcW;
                int dstBaseIdx = b * channels * dstH * dstW + c * dstH * dstW;

                for (int h = 0; h < dstH; h++)
                {
                    double srcY = (h + 0.5) * scaleH - 0.5;
                    int y0 = Math.Max(0, (int)Math.Floor(srcY));
                    int y1 = Math.Min(srcH - 1, y0 + 1);
                    double fy = srcY - y0;

                    for (int w = 0; w < dstW; w++)
                    {
                        double srcX = (w + 0.5) * scaleW - 0.5;
                        int x0 = Math.Max(0, (int)Math.Floor(srcX));
                        int x1 = Math.Min(srcW - 1, x0 + 1);
                        double fx = srcX - x0;

                        double v00 = NumOps.ToDouble(image.Data.Span[srcBaseIdx + y0 * srcW + x0]);
                        double v01 = NumOps.ToDouble(image.Data.Span[srcBaseIdx + y0 * srcW + x1]);
                        double v10 = NumOps.ToDouble(image.Data.Span[srcBaseIdx + y1 * srcW + x0]);
                        double v11 = NumOps.ToDouble(image.Data.Span[srcBaseIdx + y1 * srcW + x1]);

                        double val = v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx
                                   + v10 * fy * (1 - fx) + v11 * fy * fx;

                        resized.Data.Span[dstBaseIdx + h * dstW + w] = NumOps.FromDouble(val);
                    }
                }
            }
        }

        return resized;
    }

    /// <summary>
    /// Applies LayoutLMv3's industry-standard postprocessing: softmax for classification outputs.
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        // Apply softmax for classification outputs
        return ApplySoftmax(modelOutput);
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        int lastDim = input.Shape[^1];
        int numBatches = input.Data.Length / lastDim;

        for (int b = 0; b < numBatches; b++)
        {
            double maxVal = double.MinValue;
            for (int i = 0; i < lastDim; i++)
            {
                double val = NumOps.ToDouble(input.Data.Span[b * lastDim + i]);
                if (val > maxVal) maxVal = val;
            }

            double sumExp = 0;
            for (int i = 0; i < lastDim; i++)
            {
                double val = NumOps.ToDouble(input.Data.Span[b * lastDim + i]);
                sumExp += Math.Exp(val - maxVal);
            }

            for (int i = 0; i < lastDim; i++)
            {
                double val = NumOps.ToDouble(input.Data.Span[b * lastDim + i]);
                output.Data.Span[b * lastDim + i] = NumOps.FromDouble(Math.Exp(val - maxVal) / sumExp);
            }
        }

        return output;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "LayoutLMv3",
            ModelType = ModelType.NeuralNetwork,
            Description = "LayoutLMv3 document understanding model with unified text and image pre-training",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "max_seq_length", MaxSequenceLength },
                { "image_size", ImageSize },
                { "patch_size", _patchSize },
                { "num_classes", _numClasses },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
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
        writer.Write(ImageSize);
        writer.Write(_numClasses);
        writer.Write(_patchSize);
        writer.Write(_useNativeMode);

        // Serialize embeddings if in native mode
        if (_useNativeMode && _position1DEmbeddings is not null)
        {
            writer.Write(true);
            SerializeTensor(writer, _position1DEmbeddings);
            SerializeTensor(writer, _position2DXEmbeddings!);
            SerializeTensor(writer, _position2DYEmbeddings!);
            SerializeTensor(writer, _segmentEmbeddings!);
        }
        else
        {
            writer.Write(false);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read fields (Note: readonly fields are set in constructor, these would be for validation)
        int hiddenDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;

        // Deserialize embeddings if present
        if (reader.ReadBoolean())
        {
            _position1DEmbeddings = DeserializeTensor(reader);
            _position2DXEmbeddings = DeserializeTensor(reader);
            _position2DYEmbeddings = DeserializeTensor(reader);
            _segmentEmbeddings = DeserializeTensor(reader);
        }
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Rank);
        foreach (var dim in tensor.Shape)
            writer.Write(dim);

        writer.Write(tensor.Data.Length);
        foreach (var val in tensor.Data.ToArray())
            writer.Write(NumOps.ToDouble(val));
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        int[] shape = new int[rank];
        for (int i = 0; i < rank; i++)
            shape[i] = reader.ReadInt32();

        int length = reader.ReadInt32();
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < length; i++)
            tensor.Data.Span[i] = NumOps.FromDouble(reader.ReadDouble());

        return tensor;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LayoutLMv3<T>(
            Architecture,
            _tokenizer,
            _numClasses,
            ImageSize,
            _patchSize,
            MaxSequenceLength,
            _hiddenDim,
            _numLayers,
            _numHeads,
            _vocabSize);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Overrides Forward to handle LayoutLMv3's multimodal architecture.
    /// Image input is routed through image embedding (skipping text embedding),
    /// then through transformer layers and classification head.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        // If layer groups are populated, use modality-aware routing
        if (_imageEmbeddingLayers.Count > 0 || _transformerLayers.Count > 0)
        {
            var output = input;

            // Route through image embedding (skip text embedding for image input)
            foreach (var layer in _imageEmbeddingLayers)
                output = layer.Forward(output);

            // Route through transformer layers
            foreach (var layer in _transformerLayers)
                output = layer.Forward(output);

            // Route through classification head layers (all layers not in embedding/transformer groups)
            var excludedLayers = new HashSet<ILayer<T>>(_imageEmbeddingLayers);
            foreach (var l in _textEmbeddingLayers) excludedLayers.Add(l);
            foreach (var l in _transformerLayers) excludedLayers.Add(l);

            foreach (var layer in Layers.Where(l => !excludedLayers.Contains(l)))
                output = layer.Forward(output);

            return output;
        }

        // Fallback to sequential processing if groups not populated
        return base.Forward(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);

        if (_useNativeMode)
        {
            return Forward(preprocessed);
        }
        else
        {
            return RunOnnxInference(preprocessed);
        }
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Training is not supported in ONNX inference mode. Use native mode for training.");
        }

        SetTrainingMode(true);

        // Forward pass
        var output = Predict(input);

        // Compute loss
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        // Backward pass - compute gradients
        var lossGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        // Propagate gradients backward through layers
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        // Update embedding gradients
        UpdateEmbeddingGradients(gradient);

        // Apply optimizer update
        var paramGradients = CollectParameterGradients();
        UpdateParameters(paramGradients);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Parameter updates are not supported in ONNX inference mode.");
        }

        int expectedCount = ParameterCount;
        if (gradients.Length != expectedCount)
        {
            throw new ArgumentException(
                $"Expected {expectedCount} gradients, but got {gradients.Length}",
                nameof(gradients));
        }

        // Get current parameters and apply gradient descent update
        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.001);

        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(currentParams);
    }

    private void UpdateEmbeddingGradients(Tensor<T> gradient)
    {
        // Update position embedding gradients (simplified)
        if (_position1DEmbeddingsGradients is not null && gradient.Data.Length > 0)
        {
            int gradLen = Math.Min(gradient.Data.Length, _position1DEmbeddingsGradients.Data.Length);
            for (int i = 0; i < gradLen; i++)
            {
                _position1DEmbeddingsGradients.Data.Span[i] = NumOps.Add(
                    _position1DEmbeddingsGradients.Data.Span[i],
                    gradient.Data.Span[i % gradient.Data.Length]);
            }
        }
    }

    private Vector<T> CollectParameterGradients()
    {
        var gradients = new List<T>();

        // Collect gradients from all layers
        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            gradients.AddRange(layerGradients);
        }

        // Add embedding gradients
        if (_position1DEmbeddingsGradients is not null)
            gradients.AddRange(_position1DEmbeddingsGradients.Data.ToArray());
        if (_position2DXEmbeddingsGradients is not null)
            gradients.AddRange(_position2DXEmbeddingsGradients.Data.ToArray());
        if (_position2DYEmbeddingsGradients is not null)
            gradients.AddRange(_position2DYEmbeddingsGradients.Data.ToArray());

        return new Vector<T>([.. gradients]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
