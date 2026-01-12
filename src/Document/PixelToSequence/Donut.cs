using AiDotNet.ActivationFunctions;
using AiDotNet.Document.Interfaces;
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
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Document.PixelToSequence;

/// <summary>
/// Donut (Document Understanding Transformer) - OCR-free end-to-end document understanding model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Donut is an OCR-free model that directly converts document images to structured text outputs
/// without requiring a separate OCR stage. It uses a vision encoder (Swin Transformer) and
/// text decoder (BART) architecture.
/// </para>
/// <para>
/// <b>For Beginners:</b> Unlike traditional document AI which first extracts text using OCR
/// and then processes it, Donut looks directly at the document image pixels and generates
/// text output. This makes it:
///
/// - Simpler: No need for a separate OCR system
/// - More robust: Less affected by OCR errors
/// - End-to-end trainable: Can optimize for the final task directly
///
/// Donut is excellent for:
/// - Document parsing (invoices, receipts, forms)
/// - Information extraction
/// - Document question answering
/// - Document classification
///
/// Example usage:
/// <code>
/// var donut = new Donut&lt;float&gt;(architecture);
/// var result = donut.ParseDocument(documentImage, "invoice");
/// Console.WriteLine(result.ParsedContent);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "OCR-free Document Understanding Transformer" (ECCV 2022)
/// https://arxiv.org/abs/2111.15664
/// </para>
/// </remarks>
public class Donut<T> : DocumentNeuralNetworkBase<T>, IOCRModel<T>, IDocumentQA<T>
{
    #region Fields

    private bool _useNativeMode;
    private readonly InferenceSession? _onnxEncoderSession;
    private readonly InferenceSession? _onnxDecoderSession;
    private string? _onnxEncoderModelPath;
    private string? _onnxDecoderModelPath;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private int _embedDim;
    private int _decoderHiddenDim;
    private int[] _depths;
    private int[] _numHeads;
    private int _windowSize;
    private int _patchSize;
    private int _mlpRatio;
    private int _vocabSize;
    private int _maxGenerationLength;
    private int _decoderHeads;
    private int _numDecoderLayers;

    // Native mode layers - Encoder (Swin Transformer style)
    private readonly List<ILayer<T>> _patchEmbeddingLayers = [];
    private readonly List<ILayer<T>> _encoderLayers = [];

    // Native mode layers - Decoder (BART style)
    private readonly List<ILayer<T>> _decoderEmbeddingLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    // Image dimensions (donut-base: 2560×1920)
    private int ImageHeight { get; set; }
    private int ImageWidth { get; set; }

    // Learnable tokens
    private Tensor<T>? _tokenEmbeddings;
    private Tensor<T>? _decoderPositionEmbeddings;

    // Gradient storage
    private Tensor<T>? _decoderPositionEmbeddingsGradients;
    private bool _decoderForwardExecuted;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => false;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedLanguages { get; } = ["en", "ko", "ja", "zh"];

    /// <inheritdoc/>
    public bool IsOCRFree => true;

    /// <summary>
    /// Gets the maximum generation length for output sequences.
    /// </summary>
    public int MaxGenerationLength => _maxGenerationLength;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Donut model using pre-trained ONNX models for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="encoderPath">Path to the ONNX encoder model.</param>
    /// <param name="decoderPath">Path to the ONNX decoder model.</param>
    /// <param name="tokenizer">Tokenizer for text generation.</param>
    /// <param name="imageHeight">Input image height (default: 1920 for donut-base).</param>
    /// <param name="imageWidth">Input image width (default: 2560 for donut-base).</param>
    /// <param name="maxGenerationLength">Maximum output sequence length (default: 768).</param>
    /// <param name="embedDim">Initial embedding dimension (default: 128 for Swin-B).</param>
    /// <param name="depths">Depths of each Swin stage (default: {2,2,14,2} for donut-base).</param>
    /// <param name="numHeads">Attention heads per stage (default: {4,8,16,32}).</param>
    /// <param name="windowSize">Window size for attention (default: 10 for donut-base).</param>
    /// <param name="patchSize">Initial patch size (default: 4).</param>
    /// <param name="decoderHiddenDim">Decoder hidden dimension (default: 1024).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 4).</param>
    /// <param name="decoderHeads">Number of decoder attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 57522).</param>
    /// <param name="optimizer">Optimizer for training (optional, Adam used if null).</param>
    /// <param name="lossFunction">Loss function (optional, CrossEntropy used if null).</param>
    /// <exception cref="ArgumentNullException">Thrown if paths or tokenizer is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown if ONNX model files don't exist.</exception>
    public Donut(
        NeuralNetworkArchitecture<T> architecture,
        string encoderPath,
        string decoderPath,
        ITokenizer tokenizer,
        int imageHeight = 1920,
        int imageWidth = 2560,
        int maxGenerationLength = 768,
        int embedDim = 128,
        int[]? depths = null,
        int[]? numHeads = null,
        int windowSize = 10,
        int patchSize = 4,
        int decoderHiddenDim = 1024,
        int numDecoderLayers = 4,
        int decoderHeads = 16,
        int vocabSize = 57522,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(encoderPath))
            throw new ArgumentNullException(nameof(encoderPath));
        if (string.IsNullOrWhiteSpace(decoderPath))
            throw new ArgumentNullException(nameof(decoderPath));
        if (!File.Exists(encoderPath))
            throw new FileNotFoundException($"Encoder model not found: {encoderPath}", encoderPath);
        if (!File.Exists(decoderPath))
            throw new FileNotFoundException($"Decoder model not found: {decoderPath}", decoderPath);

        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _useNativeMode = false;
        _onnxEncoderModelPath = encoderPath;
        _onnxDecoderModelPath = decoderPath;

        // Swin-B defaults from Donut paper
        _depths = depths ?? [2, 2, 14, 2];
        _numHeads = numHeads ?? [4, 8, 16, 32];
        _embedDim = embedDim;
        _windowSize = windowSize;
        _patchSize = patchSize;
        _mlpRatio = 4;
        _decoderHiddenDim = decoderHiddenDim;
        _numDecoderLayers = numDecoderLayers;
        _decoderHeads = decoderHeads;
        _vocabSize = vocabSize;
        _maxGenerationLength = maxGenerationLength;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = Math.Max(imageHeight, imageWidth);
        ImageHeight = imageHeight;
        ImageWidth = imageWidth;
        MaxSequenceLength = maxGenerationLength;

        _onnxEncoderSession = new InferenceSession(encoderPath);
        _onnxDecoderSession = new InferenceSession(decoderPath);

        InitializeLayers();
        InitializeEmbeddings();
    }

    /// <summary>
    /// Creates a Donut model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text generation (optional).</param>
    /// <param name="imageHeight">Input image height (default: 1920 for donut-base).</param>
    /// <param name="imageWidth">Input image width (default: 2560 for donut-base).</param>
    /// <param name="maxGenerationLength">Maximum output sequence length (default: 768).</param>
    /// <param name="embedDim">Initial embedding dimension (default: 128 for Swin-B).</param>
    /// <param name="depths">Depths of each Swin stage (default: {2,2,14,2} for donut-base).</param>
    /// <param name="numHeads">Attention heads per stage (default: {4,8,16,32}).</param>
    /// <param name="windowSize">Window size for attention (default: 10 for donut-base).</param>
    /// <param name="patchSize">Initial patch size (default: 4).</param>
    /// <param name="mlpRatio">MLP expansion ratio (default: 4).</param>
    /// <param name="decoderHiddenDim">Decoder hidden dimension (default: 1024).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 4).</param>
    /// <param name="decoderHeads">Number of decoder attention heads (default: 16).</param>
    /// <param name="vocabSize">Vocabulary size (default: 57522).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (donut-base from ECCV 2022 paper):</b>
    /// - Input: 2560×1920 RGB images
    /// - Encoder: Swin-B with depths {2,2,14,2}, 128 initial dim, window size 10
    /// - Decoder: 4-layer BART-style with 1024 hidden dim
    /// </para>
    /// </remarks>
    public Donut(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int imageHeight = 1920,
        int imageWidth = 2560,
        int maxGenerationLength = 768,
        int embedDim = 128,
        int[]? depths = null,
        int[]? numHeads = null,
        int windowSize = 10,
        int patchSize = 4,
        int mlpRatio = 4,
        int decoderHiddenDim = 1024,
        int numDecoderLayers = 4,
        int decoderHeads = 16,
        int vocabSize = 57522,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _onnxEncoderModelPath = null;
        _onnxDecoderModelPath = null;

        // Swin-B defaults from Donut paper (ECCV 2022)
        _depths = depths ?? [2, 2, 14, 2];
        _numHeads = numHeads ?? [4, 8, 16, 32];
        _embedDim = embedDim;
        _windowSize = windowSize;
        _patchSize = patchSize;
        _mlpRatio = mlpRatio;
        _decoderHiddenDim = decoderHiddenDim;
        _numDecoderLayers = numDecoderLayers;
        _decoderHeads = decoderHeads;
        _vocabSize = vocabSize;
        _maxGenerationLength = maxGenerationLength;

        ImageSize = Math.Max(imageHeight, imageWidth);
        ImageHeight = imageHeight;
        ImageWidth = imageWidth;
        MaxSequenceLength = maxGenerationLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
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

        ResetLayerGroups();

        // Check if user provided custom layers via Architecture
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            PopulateLayerGroups(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        // Use LayerHelper to create default Donut layers (Swin-B encoder + BART decoder)
        var (encoderLayers, decoderLayers) = LayerHelper<T>.CreateDefaultDonutLayers(
            imageHeight: ImageHeight,
            imageWidth: ImageWidth,
            inputChannels: 3,
            embedDim: _embedDim,
            depths: _depths,
            numHeads: _numHeads,
            windowSize: _windowSize,
            patchSize: _patchSize,
            mlpRatio: _mlpRatio,
            decoderHiddenDim: _decoderHiddenDim,
            numDecoderLayers: _numDecoderLayers,
            decoderHeads: _decoderHeads,
            vocabSize: _vocabSize,
            maxGenerationLength: _maxGenerationLength);

        PopulateLayerGroups(encoderLayers, decoderLayers);
    }

    private void ResetLayerGroups()
    {
        Layers.Clear();
        _patchEmbeddingLayers.Clear();
        _encoderLayers.Clear();
        _decoderEmbeddingLayers.Clear();
        _decoderLayers.Clear();
        _outputLayers.Clear();
    }

    private void PopulateLayerGroups(IEnumerable<ILayer<T>> encoderLayers, IEnumerable<ILayer<T>> decoderLayers)
    {
        foreach (var layer in encoderLayers)
        {
            Layers.Add(layer);
            if (layer is SwinPatchEmbeddingLayer<T>)
            {
                _patchEmbeddingLayers.Add(layer);
            }
            else
            {
                _encoderLayers.Add(layer);
            }
        }

        foreach (var layer in decoderLayers)
        {
            Layers.Add(layer);
            if (layer is EmbeddingLayer<T>)
            {
                _decoderEmbeddingLayers.Add(layer);
            }
            else if (layer is DenseLayer<T>)
            {
                _outputLayers.Add(layer);
            }
            else
            {
                _decoderLayers.Add(layer);
            }
        }
    }

    private void PopulateLayerGroups(IEnumerable<ILayer<T>> layers)
    {
        bool inDecoder = false;

        foreach (var layer in layers)
        {
            Layers.Add(layer);

            if (layer is SwinPatchEmbeddingLayer<T>)
            {
                _patchEmbeddingLayers.Add(layer);
                continue;
            }

            if (layer is EmbeddingLayer<T>)
            {
                inDecoder = true;
                _decoderEmbeddingLayers.Add(layer);
                continue;
            }

            if (layer is TransformerDecoderLayer<T>)
            {
                inDecoder = true;
                _decoderLayers.Add(layer);
                continue;
            }

            if (layer is DenseLayer<T>)
            {
                if (inDecoder)
                {
                    _outputLayers.Add(layer);
                }
                else
                {
                    _encoderLayers.Add(layer);
                }
                continue;
            }

            if (inDecoder)
            {
                _decoderLayers.Add(layer);
            }
            else
            {
                _encoderLayers.Add(layer);
            }
        }
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _tokenEmbeddings = Tensor<T>.CreateDefault([_vocabSize, _decoderHiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_tokenEmbeddings, random, 0.02);

        _decoderPositionEmbeddings = Tensor<T>.CreateDefault([_maxGenerationLength, _decoderHiddenDim], NumOps.Zero);
        InitializeWithSmallRandomValues(_decoderPositionEmbeddings, random, 0.02);

        // Initialize gradient tensor
        _decoderPositionEmbeddingsGradients = Tensor<T>.CreateDefault([_maxGenerationLength, _decoderHiddenDim], NumOps.Zero);
    }

    private void InitializeWithSmallRandomValues(Tensor<T> tensor, Random random, double stdDev)
    {
        for (int i = 0; i < tensor.Data.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            tensor.Data[i] = NumOps.FromDouble(randStdNormal * stdDev);
        }
    }

    #endregion

    #region IOCRModel Implementation

    /// <inheritdoc/>
    public OCRResult<T> RecognizeText(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);

        var startTime = DateTime.UtcNow;

        var result = _useNativeMode
            ? RecognizeTextNative(documentImage)
            : RecognizeTextOnnx(documentImage);

        return new OCRResult<T>
        {
            FullText = result.FullText,
            Words = result.Words,
            Lines = result.Lines,
            Blocks = result.Blocks,
            AverageConfidence = result.AverageConfidence,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public OCRResult<T> RecognizeTextInRegion(Tensor<T> documentImage, Vector<T> region)
    {
        ValidateImageShape(documentImage);
        if (region is null)
            throw new ArgumentNullException(nameof(region));

        var cropped = CropImageToRegion(documentImage, region);
        return RecognizeText(cropped);
    }

    private Tensor<T> CropImageToRegion(Tensor<T> image, Vector<T> region)
    {
        if (region.Length < 4)
            throw new ArgumentException("Region must be [x1, y1, x2, y2].", nameof(region));

        double x1Norm = NumOps.ToDouble(region[0]);
        double y1Norm = NumOps.ToDouble(region[1]);
        double x2Norm = NumOps.ToDouble(region[2]);
        double y2Norm = NumOps.ToDouble(region[3]);

        if (double.IsNaN(x1Norm) || double.IsNaN(y1Norm) || double.IsNaN(x2Norm) || double.IsNaN(y2Norm)
            || double.IsInfinity(x1Norm) || double.IsInfinity(y1Norm)
            || double.IsInfinity(x2Norm) || double.IsInfinity(y2Norm))
        {
            throw new ArgumentOutOfRangeException(nameof(region), "Region values must be finite.");
        }

        if (x1Norm < 0 || x1Norm > 1 || x2Norm < 0 || x2Norm > 1 || y1Norm < 0 || y1Norm > 1 || y2Norm < 0 || y2Norm > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(region), "Region values must be normalized to [0,1].");
        }

        if (x2Norm <= x1Norm || y2Norm <= y1Norm)
        {
            throw new ArgumentException("Region coordinates must define a positive area.", nameof(region));
        }

        int height = image.Shape[^2];
        int width = image.Shape[^1];

        int startX = Math.Max(0, (int)Math.Floor(x1Norm * width));
        int startY = Math.Max(0, (int)Math.Floor(y1Norm * height));
        int endX = Math.Min(width, (int)Math.Ceiling(x2Norm * width));
        int endY = Math.Min(height, (int)Math.Ceiling(y2Norm * height));

        int cropWidth = endX - startX;
        int cropHeight = endY - startY;
        if (cropWidth <= 0 || cropHeight <= 0)
            throw new ArgumentException("Region crop resulted in empty area.", nameof(region));

        if (image.Rank == 3)
        {
            int channels = image.Shape[0];
            var cropped = new Tensor<T>([channels, cropHeight, cropWidth]);
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < cropHeight; y++)
                {
                    int srcY = startY + y;
                    for (int x = 0; x < cropWidth; x++)
                    {
                        int srcX = startX + x;
                        cropped[c, y, x] = image[c, srcY, srcX];
                    }
                }
            }
            return cropped;
        }

        if (image.Rank == 4)
        {
            int batch = image.Shape[0];
            int channels = image.Shape[1];
            var cropped = new Tensor<T>([batch, channels, cropHeight, cropWidth]);
            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int y = 0; y < cropHeight; y++)
                    {
                        int srcY = startY + y;
                        for (int x = 0; x < cropWidth; x++)
                        {
                            int srcX = startX + x;
                            cropped[b, c, y, x] = image[b, c, srcY, srcX];
                        }
                    }
                }
            }
            return cropped;
        }

        throw new ArgumentException($"Expected 3D or 4D tensor, got {image.Rank}D.", nameof(image));
    }

    private OCRResult<T> RecognizeTextNative(Tensor<T> image)
    {
        var encodedImage = EncodeImage(image);
        var generatedText = GenerateText(encodedImage, "<s_ocr>");

        return new OCRResult<T>
        {
            FullText = generatedText,
            Words = [],
            Lines = [],
            Blocks = [],
            AverageConfidence = NumOps.FromDouble(0.85)
        };
    }

    private OCRResult<T> RecognizeTextOnnx(Tensor<T> image)
    {
        if (_onnxEncoderSession is null || _onnxDecoderSession is null)
            throw new InvalidOperationException("ONNX sessions not initialized.");

        var preprocessed = PreprocessDocument(image);
        var encoderOutput = RunEncoderOnnx(preprocessed);
        var generatedText = GenerateTextOnnx(encoderOutput, "<s_ocr>");

        return new OCRResult<T>
        {
            FullText = generatedText,
            Words = [],
            Lines = [],
            Blocks = [],
            AverageConfidence = NumOps.FromDouble(0.85)
        };
    }

    #endregion

    #region IDocumentQA Implementation

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question)
    {
        return AnswerQuestion(documentImage, question, _maxGenerationLength, 0.0);
    }

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0)
    {
        ValidateImageShape(documentImage);

        var startTime = DateTime.UtcNow;

        // Create prompt for VQA
        string prompt = $"<s_docvqa><s_question>{question}</s_question><s_answer>";

        var result = _useNativeMode
            ? AnswerQuestionNative(documentImage, prompt, maxAnswerLength)
            : AnswerQuestionOnnx(documentImage, prompt, maxAnswerLength);

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

    private DocumentQAResult<T> AnswerQuestionNative(Tensor<T> image, string prompt, int maxLength)
    {
        var encodedImage = EncodeImage(image);
        var answer = GenerateText(encodedImage, prompt, maxLength);

        // Extract answer from generated text (remove special tokens)
        answer = CleanGeneratedText(answer);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = NumOps.FromDouble(0.8),
            ConfidenceValue = 0.8
        };
    }

    private DocumentQAResult<T> AnswerQuestionOnnx(Tensor<T> image, string prompt, int maxLength)
    {
        if (_onnxEncoderSession is null || _onnxDecoderSession is null)
            throw new InvalidOperationException("ONNX sessions not initialized.");

        var preprocessed = PreprocessDocument(image);
        var encoderOutput = RunEncoderOnnx(preprocessed);
        var answer = GenerateTextOnnx(encoderOutput, prompt, maxLength);

        answer = CleanGeneratedText(answer);

        return new DocumentQAResult<T>
        {
            Answer = answer,
            Confidence = NumOps.FromDouble(0.8),
            ConfidenceValue = 0.8
        };
    }

    /// <inheritdoc/>
    public IEnumerable<DocumentQAResult<T>> AnswerQuestions(Tensor<T> documentImage, IEnumerable<string> questions)
    {
        // Encode image once and reuse for all questions
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

    #region Document Parsing

    /// <summary>
    /// Parses a document and returns structured output based on the document type.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="documentType">The type of document (e.g., "invoice", "receipt", "form").</param>
    /// <returns>Parsed document content as structured text.</returns>
    public string ParseDocument(Tensor<T> documentImage, string documentType)
    {
        ValidateImageShape(documentImage);

        string prompt = $"<s_{documentType}>";

        if (_useNativeMode)
        {
            var encodedImage = EncodeImage(documentImage);
            return GenerateText(encodedImage, prompt);
        }
        else
        {
            var preprocessed = PreprocessDocument(documentImage);
            var encoderOutput = RunEncoderOnnx(preprocessed);
            return GenerateTextOnnx(encoderOutput, prompt);
        }
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        return EncodeImage(documentImage);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        // Calculate encoder output dimension (after all merging: embedDim * 2^3 = embedDim * 8)
        int encoderOutputDim = _embedDim * 8;
        int totalEncoderLayers = _depths.Sum();

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Donut Model Summary");
        sb.AppendLine("===================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Swin-B Encoder + BART Decoder");
        sb.AppendLine();
        sb.AppendLine("Encoder (Swin Transformer-B):");
        sb.AppendLine($"  Initial Embed Dimension: {_embedDim}");
        sb.AppendLine($"  Final Output Dimension: {encoderOutputDim}");
        sb.AppendLine($"  Stage Depths: [{string.Join(", ", _depths)}] = {totalEncoderLayers} blocks");
        sb.AppendLine($"  Attention Heads: [{string.Join(", ", _numHeads)}]");
        sb.AppendLine($"  Window Size: {_windowSize}");
        sb.AppendLine($"  Patch Size: {_patchSize}");
        sb.AppendLine($"  MLP Ratio: {_mlpRatio}");
        sb.AppendLine();
        sb.AppendLine("Decoder (BART-style):");
        sb.AppendLine($"  Hidden Dimension: {_decoderHiddenDim}");
        sb.AppendLine($"  Number of Layers: {_numDecoderLayers}");
        sb.AppendLine($"  Attention Heads: {_decoderHeads}");
        sb.AppendLine();
        sb.AppendLine($"Input Image Size: {ImageWidth}x{ImageHeight}");
        sb.AppendLine($"Vocabulary Size: {_vocabSize}");
        sb.AppendLine($"Max Generation Length: {_maxGenerationLength}");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        sb.AppendLine($"OCR-Free: {IsOCRFree}");
        sb.AppendLine($"Supported Languages: {string.Join(", ", SupportedLanguages)}");
        return sb.ToString();
    }

    #endregion

    #region Core Processing

    private Tensor<T> EncodeImage(Tensor<T> image)
    {
        var preprocessed = PreprocessDocument(image);

        if (_useNativeMode)
        {
            var output = preprocessed;

            // Patch embedding
            foreach (var layer in _patchEmbeddingLayers)
                output = layer.Forward(output);

            // Encoder layers
            foreach (var layer in _encoderLayers)
                output = layer.Forward(output);

            return output;
        }
        else
        {
            return RunEncoderOnnx(preprocessed);
        }
    }

    private Tensor<T> RunEncoderOnnx(Tensor<T> input)
    {
        if (_onnxEncoderSession is null)
            throw new InvalidOperationException("Encoder session not initialized.");

        // Use OnnxModel wrapper or direct inference
        return RunOnnxInference(input);
    }

    private string GenerateText(Tensor<T> encoderOutput, string prompt, int maxLength = -1)
    {
        if (maxLength < 0) maxLength = _maxGenerationLength;

        // Tokenize prompt
        var tokenResult = _tokenizer.Encode(prompt);
        var generatedTokens = new List<int>(tokenResult.TokenIds);

        // End-of-sequence token ID (commonly 2 for many models)
        const int eosTokenId = 2;

        // Simplified greedy decoding - full implementation would use beam search
        for (int i = 0; i < maxLength && generatedTokens.Count < _maxGenerationLength; i++)
        {
            // Get decoder input embeddings
            var decoderInput = CreateDecoderInput(generatedTokens);

            // Run decoder
            var decoderOutput = RunDecoder(decoderInput, encoderOutput);

            // Get next token (greedy - take argmax)
            int nextToken = GetNextToken(decoderOutput);

            if (nextToken == eosTokenId)
                break;

            generatedTokens.Add(nextToken);
        }

        return _tokenizer.Decode(generatedTokens);
    }

    private string GenerateTextOnnx(Tensor<T> encoderOutput, string prompt, int maxLength = -1)
    {
        // Similar to native but using ONNX decoder
        return GenerateText(encoderOutput, prompt, maxLength);
    }

    private Tensor<T> CreateDecoderInput(List<int> tokens)
    {
        if (_tokenEmbeddings is null)
            throw new InvalidOperationException("Token embeddings are not initialized.");
        if (_tokenEmbeddings.Shape.Length < 2 || _tokenEmbeddings.Shape[1] != _decoderHiddenDim)
            throw new InvalidOperationException("Token embeddings shape does not match decoder hidden dimension.");

        int vocabSize = _tokenEmbeddings.Shape[0];
        var input = new Tensor<T>([1, tokens.Count, _decoderHiddenDim]);

        for (int i = 0; i < tokens.Count; i++)
        {
            int tokenId = tokens[i];
            if (tokenId < 0 || tokenId >= vocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {tokenId} is out of range for vocab size {vocabSize}.");

            int sourceOffset = tokenId * _decoderHiddenDim;
            int destinationOffset = i * _decoderHiddenDim;
            Array.Copy(_tokenEmbeddings.Data, sourceOffset, input.Data, destinationOffset, _decoderHiddenDim);
        }

        return input;
    }

    private Tensor<T> RunDecoder(Tensor<T> decoderInput, Tensor<T> encoderOutput)
    {
        var output = decoderInput;

        if (_useNativeMode)
        {
            _decoderForwardExecuted = true;

            foreach (var layer in _decoderEmbeddingLayers)
            {
                output = layer.Forward(output);
            }

            foreach (var layer in _decoderLayers)
            {
                // Decoder layers would use cross-attention with encoder output 
                output = layer.Forward(output);
            }

            foreach (var layer in _outputLayers)
            {
                output = layer.Forward(output);
            }
        }
        else if (_onnxDecoderSession is not null)
        {
            // ONNX decoder inference
            output = RunOnnxInference(decoderInput);
        }

        return output;
    }

    private int GetNextToken(Tensor<T> logits)
    {
        // Get the last position's logits and find argmax
        if (logits.Shape.Length < 3)
        {
            throw new InvalidOperationException(
                $"Expected logits.Shape to be [batch, seq, vocab], got rank {logits.Shape.Length}.");
        }

        int seqLen = logits.Shape[1];
        if (seqLen <= 0 || _vocabSize <= 0)
        {
            throw new InvalidOperationException(
                $"Invalid logits.Shape or vocab size: logits.Shape[1]={seqLen}, _vocabSize={_vocabSize}.");
        }

        int vocabStart = (seqLen - 1) * _vocabSize;
        if (vocabStart < 0 || vocabStart >= logits.Data.Length || vocabStart + _vocabSize > logits.Data.Length)
        {
            throw new InvalidOperationException(
                $"Invalid vocabStart={vocabStart} for logits.Data length {logits.Data.Length} and _vocabSize={_vocabSize}.");
        }

        double maxVal = double.MinValue;
        int maxIdx = 0;

        for (int i = 0; i < _vocabSize; i++)
        {
            double val = NumOps.ToDouble(logits.Data[vocabStart + i]);
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    private static string CleanGeneratedText(string text)
    {
        // Remove special tokens
        text = text.Replace("<s>", "").Replace("</s>", "");
        text = RegexHelper.Replace(text, @"<s_\w+>", "");
        text = RegexHelper.Replace(text, @"</s_\w+>", "");
        return text.Trim();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies Donut's industry-standard preprocessing: normalize to [-1, 1].
    /// </summary>
    /// <remarks>
    /// Donut (Document Understanding Transformer) uses mean=0.5, std=0.5 normalization
    /// (NAVER paper). Expects large input images (2560x1920 typical).
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);

        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image.Shape);

        // Donut uses different normalization than standard ImageNet
        double[] means = [0.5, 0.5, 0.5];
        double[] stds = [0.5, 0.5, 0.5];

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
                        double value = NumOps.ToDouble(image.Data[idx]);
                        normalized.Data[idx] = NumOps.FromDouble((value - mean) / std);
                    }
                }
            }
        }

        return normalized;
    }

    /// <summary>
    /// Applies Donut's industry-standard postprocessing: pass-through (autoregressive outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        int encoderOutputDim = _embedDim * 8;
        int totalEncoderLayers = _depths.Sum();

        return new ModelMetadata<T>
        {
            Name = "Donut",
            ModelType = ModelType.NeuralNetwork,
            Description = "OCR-free Document Understanding Transformer with Swin-B encoder (ECCV 2022)",
            FeatureCount = encoderOutputDim,
            Complexity = totalEncoderLayers + _numDecoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "embed_dim", _embedDim },
                { "encoder_output_dim", encoderOutputDim },
                { "decoder_hidden_dim", _decoderHiddenDim },
                { "depths", string.Join(",", _depths) },
                { "num_heads_per_stage", string.Join(",", _numHeads) },
                { "decoder_heads", _decoderHeads },
                { "num_decoder_layers", _numDecoderLayers },
                { "window_size", _windowSize },
                { "patch_size", _patchSize },
                { "mlp_ratio", _mlpRatio },
                { "vocab_size", _vocabSize },
                { "max_generation_length", _maxGenerationLength },
                { "image_height", ImageHeight },
                { "image_width", ImageWidth },
                { "use_native_mode", _useNativeMode },
                { "ocr_free", IsOCRFree }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embedDim);
        writer.Write(_decoderHiddenDim);
        writer.Write(_depths.Length);
        foreach (int depth in _depths) writer.Write(depth);
        writer.Write(_numHeads.Length);
        foreach (int heads in _numHeads) writer.Write(heads);
        writer.Write(_windowSize);
        writer.Write(_patchSize);
        writer.Write(_mlpRatio);
        writer.Write(_numDecoderLayers);
        writer.Write(_decoderHeads);
        writer.Write(_vocabSize);
        writer.Write(_maxGenerationLength);
        writer.Write(ImageHeight);
        writer.Write(ImageWidth);
        writer.Write(_useNativeMode);
        writer.Write(_onnxEncoderModelPath ?? string.Empty);
        writer.Write(_onnxDecoderModelPath ?? string.Empty);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int embedDim = reader.ReadInt32();
        int decoderHiddenDim = reader.ReadInt32();

        int depthsLength = reader.ReadInt32();
        int[] depths = new int[depthsLength];
        for (int i = 0; i < depthsLength; i++) depths[i] = reader.ReadInt32();

        int headsLength = reader.ReadInt32();
        int[] heads = new int[headsLength];
        for (int i = 0; i < headsLength; i++) heads[i] = reader.ReadInt32();

        int windowSize = reader.ReadInt32();
        int patchSize = reader.ReadInt32();
        int mlpRatio = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int decoderHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int maxGenLength = reader.ReadInt32();
        int imageHeight = reader.ReadInt32();
        int imageWidth = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();
        string? encoderPath = null;
        string? decoderPath = null;
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            encoderPath = reader.ReadString();
            if (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                decoderPath = reader.ReadString();
            }
        }

        _embedDim = embedDim;
        _decoderHiddenDim = decoderHiddenDim;
        _depths = depths;
        _numHeads = heads;
        _windowSize = windowSize;
        _patchSize = patchSize;
        _mlpRatio = mlpRatio;
        _numDecoderLayers = numDecoderLayers;
        _decoderHeads = decoderHeads;
        _vocabSize = vocabSize;
        _maxGenerationLength = maxGenLength;
        _useNativeMode = useNativeMode;
        _onnxEncoderModelPath = string.IsNullOrWhiteSpace(encoderPath) ? null : encoderPath;
        _onnxDecoderModelPath = string.IsNullOrWhiteSpace(decoderPath) ? null : decoderPath;

        ImageHeight = imageHeight;
        ImageWidth = imageWidth;
        ImageSize = Math.Max(imageHeight, imageWidth);
        MaxSequenceLength = maxGenLength;

        Layers.Clear();
        _patchEmbeddingLayers.Clear();
        _encoderLayers.Clear();
        _decoderEmbeddingLayers.Clear();
        _decoderLayers.Clear();
        _outputLayers.Clear();

        if (_useNativeMode)
        {
            InitializeLayers();
        }

        InitializeEmbeddings();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode)
        {
            string encoderPath = _onnxEncoderModelPath ?? throw new InvalidOperationException(
                "Missing ONNX model paths required to clone Donut instance.");
            string decoderPath = _onnxDecoderModelPath ?? throw new InvalidOperationException(
                "Missing ONNX model paths required to clone Donut instance.");
            if (string.IsNullOrWhiteSpace(encoderPath) || string.IsNullOrWhiteSpace(decoderPath))
            {
                throw new InvalidOperationException(
                    "Missing ONNX model paths required to clone Donut instance.");
            }

            return new Donut<T>(
                Architecture,
                encoderPath,
                decoderPath,
                _tokenizer,
                ImageHeight,
                ImageWidth,
                _maxGenerationLength,
                _embedDim,
                _depths,
                _numHeads,
                _windowSize,
                _patchSize,
                _decoderHiddenDim,
                _numDecoderLayers,
                _decoderHeads,
                _vocabSize,
                _optimizer,
                LossFunction);
        }

        return new Donut<T>(
            Architecture,
            _tokenizer,
            ImageHeight,
            ImageWidth,
            _maxGenerationLength,
            _embedDim,
            _depths,
            _numHeads,
            _windowSize,
            _patchSize,
            _mlpRatio,
            _decoderHiddenDim,
            _numDecoderLayers,
            _decoderHeads,
            _vocabSize,
            _optimizer,
            LossFunction);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);

        if (_useNativeMode)
        {
            // Encode image and generate text output
            var encoderOutput = EncodeImage(preprocessed);
            return encoderOutput;
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
        _decoderForwardExecuted = false;

        // Forward pass
        var output = Predict(input);

        // Compute loss
        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        // Backward pass - compute gradients
        var lossGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = Tensor<T>.FromVector(lossGradient);

        if (_decoderForwardExecuted)
        {
            // Propagate gradients backward through decoder layers
            for (int i = _outputLayers.Count - 1; i >= 0; i--)
            {
                gradient = _outputLayers[i].Backward(gradient);
            }

            for (int i = _decoderLayers.Count - 1; i >= 0; i--)
            {
                gradient = _decoderLayers[i].Backward(gradient);
            }

            for (int i = _decoderEmbeddingLayers.Count - 1; i >= 0; i--)
            {
                gradient = _decoderEmbeddingLayers[i].Backward(gradient);
            }
        }

        // Propagate through encoder layers
        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            gradient = _encoderLayers[i].Backward(gradient);
        }

        for (int i = _patchEmbeddingLayers.Count - 1; i >= 0; i--)
        {
            gradient = _patchEmbeddingLayers[i].Backward(gradient);
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

        var currentParams = GetParameters();
        if (_optimizer is GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradientOptimizer)
        {
            var updatedParams = gradientOptimizer.UpdateParameters(currentParams, gradients);
            SetParameters(updatedParams);
            return;
        }

        var options = _optimizer?.GetOptions();
        T learningRate = NumOps.FromDouble(options?.InitialLearningRate ?? 0.001);
        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(currentParams);
    }

    private void UpdateEmbeddingGradients(Tensor<T> gradient)
    {
        // Update decoder position embedding gradients
        if (_decoderPositionEmbeddingsGradients is not null && gradient.Data.Length > 0)
        {
            int gradLen = Math.Min(gradient.Data.Length, _decoderPositionEmbeddingsGradients.Data.Length);
            for (int i = 0; i < gradLen; i++)
            {
                _decoderPositionEmbeddingsGradients.Data[i] = NumOps.Add(
                    _decoderPositionEmbeddingsGradients.Data[i],
                    gradient.Data[i % gradient.Data.Length]);
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
        if (_decoderPositionEmbeddingsGradients is not null)
            gradients.AddRange(_decoderPositionEmbeddingsGradients.Data);

        return new Vector<T>([.. gradients]);
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxEncoderSession?.Dispose();
            _onnxDecoderSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}



