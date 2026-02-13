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

namespace AiDotNet.Document.VisionLanguage;

/// <summary>
/// UDOP (Unifying Vision, Text, and Layout for Universal Document Processing) neural network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// UDOP is a foundation model for document AI that unifies text, image, and layout modalities
/// within a single encoder-decoder framework. It can perform multiple document tasks through
/// task-specific prompting.
/// </para>
/// <para>
/// <b>For Beginners:</b> UDOP can handle many document tasks with one model:
/// 1. Document classification
/// 2. Information extraction (NER, key-value pairs)
/// 3. Document question answering
/// 4. Document layout analysis
/// 5. Document generation
///
/// Example usage:
/// <code>
/// var model = new UDOP&lt;float&gt;(architecture);
/// var result = model.AnswerQuestion(documentImage, "What is the invoice total?");
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Unifying Vision, Text, and Layout for Universal Document Processing" (CVPR 2023)
/// https://arxiv.org/abs/2212.02623
/// </para>
/// </remarks>
public class UDOP<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>, IDocumentClassifier<T>
{
    private readonly UDOPOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numEncoderLayers;
    private readonly int _numDecoderLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private readonly int _numClasses;

    // Native mode layers
    private readonly List<ILayer<T>> _visualEncoderLayers = [];
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private readonly List<ILayer<T>> _unifiedEncoderLayers = [];
    private readonly List<ILayer<T>> _decoderLayers = [];

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
        LayoutElementType.FormField,
        LayoutElementType.Equation
    ];

    /// <summary>
    /// Gets the available document classification categories.
    /// </summary>
    public IReadOnlyList<string> AvailableCategories { get; } =
    [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific", "specification", "file_folder", "news_article",
        "budget", "invoice", "presentation", "questionnaire", "resume", "memo"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a UDOP model using a pre-trained ONNX model for inference.
    /// </summary>
    public UDOP(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numClasses = 16,
        int imageSize = 224,
        int maxSequenceLength = 2048,
        int hiddenDim = 1024,
        int numEncoderLayers = 12,
        int numDecoderLayers = 12,
        int numHeads = 16,
        int vocabSize = 50000,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        UDOPOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new UDOPOptions();
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
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a UDOP model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (UDOP-Large from CVPR 2023):</b>
    /// - Vision Transformer for image encoding
    /// - T5-style text encoder
    /// - Unified cross-modal encoder
    /// - T5-style decoder for generation
    /// - Hidden dimension: 1024
    /// - Encoder/Decoder layers: 12 each
    /// </para>
    /// </remarks>
    public UDOP(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numClasses = 16,
        int imageSize = 224,
        int maxSequenceLength = 2048,
        int hiddenDim = 1024,
        int numEncoderLayers = 12,
        int numDecoderLayers = 12,
        int numHeads = 16,
        int vocabSize = 50000,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        UDOPOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new UDOPOptions();
        Options = _options;

        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

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

        var (encoderLayers, decoderLayers) = LayerHelper<T>.CreateDefaultUDOPLayers(
            hiddenDim: _hiddenDim,
            numEncoderLayers: _numEncoderLayers,
            numDecoderLayers: _numDecoderLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            maxSequenceLength: MaxSequenceLength);

        Layers.AddRange(encoderLayers);
        Layers.AddRange(decoderLayers);
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
        return AnswerQuestion(documentImage, question, 256, 0.0);
    }

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        // UDOP uses generative output - decode the sequence
        var (answer, confidence) = DecodeGenerativeOutput(output, maxAnswerLength);

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
    /// Decodes generative output from UDOP model.
    /// </summary>
    private (string answer, double confidence) DecodeGenerativeOutput(Tensor<T> output, int maxLength)
    {
        var tokens = new List<int>();
        double totalConfidence = 0;
        int seqLen = Math.Min(output.Shape[0], maxLength);

        for (int t = 0; t < seqLen; t++)
        {
            int vocabSize = output.Shape.Length > 1 ? output.Shape[1] : _vocabSize;
            double maxVal = double.MinValue;
            int maxIdx = 0;

            for (int v = 0; v < vocabSize; v++)
            {
                double val = NumOps.ToDouble(output[t, v]);
                if (val > maxVal) { maxVal = val; maxIdx = v; }
            }

            // T5-style tokens: 0=PAD, 1=EOS
            if (maxIdx == 1) break; // EOS
            if (maxIdx == 0) continue; // Skip PAD
            tokens.Add(maxIdx);
            totalConfidence += maxVal;
        }

        string answer = DecodeTokensToText(tokens);
        double confidence = tokens.Count > 0 ? Math.Max(0, Math.Min(1, totalConfidence / tokens.Count)) : 0;

        return (string.IsNullOrEmpty(answer) ? "[No answer found]" : answer, confidence);
    }

    /// <summary>
    /// Decodes token IDs to text using T5-style vocabulary.
    /// </summary>
    private static string DecodeTokensToText(List<int> tokens)
    {
        if (tokens.Count == 0) return string.Empty;

        var sb = new System.Text.StringBuilder();
        foreach (int token in tokens)
        {
            char c = token switch
            {
                >= 2 and <= 33 => (char)(token - 2 + 32),    // Space, punctuation, digits
                >= 34 and <= 59 => (char)(token - 34 + 65),  // A-Z
                >= 60 and <= 85 => (char)(token - 60 + 97),  // a-z
                >= 86 and <= 213 => (char)(token - 86 + 128), // Extended ASCII
                _ => (char)((token % 95) + 32) // Fallback
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

    #region IDocumentClassifier Implementation

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage)
    {
        return ClassifyDocument(documentImage, 5);
    }

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage, int topK)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var probs = ApplySoftmax(output);
        var topPredictions = GetTopKPredictions(probs, topK);

        return new DocumentClassificationResult<T>
        {
            PredictedCategory = topPredictions[0].Category,
            Confidence = NumOps.FromDouble(topPredictions[0].Score),
            ConfidenceValue = topPredictions[0].Score,
            TopPredictions = topPredictions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private List<(string Category, double Score)> GetTopKPredictions(Tensor<T> probs, int k)
    {
        var predictions = new List<(string Category, double Score)>();
        int numClasses = Math.Min(probs.Data.Length, AvailableCategories.Count);

        for (int i = 0; i < numClasses; i++)
        {
            predictions.Add((AvailableCategories[i], NumOps.ToDouble(probs.Data.Span[i])));
        }

        return predictions.OrderByDescending(p => p.Score).Take(k).ToList();
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        int length = input.Data.Length;

        double maxVal = double.MinValue;
        for (int i = 0; i < length; i++)
        {
            double val = NumOps.ToDouble(input.Data.Span[i]);
            if (val > maxVal) maxVal = val;
        }

        double sumExp = 0;
        for (int i = 0; i < length; i++)
        {
            sumExp += Math.Exp(NumOps.ToDouble(input.Data.Span[i]) - maxVal);
        }

        for (int i = 0; i < length; i++)
        {
            double val = NumOps.ToDouble(input.Data.Span[i]);
            output.Data.Span[i] = NumOps.FromDouble(Math.Exp(val - maxVal) / sumExp);
        }

        return output;
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
        sb.AppendLine("UDOP Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Unified Vision-Text-Layout Encoder-Decoder");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Encoder Layers: {_numEncoderLayers}");
        sb.AppendLine($"Decoder Layers: {_numDecoderLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Capabilities: Layout, QA, Classification, Generation");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies UDOP's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// UDOP (Unified Document Processing) uses ImageNet normalization with
    /// mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] (Microsoft paper).
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
    /// Applies UDOP's industry-standard postprocessing: pass-through (unified outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "UDOP",
            ModelType = ModelType.NeuralNetwork,
            Description = "UDOP for unified document processing (CVPR 2023)",
            FeatureCount = _hiddenDim,
            Complexity = _numEncoderLayers + _numDecoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_encoder_layers", _numEncoderLayers },
                { "num_decoder_layers", _numDecoderLayers },
                { "num_heads", _numHeads },
                { "image_size", ImageSize },
                { "vocab_size", _vocabSize },
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
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_vocabSize);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_numClasses);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numEncoderLayers = reader.ReadInt32();
        int numDecoderLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new UDOP<T>(Architecture, _tokenizer, _numClasses, ImageSize, MaxSequenceLength,
            _hiddenDim, _numEncoderLayers, _numDecoderLayers, _numHeads, _vocabSize);
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
        T lr = NumOps.FromDouble(0.0001);
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
