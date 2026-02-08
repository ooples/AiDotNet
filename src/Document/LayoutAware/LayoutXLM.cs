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

namespace AiDotNet.Document.LayoutAware;

/// <summary>
/// LayoutXLM neural network for multilingual document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutXLM extends LayoutLMv2 to support multilingual documents by using XLM-RoBERTa
/// as the text backbone and training on documents from multiple languages.
/// </para>
/// <para>
/// <b>For Beginners:</b> LayoutXLM understands documents in many languages:
/// 1. Supports 53 languages out-of-the-box
/// 2. Can handle mixed-language documents
/// 3. Zero-shot cross-lingual transfer (train on one language, test on another)
///
/// Key features:
/// - XLM-RoBERTa multilingual text encoder
/// - Visual backbone (ResNeXt-FPN) for image features
/// - Language-agnostic layout understanding
/// - Pre-trained on XFUND dataset (7 languages)
///
/// Example usage:
/// <code>
/// var model = new LayoutXLM&lt;float&gt;(architecture);
/// var result = model.DetectLayout(multilingualDocumentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding" (ACL 2022)
/// https://arxiv.org/abs/2104.08836
/// </para>
/// </remarks>
public class LayoutXLM<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>
{
    private readonly LayoutXLMOptions _options;

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
    private readonly int _numLanguages;

    // Native mode layers
    private readonly List<ILayer<T>> _visualBackboneLayers = [];
    private readonly List<ILayer<T>> _textEmbeddingLayers = [];
    private readonly List<ILayer<T>> _transformerLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    // Learnable embeddings
    private Tensor<T>? _positionEmbeddings;
    private Tensor<T>? _spatialPositionEmbeddings;
    private Tensor<T>? _visualPositionEmbeddings;
    private Tensor<T>? _languageEmbeddings;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the number of languages supported.
    /// </summary>
    public int NumLanguages => _numLanguages;

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
    /// Creates a LayoutXLM model using a pre-trained ONNX model for inference.
    /// </summary>
    public LayoutXLM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numClasses = 7,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 250002,
        int visualBackboneChannels = 256,
        int numLanguages = 53,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LayoutXLMOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutXLMOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _useNativeMode = false;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _visualBackboneChannels = visualBackboneChannels;
        _numLanguages = numLanguages;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a LayoutXLM model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (LayoutXLM-Base from ACL 2022):</b>
    /// - Text encoder: XLM-RoBERTa-base architecture
    /// - Visual backbone: ResNeXt-101 FPN
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// - Vocabulary: 250,002 tokens (multilingual)
    /// - Supports: 53 languages
    /// </para>
    /// </remarks>
    public LayoutXLM(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numClasses = 7,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 250002,
        int visualBackboneChannels = 256,
        int numLanguages = 53,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        LayoutXLMOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new LayoutXLMOptions();
        Options = _options;

        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _visualBackboneChannels = visualBackboneChannels;
        _numLanguages = numLanguages;
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultLayoutXLMLayers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            visualBackboneChannels: _visualBackboneChannels,
            numClasses: _numClasses));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _positionEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        _spatialPositionEmbeddings = Tensor<T>.CreateDefault([1024, _hiddenDim], NumOps.Zero);
        _visualPositionEmbeddings = Tensor<T>.CreateDefault([(ImageSize / 16) * (ImageSize / 16), _hiddenDim], NumOps.Zero);
        _languageEmbeddings = Tensor<T>.CreateDefault([_numLanguages, _hiddenDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_positionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_spatialPositionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_visualPositionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_languageEmbeddings, random, 0.02);
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

        // Extract answer using start/end logits from model output
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
    /// <remarks>
    /// LayoutXLM outputs token-level predictions. For QA, we find the span
    /// with highest start and end logits within the max answer length.
    /// </remarks>
    private (string answer, double confidence) ExtractAnswer(Tensor<T> output, int maxAnswerLength)
    {
        int seqLen = output.Shape[0];
        int hiddenDim = output.Shape.Length > 1 ? output.Shape[1] : _hiddenDim;

        // Find best start and end positions
        double bestStartScore = double.MinValue;
        double bestEndScore = double.MinValue;
        int bestStart = 0;
        int bestEnd = 0;

        // Interpret first and last values in hidden dimension as start/end logits
        for (int i = 0; i < seqLen; i++)
        {
            double startScore = NumOps.ToDouble(output[i, 0]);
            if (startScore > bestStartScore)
            {
                bestStartScore = startScore;
                bestStart = i;
            }
        }

        // Find best end position after start within max answer length
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

        // Extract token sequence and convert to text
        var tokens = new List<int>();
        for (int i = bestStart; i <= bestEnd && i < seqLen; i++)
        {
            // Extract argmax token at this position
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
            // BERT vocabulary mapping (simplified)
            char c = token switch
            {
                >= 1000 and <= 1031 => (char)(token - 1000 + 32),  // Space, punctuation
                >= 1032 and <= 1057 => (char)(token - 1032 + 65),  // A-Z
                >= 1058 and <= 1083 => (char)(token - 1058 + 97),  // a-z
                >= 103 and <= 125 => (char)(token - 103 + 48),     // Digits
                >= 126 and <= 151 => (char)(token - 126 + 65),     // A-Z
                >= 152 and <= 177 => (char)(token - 152 + 97),     // a-z
                _ => (char)((token % 95) + 32) // Fallback to printable ASCII
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
        sb.AppendLine("LayoutXLM Model Summary");
        sb.AppendLine("=======================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: XLM-RoBERTa + ResNeXt-FPN visual backbone");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Vocabulary Size: {_vocabSize}");
        sb.AppendLine($"Visual Backbone Channels: {_visualBackboneChannels}");
        sb.AppendLine($"Languages Supported: {_numLanguages}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Multilingual: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies LayoutXLM's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// LayoutXLM (Microsoft paper) is the multilingual version of LayoutLMv2, using same ImageNet normalization.
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
    /// Applies LayoutXLM's industry-standard postprocessing: pass-through (multilingual outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "LayoutXLM",
            ModelType = ModelType.NeuralNetwork,
            Description = "LayoutXLM for multilingual document understanding (ACL 2022)",
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
                { "num_languages", _numLanguages },
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
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_visualBackboneChannels);
        writer.Write(_numClasses);
        writer.Write(_numLanguages);
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
        int numLanguages = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LayoutXLM<T>(Architecture, _tokenizer, _numClasses, ImageSize, MaxSequenceLength,
            _hiddenDim, _numLayers, _numHeads, _vocabSize, _visualBackboneChannels, _numLanguages);
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
