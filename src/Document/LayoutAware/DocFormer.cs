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
/// DocFormer neural network for end-to-end document understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DocFormer is a multi-modal transformer that jointly learns text, visual, and spatial features
/// for document understanding tasks. It uses shared spatial encodings across all modalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> DocFormer combines three types of information:
/// 1. Text content (what the words say)
/// 2. Visual features (what the document looks like)
/// 3. Spatial layout (where elements are positioned)
///
/// Unlike LayoutLM which adds position embeddings to text, DocFormer uses shared
/// spatial encodings that align all three modalities in the same coordinate space.
///
/// Example usage:
/// <code>
/// var model = new DocFormer&lt;float&gt;(architecture);
/// var result = model.DetectLayout(documentImage);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DocFormer: End-to-End Transformer for Document Understanding" (ICCV 2021)
/// https://arxiv.org/abs/2106.11539
/// </para>
/// </remarks>
public class DocFormer<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentClassifier<T>
{
    private readonly DocFormerOptions _options;

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
    private readonly int _spatialDim;

    // Native mode layers
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private readonly List<ILayer<T>> _visualEncoderLayers = [];
    private readonly List<ILayer<T>> _multiModalLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    // Learnable spatial embeddings
    private Tensor<T>? _spatialXEmbeddings;
    private Tensor<T>? _spatialYEmbeddings;

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
    /// Creates a DocFormer model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="numClasses">Number of output classes (default: 16 for RVL-CDIP).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="spatialDim">Spatial embedding dimension (default: 128).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    public DocFormer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numClasses = 16,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int spatialDim = 128,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DocFormerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new DocFormerOptions();
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
        _spatialDim = spatialDim;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a DocFormer model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text processing (optional).</param>
    /// <param name="numClasses">Number of output classes (default: 16 for RVL-CDIP).</param>
    /// <param name="imageSize">Input image size (default: 224).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522).</param>
    /// <param name="spatialDim">Spatial embedding dimension (default: 128).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (DocFormer-Base from ICCV 2021):</b>
    /// - Text encoder: BERT-base architecture
    /// - Visual encoder: ResNet-50 backbone
    /// - Shared spatial encodings for all modalities
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// - Image size: 224x224
    /// </para>
    /// </remarks>
    public DocFormer(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numClasses = 16,
        int imageSize = 224,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int spatialDim = 128,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DocFormerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new DocFormerOptions();
        Options = _options;

        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _spatialDim = spatialDim;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);

        InitializeLayers();
        InitializeSpatialEmbeddings();
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultDocFormerLayers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            imageSize: ImageSize,
            spatialDim: _spatialDim,
            numClasses: _numClasses));
    }

    private void InitializeSpatialEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        // Shared spatial embeddings for X and Y coordinates
        _spatialXEmbeddings = Tensor<T>.CreateDefault([1024, _spatialDim], NumOps.Zero);
        _spatialYEmbeddings = Tensor<T>.CreateDefault([1024, _spatialDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_spatialXEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_spatialYEmbeddings, random, 0.02);
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

        // Apply softmax for classification probabilities
        var probs = ApplySoftmax(output);

        // Get top-K predictions
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
        sb.AppendLine("DocFormer Model Summary");
        sb.AppendLine("=======================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: Multi-modal Transformer with shared spatial encodings");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Spatial Embedding Dim: {_spatialDim}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Uses Visual Features: Yes");
        sb.AppendLine($"Uses Shared Spatial Encodings: Yes");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies DocFormer's industry-standard preprocessing: ImageNet normalization.
    /// </summary>
    /// <remarks>
    /// DocFormer uses ImageNet normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
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
    /// Applies DocFormer's industry-standard postprocessing: pass-through (multimodal outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "DocFormer",
            ModelType = ModelType.NeuralNetwork,
            Description = "DocFormer with shared spatial encodings (ICCV 2021)",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "image_size", ImageSize },
                { "spatial_dim", _spatialDim },
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
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_spatialDim);
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
        int spatialDim = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DocFormer<T>(Architecture, _tokenizer, _numClasses, ImageSize, MaxSequenceLength,
            _hiddenDim, _numLayers, _numHeads, _vocabSize, _spatialDim);
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
