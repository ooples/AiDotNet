using AiDotNet.Document.Interfaces;
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
/// LayoutLM (v1) neural network for document understanding with layout-aware pre-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutLM is the first generation of Microsoft's layout-aware document understanding models.
/// It combines text embeddings with 2D position embeddings to jointly model text and layout.
/// </para>
/// <para>
/// <b>For Beginners:</b> LayoutLM understands documents by learning from both:
/// 1. The text content (what the words say)
/// 2. The layout structure (where words are positioned on the page)
///
/// Unlike LayoutLMv2/v3, this version does NOT use visual features (images),
/// making it lighter but less powerful for visually-rich documents.
///
/// Example usage:
/// <code>
/// var model = new LayoutLM&lt;float&gt;(architecture, tokenizer);
/// var embeddings = model.EncodeDocument(documentText, boundingBoxes);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "LayoutLM: Pre-training of Text and Layout for Document Image Understanding" (KDD 2020)
/// https://arxiv.org/abs/1912.13318
/// </para>
/// </remarks>
public class LayoutLM<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>
{
    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private readonly int _maxPosition2D;
    private readonly int _numClasses;

    // Native mode layers
    private readonly List<ILayer<T>> _embeddingLayers = [];
    private readonly List<ILayer<T>> _transformerLayers = [];
    private readonly List<ILayer<T>> _classificationLayers = [];

    // Learnable embeddings
    private Tensor<T>? _wordEmbeddings;
    private Tensor<T>? _positionEmbeddings;
    private Tensor<T>? _position2DXEmbeddings;
    private Tensor<T>? _position2DYEmbeddings;
    private Tensor<T>? _position2DWEmbeddings;
    private Tensor<T>? _position2DHEmbeddings;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => 0; // LayoutLM v1 doesn't use images

    /// <inheritdoc/>
    public IReadOnlyList<LayoutElementType> SupportedElementTypes { get; } =
    [
        LayoutElementType.Text,
        LayoutElementType.Title,
        LayoutElementType.List,
        LayoutElementType.Table,
        LayoutElementType.Figure,
        LayoutElementType.FormField
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a LayoutLM model using a pre-trained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer for text processing.</param>
    /// <param name="numClasses">Number of output classes (default: 7 for FUNSD).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768 for base).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522 for BERT).</param>
    /// <param name="maxPosition2D">Max 2D position value (default: 1024).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    public LayoutLM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numClasses = 7,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int maxPosition2D = 1024,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
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
        _maxPosition2D = maxPosition2D;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a LayoutLM model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="tokenizer">Tokenizer for text processing (optional).</param>
    /// <param name="numClasses">Number of output classes (default: 7 for FUNSD).</param>
    /// <param name="maxSequenceLength">Maximum sequence length (default: 512).</param>
    /// <param name="hiddenDim">Hidden dimension (default: 768 for base).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="vocabSize">Vocabulary size (default: 30522 for BERT).</param>
    /// <param name="maxPosition2D">Max 2D position value (default: 1024).</param>
    /// <param name="optimizer">Optimizer for training (optional).</param>
    /// <param name="lossFunction">Loss function (optional).</param>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (LayoutLM-Base from KDD 2020):</b>
    /// - Architecture: BERT-base with 2D position embeddings
    /// - Hidden dimension: 768
    /// - Layers: 12, Heads: 12
    /// - 2D position range: 0-1024 (normalized coordinates)
    /// </para>
    /// </remarks>
    public LayoutLM(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numClasses = 7,
        int maxSequenceLength = 512,
        int hiddenDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int vocabSize = 30522,
        int maxPosition2D = 1024,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _numClasses = numClasses;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _maxPosition2D = maxPosition2D;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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

        Layers.AddRange(LayerHelper<T>.CreateDefaultLayoutLMLayers(
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            maxSequenceLength: MaxSequenceLength,
            numClasses: _numClasses));
    }

    private void InitializeEmbeddings()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        _wordEmbeddings = Tensor<T>.CreateDefault([_vocabSize, _hiddenDim], NumOps.Zero);
        _positionEmbeddings = Tensor<T>.CreateDefault([MaxSequenceLength, _hiddenDim], NumOps.Zero);
        _position2DXEmbeddings = Tensor<T>.CreateDefault([_maxPosition2D, _hiddenDim], NumOps.Zero);
        _position2DYEmbeddings = Tensor<T>.CreateDefault([_maxPosition2D, _hiddenDim], NumOps.Zero);
        _position2DWEmbeddings = Tensor<T>.CreateDefault([_maxPosition2D, _hiddenDim], NumOps.Zero);
        _position2DHEmbeddings = Tensor<T>.CreateDefault([_maxPosition2D, _hiddenDim], NumOps.Zero);

        InitializeWithSmallRandomValues(_wordEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_positionEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_position2DXEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_position2DYEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_position2DWEmbeddings, random, 0.02);
        InitializeWithSmallRandomValues(_position2DHEmbeddings, random, 0.02);
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

    #region ILayoutDetector Implementation

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage)
    {
        return DetectLayout(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold)
    {
        // LayoutLM v1 requires OCR input, not raw images
        // This is a simplified implementation that works with pre-extracted features
        var startTime = DateTime.UtcNow;

        var output = _useNativeMode
            ? Forward(documentImage)
            : RunOnnxInference(documentImage);

        var result = ParseLayoutOutput(output, confidenceThreshold);

        return new DocumentLayoutResult<T>
        {
            Regions = result.Regions,
            ReadingOrder = result.ReadingOrder,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private DocumentLayoutResult<T> ParseLayoutOutput(Tensor<T> output, double threshold)
    {
        var regions = new List<LayoutRegion<T>>();

        int seqLen = output.Shape[0];
        int numClasses = output.Shape.Length > 1 ? output.Shape[1] : _numClasses;

        for (int i = 0; i < seqLen; i++)
        {
            double maxConf = 0;
            int maxClass = 0;
            for (int c = 0; c < numClasses; c++)
            {
                double conf = NumOps.ToDouble(output[i, c]);
                if (conf > maxConf)
                {
                    maxConf = conf;
                    maxClass = c;
                }
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

        return new DocumentLayoutResult<T>
        {
            Regions = regions
        };
    }

    #endregion

    #region IDocumentModel Implementation

    /// <inheritdoc/>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        return _useNativeMode ? Forward(documentImage) : RunOnnxInference(documentImage);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        // LayoutLM works with text+bbox, not raw images
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("LayoutLM (v1) Model Summary");
        sb.AppendLine("===========================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: BERT-base with 2D position embeddings");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Number of Layers: {_numLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Vocabulary Size: {_vocabSize}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Max 2D Position: {_maxPosition2D}");
        sb.AppendLine($"Number of Classes: {_numClasses}");
        sb.AppendLine($"Uses Visual Features: No");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies LayoutLM's industry-standard preprocessing: pass-through (works with pre-tokenized input).
    /// </summary>
    /// <remarks>
    /// LayoutLM v1 (Microsoft paper) works primarily with pre-tokenized text and bounding boxes.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        return rawImage; // LayoutLM v1 works with pre-tokenized input
    }

    /// <summary>
    /// Applies LayoutLM's industry-standard postprocessing: softmax over class dimension.
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput)
    {
        return ApplySoftmax(modelOutput);
    }

    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        int seqLen = input.Shape[0];
        int numClasses = input.Shape.Length > 1 ? input.Shape[1] : _numClasses;

        for (int s = 0; s < seqLen; s++)
        {
            double maxVal = double.MinValue;
            for (int c = 0; c < numClasses; c++)
            {
                double val = NumOps.ToDouble(input[s, c]);
                if (val > maxVal) maxVal = val;
            }

            double sumExp = 0;
            for (int c = 0; c < numClasses; c++)
            {
                sumExp += Math.Exp(NumOps.ToDouble(input[s, c]) - maxVal);
            }

            for (int c = 0; c < numClasses; c++)
            {
                double val = NumOps.ToDouble(input[s, c]);
                output[s, c] = NumOps.FromDouble(Math.Exp(val - maxVal) / sumExp);
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
            Name = "LayoutLM",
            ModelType = ModelType.NeuralNetwork,
            Description = "LayoutLM v1 with text and 2D position embeddings (KDD 2020)",
            FeatureCount = _hiddenDim,
            Complexity = _numLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_layers", _numLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "max_sequence_length", MaxSequenceLength },
                { "max_position_2d", _maxPosition2D },
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
        writer.Write(_maxPosition2D);
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
        int maxSeqLen = reader.ReadInt32();
        int maxPos2D = reader.ReadInt32();
        int numClasses = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new LayoutLM<T>(
            Architecture,
            _tokenizer,
            _numClasses,
            MaxSequenceLength,
            _hiddenDim,
            _numLayers,
            _numHeads,
            _vocabSize,
            _maxPosition2D);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : RunOnnxInference(input);
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
