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

namespace AiDotNet.Document.GraphBased;

/// <summary>
/// PICK (Processing Key Information Extraction) neural network for document key information extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PICK uses a graph neural network approach to extract key information from documents.
/// It models text segments as nodes and their relationships as edges, enabling
/// better understanding of document structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> PICK is especially good at:
/// 1. Extracting key-value pairs from invoices and receipts
/// 2. Understanding relationships between text segments
/// 3. Handling complex document layouts
/// 4. Named Entity Recognition in documents
///
/// Example usage:
/// <code>
/// var model = new PICK&lt;float&gt;(architecture);
/// var result = model.ExtractKeyInfo(documentImage);
/// foreach (var entity in result.Entities)
///     Console.WriteLine($"{entity.Label}: {entity.Text}");
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks" (ICPR 2020)
/// https://arxiv.org/abs/2004.07464
/// </para>
/// </remarks>
public class PICK<T> : DocumentNeuralNetworkBase<T>, IFormUnderstanding<T>
{
    private readonly PICKOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numGcnLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;
    private readonly int _numEntityTypes;

    // Native mode layers
    private readonly List<ILayer<T>> _textEncoderLayers = [];
    private readonly List<ILayer<T>> _gcnLayers = [];
    private readonly List<ILayer<T>> _outputLayers = [];

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.Form;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <inheritdoc/>
    public int ExpectedImageSize => ImageSize;

    /// <summary>
    /// Gets the supported entity types for extraction.
    /// </summary>
    public IReadOnlyList<string> SupportedEntityTypes { get; } =
    [
        "SELLER", "ADDRESS", "DATE", "TOTAL", "TAX", "ITEM", "QUANTITY", "PRICE",
        "INVOICE_NUMBER", "BUYER", "PAYMENT_METHOD", "DUE_DATE", "CURRENCY", "OTHER"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PICK model using a pre-trained ONNX model for inference.
    /// </summary>
    public PICK(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        int numEntityTypes = 14,
        int imageSize = 512,
        int maxSequenceLength = 512,
        int hiddenDim = 256,
        int numGcnLayers = 2,
        int numHeads = 8,
        int vocabSize = 30522,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        PICKOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new PICKOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _useNativeMode = false;
        _numEntityTypes = numEntityTypes;
        _hiddenDim = hiddenDim;
        _numGcnLayers = numGcnLayers;
        _numHeads = numHeads;
        _vocabSize = vocabSize;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        ImageSize = imageSize;
        MaxSequenceLength = maxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a PICK model using native layers for training and inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Default Configuration (PICK from ICPR 2020):</b>
    /// - BERT-based text encoder
    /// - 2-layer Graph Convolutional Network
    /// - BiLSTM for sequence modeling
    /// - CRF decoder for NER
    /// - Hidden dimension: 256
    /// </para>
    /// </remarks>
    public PICK(
        NeuralNetworkArchitecture<T> architecture,
        ITokenizer? tokenizer = null,
        int numEntityTypes = 14,
        int imageSize = 512,
        int maxSequenceLength = 512,
        int hiddenDim = 256,
        int numGcnLayers = 2,
        int numHeads = 8,
        int vocabSize = 30522,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        PICKOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new PICKOptions();
        Options = _options;

        _useNativeMode = true;
        _numEntityTypes = numEntityTypes;
        _hiddenDim = hiddenDim;
        _numGcnLayers = numGcnLayers;
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

        Layers.AddRange(LayerHelper<T>.CreateDefaultPICKLayers(
            hiddenDim: _hiddenDim,
            numGcnLayers: _numGcnLayers,
            numHeads: _numHeads,
            vocabSize: _vocabSize,
            numEntityTypes: _numEntityTypes,
            maxSequenceLength: MaxSequenceLength));
    }

    #endregion

    #region IFormUnderstanding Implementation

    /// <inheritdoc/>
    public FormFieldResult<T> ExtractFormFields(Tensor<T> documentImage)
    {
        return ExtractFormFields(documentImage, 0.5);
    }

    /// <inheritdoc/>
    public FormFieldResult<T> ExtractFormFields(Tensor<T> documentImage, double confidenceThreshold)
    {
        ValidateImageShape(documentImage);
        var startTime = DateTime.UtcNow;

        var preprocessed = PreprocessDocument(documentImage);
        var output = _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);

        var fields = ParseFieldOutput(output, confidenceThreshold);

        return new FormFieldResult<T>
        {
            Fields = fields,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public Dictionary<string, string> ExtractKeyValuePairs(Tensor<T> documentImage)
    {
        var result = ExtractFormFields(documentImage);
        var pairs = new Dictionary<string, string>();

        foreach (var field in result.Fields)
        {
            if (!string.IsNullOrEmpty(field.FieldName) && !string.IsNullOrEmpty(field.FieldValue))
            {
                pairs[field.FieldName] = field.FieldValue;
            }
        }

        return pairs;
    }

    /// <inheritdoc/>
    public IEnumerable<CheckboxResult<T>> DetectCheckboxes(Tensor<T> documentImage)
    {
        // PICK is designed for text extraction, not checkbox detection
        yield break;
    }

    /// <inheritdoc/>
    public IEnumerable<SignatureResult<T>> DetectSignatures(Tensor<T> documentImage)
    {
        // PICK is designed for text extraction, not signature detection
        yield break;
    }

    private List<FormField<T>> ParseFieldOutput(Tensor<T> output, double threshold)
    {
        var fields = new List<FormField<T>>();
        int seqLen = output.Shape[0];
        int numClasses = output.Shape.Length > 1 ? output.Shape[1] : _numEntityTypes;

        for (int i = 0; i < seqLen; i++)
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
                string entityType = maxClass < SupportedEntityTypes.Count
                    ? SupportedEntityTypes[maxClass]
                    : "UNKNOWN";

                fields.Add(new FormField<T>
                {
                    FieldName = entityType,
                    FieldValue = $"[Token {i}]",
                    FieldType = entityType,
                    Confidence = NumOps.FromDouble(maxConf),
                    ConfidenceValue = maxConf,
                    BoundingBox = Vector<T>.Empty()
                });
            }
        }

        return fields;
    }

    /// <summary>
    /// Extracts key information entities from a document.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Key information extraction result.</returns>
    public KeyInfoExtractionResult<T> ExtractKeyInfo(Tensor<T> documentImage)
    {
        var formResult = ExtractFormFields(documentImage);

        var entities = formResult.Fields.Select(f => new ExtractedEntity<T>
        {
            Label = f.FieldName,
            Text = f.FieldValue,
            EntityType = f.FieldType,
            Confidence = f.Confidence,
            ConfidenceValue = f.ConfidenceValue,
            BoundingBox = f.BoundingBox
        }).ToList();

        return new KeyInfoExtractionResult<T>
        {
            Entities = entities,
            ProcessingTimeMs = formResult.ProcessingTimeMs
        };
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
        sb.AppendLine("PICK Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine($"Architecture: BERT + Graph Convolutional Network");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"GCN Layers: {_numGcnLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads}");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine($"Number of Entity Types: {_numEntityTypes}");
        sb.AppendLine($"Supported Entity Types: {string.Join(", ", SupportedEntityTypes.Take(5))}...");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <summary>
    /// Applies PICK's industry-standard preprocessing: pass-through (PICK works with text + bbox input).
    /// </summary>
    /// <remarks>
    /// PICK (ICPR 2020) primarily processes text and bounding box features rather than raw images.
    /// </remarks>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        // PICK works with text + bbox input, preprocessing is mainly for compatibility
        return rawImage;
    }

    /// <summary>
    /// Applies PICK's industry-standard postprocessing: pass-through (entity extraction outputs are already final).
    /// </summary>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "PICK",
            ModelType = ModelType.NeuralNetwork,
            Description = "PICK for key information extraction (ICPR 2020)",
            FeatureCount = _hiddenDim,
            Complexity = _numGcnLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_gcn_layers", _numGcnLayers },
                { "num_heads", _numHeads },
                { "vocab_size", _vocabSize },
                { "num_entity_types", _numEntityTypes },
                { "use_native_mode", _useNativeMode }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numGcnLayers);
        writer.Write(_numHeads);
        writer.Write(_vocabSize);
        writer.Write(ImageSize);
        writer.Write(MaxSequenceLength);
        writer.Write(_numEntityTypes);
        writer.Write(_useNativeMode);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numGcnLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int vocabSize = reader.ReadInt32();
        int imageSize = reader.ReadInt32();
        int maxSeqLen = reader.ReadInt32();
        int numEntityTypes = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        ImageSize = imageSize;
        MaxSequenceLength = maxSeqLen;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new PICK<T>(Architecture, _tokenizer, _numEntityTypes, ImageSize, MaxSequenceLength,
            _hiddenDim, _numGcnLayers, _numHeads, _vocabSize);
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

/// <summary>
/// Result of key information extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class KeyInfoExtractionResult<T>
{
    /// <summary>
    /// Gets or sets the extracted entities.
    /// </summary>
    public IList<ExtractedEntity<T>> Entities { get; set; } = [];

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; set; }
}

/// <summary>
/// An extracted entity from a document.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExtractedEntity<T>
{
    /// <summary>
    /// Gets or sets the entity label.
    /// </summary>
    public string Label { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the extracted text.
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the entity type.
    /// </summary>
    public string EntityType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score.
    /// </summary>
    public T Confidence { get; set; } = default!;

    /// <summary>
    /// Gets or sets the confidence as a double.
    /// </summary>
    public double ConfidenceValue { get; set; }

    /// <summary>
    /// Gets or sets the bounding box.
    /// </summary>
    public Vector<T> BoundingBox { get; set; } = Vector<T>.Empty();
}
