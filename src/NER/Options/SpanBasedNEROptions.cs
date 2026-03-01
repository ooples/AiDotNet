using AiDotNet.Enums;
using AiDotNet.Onnx;

namespace AiDotNet.NER.Options;

/// <summary>
/// Options shared by span-based NER models (SpERT, BiaffineNER, PURE).
/// </summary>
/// <remarks>
/// Span-based NER models enumerate all possible spans in a sentence and classify each span
/// as an entity type or non-entity. This contrasts with sequence labeling (BIO tagging) where
/// each token is independently labeled. Span-based models handle nested entities naturally
/// because overlapping spans can each have different labels.
/// </remarks>
public class SpanBasedNEROptions
{
    #region Fields

    private int _hiddenDimension = 768;
    private int _numAttentionHeads = 12;
    private int _numTransformerLayers = 12;
    private int _intermediateDimension = 3072;
    private int _numLabels = 9;
    private int _maxSequenceLength = 256;
    private int _maxSpanLength = 10;
    private int _spanEmbeddingDimension = 256;
    private double _dropoutRate = 0.1;
    private double _learningRate = 5e-5;
    private int _negativeSpanSampleRatio = 100;

    #endregion

    #region Properties

    /// <summary>
    /// Gets or sets the hidden dimension of the encoder.
    /// </summary>
    public int HiddenDimension
    {
        get => _hiddenDimension;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "HiddenDimension must be positive.");
            _hiddenDimension = value;
        }
    }

    /// <summary>
    /// Gets or sets the number of attention heads in each transformer layer.
    /// </summary>
    public int NumAttentionHeads
    {
        get => _numAttentionHeads;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "NumAttentionHeads must be positive.");
            _numAttentionHeads = value;
        }
    }

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// </summary>
    public int NumTransformerLayers
    {
        get => _numTransformerLayers;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "NumTransformerLayers must be positive.");
            _numTransformerLayers = value;
        }
    }

    /// <summary>
    /// Gets or sets the feed-forward intermediate dimension.
    /// </summary>
    public int IntermediateDimension
    {
        get => _intermediateDimension;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "IntermediateDimension must be positive.");
            _intermediateDimension = value;
        }
    }

    /// <summary>
    /// Gets or sets the number of entity labels (including O tag).
    /// </summary>
    public int NumLabels
    {
        get => _numLabels;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "NumLabels must be positive.");
            _numLabels = value;
        }
    }

    /// <summary>
    /// Gets or sets the maximum sequence length in tokens.
    /// </summary>
    public int MaxSequenceLength
    {
        get => _maxSequenceLength;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "MaxSequenceLength must be positive.");
            _maxSequenceLength = value;
        }
    }

    /// <summary>
    /// Gets or sets the maximum span length (number of tokens in a single entity span).
    /// </summary>
    /// <remarks>
    /// Most named entities are 1-10 tokens long. Setting this value appropriately reduces
    /// the number of candidate spans that need to be evaluated. A value of 10 covers
    /// ~99% of entities in most NER datasets.
    /// </remarks>
    public int MaxSpanLength
    {
        get => _maxSpanLength;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "MaxSpanLength must be positive.");
            _maxSpanLength = value;
        }
    }

    /// <summary>
    /// Gets or sets the span embedding dimension used for span classification.
    /// </summary>
    public int SpanEmbeddingDimension
    {
        get => _spanEmbeddingDimension;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "SpanEmbeddingDimension must be positive.");
            _spanEmbeddingDimension = value;
        }
    }

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    public double DropoutRate
    {
        get => _dropoutRate;
        set
        {
            if (value < 0 || value > 1) throw new ArgumentOutOfRangeException(nameof(value), "DropoutRate must be in [0, 1].");
            _dropoutRate = value;
        }
    }

    /// <summary>
    /// Gets or sets the learning rate for the optimizer.
    /// </summary>
    public double LearningRate
    {
        get => _learningRate;
        set
        {
            if (value <= 0) throw new ArgumentOutOfRangeException(nameof(value), "LearningRate must be positive.");
            _learningRate = value;
        }
    }

    /// <summary>
    /// Gets or sets the ratio of negative (non-entity) spans to positive (entity) spans
    /// sampled during training.
    /// </summary>
    /// <remarks>
    /// Since most spans in a sentence are not entities, training uses negative sampling.
    /// A ratio of 100 means for every entity span, 100 non-entity spans are sampled.
    /// </remarks>
    public int NegativeSpanSampleRatio
    {
        get => _negativeSpanSampleRatio;
        set
        {
            if (value < 1) throw new ArgumentOutOfRangeException(nameof(value), "NegativeSpanSampleRatio must be >= 1.");
            _negativeSpanSampleRatio = value;
        }
    }

    /// <summary>
    /// Gets or sets the NER model variant.
    /// </summary>
    public NERModelVariant Variant { get; set; } = NERModelVariant.Base;

    /// <summary>
    /// Gets or sets the path to an ONNX model file for inference mode.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets ONNX runtime inference options.
    /// </summary>
    public OnnxModelOptions OnnxOptions
    {
        get => _onnxOptions;
        set => _onnxOptions = value ?? throw new ArgumentNullException(nameof(value), "OnnxOptions cannot be null.");
    }
    private OnnxModelOptions _onnxOptions = new();

    /// <summary>
    /// Gets or sets the label names for the NER tags.
    /// </summary>
    public string[] LabelNames { get; set; } = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new instance with default settings (BERT-base defaults).
    /// </summary>
    public SpanBasedNEROptions() { }

    /// <summary>
    /// Deep-copy constructor.
    /// </summary>
    public SpanBasedNEROptions(SpanBasedNEROptions other)
    {
        _hiddenDimension = other._hiddenDimension;
        _numAttentionHeads = other._numAttentionHeads;
        _numTransformerLayers = other._numTransformerLayers;
        _intermediateDimension = other._intermediateDimension;
        _numLabels = other._numLabels;
        _maxSequenceLength = other._maxSequenceLength;
        _maxSpanLength = other._maxSpanLength;
        _spanEmbeddingDimension = other._spanEmbeddingDimension;
        _dropoutRate = other._dropoutRate;
        _learningRate = other._learningRate;
        _negativeSpanSampleRatio = other._negativeSpanSampleRatio;
        Variant = other.Variant;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        LabelNames = (string[])other.LabelNames.Clone();
    }

    #endregion
}
