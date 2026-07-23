using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.NER.Options;

/// <summary>
/// Base configuration options shared by all transformer-based NER models (BERT-NER, RoBERTa-NER,
/// DeBERTa-NER, ELECTRA-NER, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Transformer-based NER models share a common architecture: a pre-trained transformer encoder
/// produces contextualized token representations, followed by a token classification head that
/// predicts BIO labels for each token. The key differences between variants are in the
/// pre-training strategy and attention mechanism, not the NER fine-tuning architecture.
///
/// The standard transformer NER architecture consists of:
/// 1. <b>Transformer encoder:</b> Multi-layer self-attention with feed-forward networks.
///    Each layer has multi-head attention (captures token-to-token relationships) and a
///    feed-forward network (transforms each token's representation independently).
/// 2. <b>Classification head:</b> A linear projection from hidden_size to num_labels,
///    optionally with a CRF layer for structured prediction.
/// 3. <b>Optional CRF:</b> Can be added on top of the transformer for structured decoding.
///
/// This base options class provides the shared parameters. Variant-specific options classes
/// (BERTNEROptions, RoBERTaNEROptions, etc.) can extend this with model-specific settings.
/// </para>
/// <para>
/// <b>For Beginners:</b> Transformer models are the most powerful NER architectures available.
/// They read text using "self-attention" - each word looks at every other word in the sentence
/// to understand context. This is more powerful than BiLSTM-CRF because transformers can
/// capture long-range dependencies (e.g., a pronoun referring to an entity mentioned 50 words ago).
///
/// The tradeoff is that transformer models are much larger (110M+ parameters for BERT-base vs
/// ~1M for BiLSTM-CRF) and slower to train. However, they achieve state-of-the-art accuracy
/// on all NER benchmarks.
/// </para>
/// </remarks>
public class TransformerNEROptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with BERT-base defaults.
    /// </summary>
    public TransformerNEROptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by deep-copying all settings from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TransformerNEROptions(TransformerNEROptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        NumTransformerLayers = other.NumTransformerLayers;
        IntermediateDimension = other.IntermediateDimension;
        NumLabels = other.NumLabels;
        MaxSequenceLength = other.MaxSequenceLength;
        UseCRF = other.UseCRF;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        LabelNames = [.. other.LabelNames];
    }

    #region Architecture

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Transformer NER variants map to standard transformer sizes:
    /// - <b>Tiny:</b> 2 layers, 128 hidden, 2 heads (~4M params)
    /// - <b>Small:</b> 4 layers, 256 hidden, 4 heads (~14M params)
    /// - <b>Base:</b> 12 layers, 768 hidden, 12 heads (~110M params, BERT-base default)
    /// - <b>Large:</b> 24 layers, 1024 hidden, 16 heads (~340M params, BERT-large)
    /// - <b>XLarge:</b> 36 layers, 1280 hidden, 20 heads (~680M params)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> "Base" is the standard size that works well. "Large" gives better
    /// accuracy but needs more memory and compute.
    /// </para>
    /// </remarks>
    public NERModelVariant Variant { get; set; } = NERModelVariant.Base;

    /// <summary>
    /// Gets or sets the hidden dimension of the transformer encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is both the input embedding dimension and the hidden state dimension throughout
    /// the transformer. Unlike BiLSTM-CRF where embedding and hidden dimensions can differ,
    /// transformers use a single dimension for all internal representations.
    ///
    /// Standard values:
    /// - <b>768:</b> BERT-base, RoBERTa-base, DeBERTa-base, ELECTRA-base
    /// - <b>1024:</b> BERT-large, RoBERTa-large, DeBERTa-large
    /// - <b>256:</b> DistilBERT-base, TinyBERT
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the internal representations. Larger values
    /// capture more information but require more memory. 768 is the standard for base models.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int HiddenDimension
    {
        get => _hiddenDimension;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(HiddenDimension),
                    $"Hidden dimension must be positive. Got: {value}");
            _hiddenDimension = value;
        }
    }
    private int _hiddenDimension = 768;

    /// <summary>
    /// Gets or sets the number of self-attention heads in each transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multi-head attention allows the model to attend to different types of relationships
    /// simultaneously. Each head has dimension = HiddenDimension / NumAttentionHeads.
    ///
    /// Standard values:
    /// - <b>12:</b> BERT-base (768/12 = 64 per head)
    /// - <b>16:</b> BERT-large (1024/16 = 64 per head)
    ///
    /// HiddenDimension must be divisible by NumAttentionHeads.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each attention head looks at the text from a different perspective.
    /// One head might focus on syntactic relationships (subject-verb), another on semantic
    /// relationships (synonyms), etc. More heads means more perspectives.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int NumAttentionHeads
    {
        get => _numAttentionHeads;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumAttentionHeads),
                    $"Number of attention heads must be positive. Got: {value}");
            _numAttentionHeads = value;
        }
    }
    private int _numAttentionHeads = 12;

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More layers capture more complex patterns at the cost of compute:
    /// - <b>6:</b> DistilBERT (distilled from 12 layers)
    /// - <b>12:</b> BERT-base (standard)
    /// - <b>24:</b> BERT-large
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each layer processes the text through attention and feed-forward
    /// networks. Lower layers capture basic features, higher layers capture complex patterns.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int NumTransformerLayers
    {
        get => _numTransformerLayers;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumTransformerLayers),
                    $"Number of transformer layers must be positive. Got: {value}");
            _numTransformerLayers = value;
        }
    }
    private int _numTransformerLayers = 12;

    /// <summary>
    /// Gets or sets the intermediate (feed-forward) dimension in each transformer layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The feed-forward network in each transformer layer expands the representation to this
    /// dimension, applies a non-linearity (GELU), then projects back to HiddenDimension.
    /// Typically 4x the hidden dimension.
    ///
    /// - <b>3072:</b> BERT-base (768 * 4)
    /// - <b>4096:</b> BERT-large (1024 * 4)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls the size of an internal expansion layer. The
    /// transformer briefly expands each token's representation to capture more complex
    /// patterns, then compresses it back down. Default of 3072 follows BERT-base.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int IntermediateDimension
    {
        get => _intermediateDimension;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(IntermediateDimension),
                    $"Intermediate dimension must be positive. Got: {value}");
            _intermediateDimension = value;
        }
    }
    private int _intermediateDimension = 3072;

    /// <summary>
    /// Gets or sets the number of entity label classes in the BIO tagging scheme.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int NumLabels
    {
        get => _numLabels;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumLabels),
                    $"Number of labels must be positive. Got: {value}");
            _numLabels = value;
        }
    }
    private int _numLabels = 9;

    /// <summary>
    /// Gets or sets the maximum input sequence length in tokens.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Transformer models have a fixed maximum sequence length due to positional embeddings.
    /// Standard values:
    /// - <b>128:</b> Short texts, fast inference
    /// - <b>256:</b> Default for most NER tasks
    /// - <b>512:</b> BERT/RoBERTa maximum, for long documents
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Maximum number of tokens the model can process at once.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int MaxSequenceLength
    {
        get => _maxSequenceLength;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(MaxSequenceLength),
                    $"Maximum sequence length must be positive. Got: {value}");
            _maxSequenceLength = value;
        }
    }
    private int _maxSequenceLength = 256;

    /// <summary>
    /// Gets or sets whether to use CRF decoding on top of the transformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Adding CRF on top of a transformer provides a modest improvement (0.3-0.5% F1) for
    /// NER by enforcing valid BIO transitions. However, the improvement is smaller than for
    /// BiLSTM-CRF because transformers already capture sequence-level patterns through
    /// self-attention. Default is false for pure transformer inference.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> CRF is less important for transformer models than for LSTM models.
    /// Set to true if you want maximum accuracy at the cost of slightly slower inference.
    /// </para>
    /// </remarks>
    public bool UseCRF { get; set; } = false;

    /// <summary>
    /// Gets or sets the BIO label names.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when value is null or empty.</exception>
    public string[] LabelNames
    {
        get => _labelNames;
        set
        {
            if (value is null || value.Length == 0)
                throw new ArgumentException("Label names array cannot be null or empty.", nameof(LabelNames));
            _labelNames = value;
        }
    }
    private string[] _labelNames =
    [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ];

    #endregion

    #region Model Loading

    /// <summary>
    /// Gets or sets the path to a pre-trained ONNX model file.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX Runtime configuration options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Transformer NER models use much lower learning rates than LSTM models because they
    /// are fine-tuning pre-trained weights rather than training from scratch.
    ///
    /// Recommended values:
    /// - <b>5e-5:</b> Standard for BERT/RoBERTa fine-tuning (original BERT paper)
    /// - <b>3e-5:</b> More conservative, lower risk of catastrophic forgetting
    /// - <b>2e-5:</b> For smaller datasets or when fine-tuning is unstable
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This must be much smaller than for LSTM models (0.00005 vs 0.001)
    /// because transformers have pre-trained weights that we want to adjust gently, not overwrite.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public double LearningRate
    {
        get => _learningRate;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(LearningRate),
                    $"Learning rate must be positive. Got: {value}");
            _learningRate = value;
        }
    }
    private double _learningRate = 5e-5;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Applied in attention layers and feed-forward networks. BERT uses 0.1, which is lower
    /// than the 0.5 used in BiLSTM-CRF because transformer models are already heavily regularized
    /// by the pre-training objective.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The default of 0.1 follows the original BERT paper. Transformer
    /// models need less dropout because they're pre-trained on massive data.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is outside [0, 1].</exception>
    public double DropoutRate
    {
        get => _dropoutRate;
        set
        {
            if (value < 0 || value > 1)
                throw new ArgumentOutOfRangeException(nameof(DropoutRate),
                    $"Dropout rate must be between 0.0 and 1.0. Got: {value}");
            _dropoutRate = value;
        }
    }
    private double _dropoutRate = 0.1;

    #endregion
}
