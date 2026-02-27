using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.NER.Options;

/// <summary>
/// Configuration options for the LSTM-CRF Named Entity Recognition model.
/// </summary>
/// <remarks>
/// <para>
/// LSTM-CRF (Huang, Xu, and Yu, 2015 - "Bidirectional LSTM-CRF Models for Sequence Tagging")
/// is a simpler variant that uses unidirectional LSTM with a CRF layer. While the original paper
/// actually proposed both unidirectional and bidirectional variants, this class implements the
/// unidirectional version for scenarios where lower latency or streaming inference is needed.
///
/// The architecture consists of:
///
/// 1. <b>Unidirectional LSTM:</b> Processes the token sequence left-to-right only, providing
///    each token with context from preceding words. This enables streaming/online inference
///    since each token can be processed as soon as it arrives, without waiting for the full
///    sentence. The tradeoff is that right-context is not available, which reduces accuracy
///    by 1-2% F1 compared to BiLSTM-CRF.
///
/// 2. <b>CRF decoder:</b> Models label transition dependencies to produce globally optimal
///    label sequences. The CRF is especially important for unidirectional models because it
///    partially compensates for the lack of right-context by leveraging label sequence patterns.
///
/// Default values:
/// - 100-dimensional word embeddings
/// - Single LSTM layer with 100 hidden units
/// - 50% dropout rate
/// - 9 CoNLL-2003 BIO labels
/// </para>
/// <para>
/// <b>For Beginners:</b> LSTM-CRF is a simpler, faster version of BiLSTM-CRF that only reads
/// text in one direction (left to right). It's like reading a sentence without being able to
/// look ahead. This makes it faster for real-time applications but slightly less accurate.
///
/// Use LSTM-CRF when:
/// - You need lower latency for real-time NER
/// - You're processing streaming text (words arriving one at a time)
/// - You want a simpler model with fewer parameters
/// - Slightly lower accuracy is acceptable
/// </para>
/// </remarks>
public class LSTMCRFOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public LSTMCRFOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by deep-copying all settings from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para>
    /// All mutable state is deep-copied to prevent shared mutation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Creates an independent copy. Changes to the copy won't affect the original.
    /// </para>
    /// </remarks>
    public LSTMCRFOptions(LSTMCRFOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        EmbeddingDimension = other.EmbeddingDimension;
        HiddenDimension = other.HiddenDimension;
        NumLSTMLayers = other.NumLSTMLayers;
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

    /// <inheritdoc cref="BiLSTMCRFOptions.Variant" />
    public NERModelVariant Variant { get; set; } = NERModelVariant.Base;

    /// <summary>
    /// Gets or sets the input token embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Must match the dimension of the pre-trained word embeddings.
    /// Common values: 100 (GloVe-100d), 300 (GloVe-300d), 768 (BERT-base).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the word vectors you're using as input.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int EmbeddingDimension
    {
        get => _embeddingDimension;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(EmbeddingDimension),
                    $"Embedding dimension must be positive. Got: {value}");
            _embeddingDimension = value;
        }
    }
    private int _embeddingDimension = 100;

    /// <summary>
    /// Gets or sets the LSTM hidden state dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Since this is a unidirectional LSTM (not bidirectional), the full hidden state is used
    /// directly. The default of 100 matches the original paper's configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Controls how much information the model remembers about context.
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
    private int _hiddenDimension = 100;

    /// <summary>
    /// Gets or sets the number of stacked LSTM layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For unidirectional LSTM-CRF, 1-2 layers is typical.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Start with 1 layer. Add more only with large datasets.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int NumLSTMLayers
    {
        get => _numLSTMLayers;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumLSTMLayers),
                    $"Number of LSTM layers must be positive. Got: {value}");
            _numLSTMLayers = value;
        }
    }
    private int _numLSTMLayers = 1;

    /// <summary>
    /// Gets or sets the number of entity label classes in the BIO tagging scheme.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default is 9 for CoNLL-2003 BIO scheme. Formula: numLabels = 2 * numEntityTypes + 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Number of different labels the model can assign to each word.
    /// </para>
    /// </remarks>
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
    /// Sentences exceeding this will be truncated. Default of 256 handles most use cases.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Maximum number of words the model can process per sentence.
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
    /// Gets or sets whether to use CRF decoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CRF is especially important for unidirectional models because it partially compensates
    /// for the lack of right-context by enforcing valid BIO transitions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Keep true. CRF is even more important here than in BiLSTM-CRF
    /// because the model can't look ahead, so the CRF helps ensure consistent predictions.
    /// </para>
    /// </remarks>
    public bool UseCRF { get; set; } = true;

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

    /// <inheritdoc cref="BiLSTMCRFOptions.ModelPath" />
    public string? ModelPath { get; set; }

    /// <inheritdoc cref="BiLSTMCRFOptions.OnnxOptions" />
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for the optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default of 1e-3 for AdamW optimizer. For SGD, use 1e-2.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Controls how fast the model learns. Default works for most cases.
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
    private double _learningRate = 1e-3;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Must be between 0.0 and 1.0. Default of 0.5 follows standard NER practice.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Prevents overfitting by randomly turning off neurons during training.
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
    private double _dropoutRate = 0.5;

    #endregion
}
