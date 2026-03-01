using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.NER.Options;

/// <summary>
/// Configuration options for the CNN-BiLSTM-CRF Named Entity Recognition model.
/// </summary>
/// <remarks>
/// <para>
/// CNN-BiLSTM-CRF (Ma and Hovy, ACL 2016 - "End-to-end Sequence Labeling via Bi-directional
/// LSTM-CNNs-CRF") extends the BiLSTM-CRF architecture with character-level Convolutional Neural
/// Network (CNN) embeddings. The architecture has three main components:
///
/// 1. <b>Character-level CNN:</b> A 1D CNN processes each word's character sequence to capture
///    morphological features (capitalization, prefixes, suffixes, word shape). Unlike the
///    character BiLSTM in Lample et al. (2016), a CNN is faster and better at capturing local
///    n-gram patterns. The CNN output is max-pooled to produce a fixed-size vector per word.
///
/// 2. <b>BiLSTM encoder:</b> Processes the concatenation of word embeddings and character CNN
///    features in both forward and backward directions. Each token gets a representation
///    that captures its full sentence context plus its morphological features.
///
/// 3. <b>CRF decoder:</b> Models label transition dependencies to produce globally optimal
///    label sequences, enforcing valid BIO constraints via the Viterbi algorithm.
///
/// Default values follow the original Ma and Hovy (2016) paper:
/// - 100-dimensional GloVe word embeddings
/// - Character CNN with 30-dimensional embeddings, 30 filters of width 3
/// - Single BiLSTM layer with 200 hidden units per direction
/// - 50% dropout rate
/// - 9 CoNLL-2003 BIO labels
/// </para>
/// <para>
/// <b>For Beginners:</b> CNN-BiLSTM-CRF is like BiLSTM-CRF but with an extra component that
/// looks at the letters within each word. A CNN (Convolutional Neural Network) slides a small
/// window across the characters of each word to detect patterns like capitalization ("John" vs
/// "john"), suffixes ("-tion", "-ing"), and word shapes. These character features are combined
/// with word embeddings before being fed into the BiLSTM.
///
/// Compared to BiLSTM-CRF with character BiLSTM:
/// - CNN is faster to compute (parallel vs sequential processing)
/// - CNN is better at capturing local character n-grams (suffixes, prefixes)
/// - BiLSTM is better at capturing long-range character dependencies (rare)
/// - In practice, both achieve similar NER accuracy (~91% F1 on CoNLL-2003)
/// </para>
/// </remarks>
public class CNNBiLSTMCRFOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values matching the original Ma and Hovy (2016) paper.
    /// </summary>
    public CNNBiLSTMCRFOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by deep-copying all settings from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para>
    /// All mutable state is deep-copied to prevent mutations on one instance from affecting
    /// the other. This includes the LabelNames array and OnnxOptions object.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates an independent copy of an existing options object.
    /// Changing the copy won't affect the original, and vice versa.
    /// </para>
    /// </remarks>
    public CNNBiLSTMCRFOptions(CNNBiLSTMCRFOptions other)
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
        CharEmbeddingDimension = other.CharEmbeddingDimension;
        CharCNNFilters = other.CharCNNFilters;
        CharCNNKernelSize = other.CharCNNKernelSize;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        LabelNames = [.. other.LabelNames];
    }

    #region Architecture

    /// <summary>
    /// Gets or sets the model size variant, which controls the overall capacity of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different variants provide different tradeoffs between accuracy and computational cost:
    /// - <b>Tiny:</b> 100 hidden units, fastest inference, suitable for mobile/edge deployment
    /// - <b>Small:</b> 150 hidden units, good balance for real-time applications
    /// - <b>Base:</b> 200 hidden units (default), matches the original Ma and Hovy (2016) paper
    /// - <b>Large:</b> 300 hidden units, higher accuracy for offline processing
    /// - <b>XLarge:</b> 512 hidden units, maximum accuracy for research/evaluation
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this like T-shirt sizes. "Base" is the standard size that
    /// works well for most tasks. "Large" is for when you have more computing power and want
    /// better accuracy. "Tiny" is for when speed matters more than accuracy.
    /// </para>
    /// </remarks>
    public NERModelVariant Variant { get; set; } = NERModelVariant.Base;

    /// <summary>
    /// Gets or sets the input token embedding dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This must match the dimension of the pre-trained word embeddings you feed into the model.
    /// The original Ma and Hovy (2016) paper uses 100-dimensional GloVe embeddings.
    ///
    /// Common values:
    /// - <b>100:</b> GloVe-100d (used in the original paper)
    /// - <b>300:</b> GloVe-300d or Word2Vec-300d (standard for NER research)
    /// - <b>768:</b> BERT-base hidden states (for hybrid BERT+CNN-BiLSTM-CRF models)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the word vectors you're using as input.
    /// If you downloaded GloVe-100d embeddings, set this to 100. The number must match exactly.
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
    /// Gets or sets the LSTM hidden state dimension per direction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the number of hidden units in each LSTM direction. The original Ma and Hovy (2016)
    /// paper uses 200 hidden units per direction, which is larger than the 100 units used by
    /// Lample et al. (2016) in BiLSTM-CRF. The larger hidden dimension compensates for the
    /// richer character-level features provided by the CNN.
    ///
    /// Recommended values:
    /// - <b>200:</b> Original paper default (F1 ~91.2% on CoNLL-2003)
    /// - <b>256:</b> Common in practice, slightly better for complex datasets
    /// - <b>100:</b> Sufficient for simpler tasks with smaller datasets
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how much information the model can remember about
    /// each word's context. The default of 200 follows the original paper. Increase to 256
    /// if you have a large dataset, or decrease to 100 for faster training on small datasets.
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
    private int _hiddenDimension = 200;

    /// <summary>
    /// Gets or sets the number of stacked BiLSTM layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The original Ma and Hovy (2016) paper uses a single BiLSTM layer, which is sufficient
    /// for most NER tasks when combined with character CNN features.
    ///
    /// - <b>1:</b> Standard configuration (recommended)
    /// - <b>2:</b> Can help on large, complex datasets (e.g., OntoNotes)
    /// - <b>3+:</b> Diminishing returns; use with strong regularization
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> More layers means more complex patterns can be learned, but also
    /// needs more data. Start with 1 (the default) and only increase if the model underfits.
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
    /// Must match the number of entries in <see cref="LabelNames"/>.
    /// Default is 9 for the CoNLL-2003 BIO scheme:
    /// O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.
    ///
    /// For custom entity types, use: numLabels = 2 * numEntityTypes + 1 (BIO scheme).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how many different labels the model outputs. The default
    /// of 9 handles 4 entity types (person, organization, location, miscellaneous).
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
    /// Sentences exceeding this length will be truncated. The CRF layer's internal sequence
    /// dimension is set to this value.
    ///
    /// - <b>128:</b> Sufficient for most sentence-level NER
    /// - <b>256:</b> Default, handles paragraph-level text
    /// - <b>512:</b> For document-level NER with very long sentences
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the maximum number of words the model can handle in one
    /// sentence. The default of 256 is plenty for most use cases.
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
    /// Gets or sets whether to use CRF (Conditional Random Field) decoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true (default), the model uses a CRF layer for structured prediction that enforces
    /// valid BIO label transitions. This typically improves F1 score by 1-2%.
    /// Set to false for faster inference when structured constraints are not needed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Keep this set to true for best accuracy. The CRF ensures the
    /// model's predictions follow valid BIO labeling rules.
    /// </para>
    /// </remarks>
    public bool UseCRF { get; set; } = true;

    /// <summary>
    /// Gets or sets the character embedding vector dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each character (a-z, A-Z, 0-9, punctuation) is mapped to a dense vector of this dimension.
    /// The original Ma and Hovy (2016) paper uses 30-dimensional character embeddings, which are
    /// fed into the 1D CNN to extract character-level features.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Just like word embeddings represent words as numbers, character
    /// embeddings represent individual letters as numbers. The default of 30 works well.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int CharEmbeddingDimension
    {
        get => _charEmbeddingDimension;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(CharEmbeddingDimension),
                    $"Character embedding dimension must be positive. Got: {value}");
            _charEmbeddingDimension = value;
        }
    }
    private int _charEmbeddingDimension = 30;

    /// <summary>
    /// Gets or sets the number of CNN filters for character-level feature extraction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of 1D convolutional filters applied to the character embedding sequence.
    /// Each filter learns to detect a specific character n-gram pattern (e.g., capitalization,
    /// common suffixes like "-tion" or "-ing", number patterns like "2023").
    ///
    /// The original Ma and Hovy (2016) paper uses 30 filters. After max-pooling over the
    /// character sequence, the output is a vector of size CharCNNFilters that captures the
    /// most salient character-level features for each word.
    ///
    /// This value determines the size of the character feature vector that is concatenated
    /// with word embeddings, making the effective input to the BiLSTM:
    /// EmbeddingDimension + CharCNNFilters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each filter detects one type of character pattern. 30 filters
    /// means the model can detect 30 different character patterns simultaneously. The default
    /// of 30 follows the original paper and captures common patterns like capitalization,
    /// suffixes, and word shapes.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int CharCNNFilters
    {
        get => _charCNNFilters;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(CharCNNFilters),
                    $"Number of character CNN filters must be positive. Got: {value}");
            _charCNNFilters = value;
        }
    }
    private int _charCNNFilters = 30;

    /// <summary>
    /// Gets or sets the kernel (window) size for the character CNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The kernel size determines how many consecutive characters the CNN examines at once.
    /// The original Ma and Hovy (2016) paper uses a kernel size of 3, which captures
    /// character trigrams (3-character patterns). This is effective for detecting:
    /// - Capitalization patterns (e.g., "The", "THE", "the")
    /// - Common suffixes (e.g., "ing", "tion", "ous")
    /// - Common prefixes (e.g., "pre", "un-", "re-")
    /// - Number patterns (e.g., "199", "202")
    ///
    /// Larger kernel sizes capture longer patterns but are less computationally efficient.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The kernel size is like a magnifying glass that looks at a few
    /// characters at a time. A size of 3 means it looks at 3 characters simultaneously (e.g.,
    /// "ing", "the", "ohn"). The default of 3 works well for English NER.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when value is not positive.</exception>
    public int CharCNNKernelSize
    {
        get => _charCNNKernelSize;
        set
        {
            if (value <= 0)
                throw new ArgumentOutOfRangeException(nameof(CharCNNKernelSize),
                    $"Character CNN kernel size must be positive. Got: {value}");
            _charCNNKernelSize = value;
        }
    }
    private int _charCNNKernelSize = 3;

    /// <summary>
    /// Gets or sets the BIO label names for the tagging scheme.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Maps label indices to human-readable label names. Length must equal <see cref="NumLabels"/>.
    /// Default follows CoNLL-2003.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These are the names of the entity types your model will recognize.
    /// Customize this for your domain (e.g., drug names and diseases for medical text).
    /// </para>
    /// </remarks>
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
    /// Gets or sets the path to a pre-trained ONNX model file for inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When set, the model loads weights from this ONNX file instead of using native layers.
    /// Leave null for native training mode.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If you have a pre-trained model file (ending in .onnx), set this
    /// path to load it. Leave null to train from scratch.
    /// </para>
    /// </remarks>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX Runtime configuration options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls ONNX Runtime settings like execution providers (CPU, CUDA, DirectML).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These settings control how the ONNX model runs. The defaults
    /// work well for most cases.
    /// </para>
    /// </remarks>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>
    /// Gets or sets the learning rate for the optimizer during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The original Ma and Hovy (2016) paper uses SGD with a learning rate of 0.015 and
    /// learning rate decay. Modern implementations typically use Adam/AdamW with a lower rate.
    ///
    /// Recommended values:
    /// - <b>1e-3:</b> Default for AdamW optimizer
    /// - <b>1.5e-2:</b> For SGD optimizer (matching original paper)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate controls how fast the model learns. The default
    /// of 0.001 works well with AdamW. If training is unstable, try halving it.
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
    /// Gets or sets the dropout rate for regularization during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The original Ma and Hovy (2016) paper uses 0.5 dropout, applied on word embeddings,
    /// character embeddings, and the BiLSTM output before the CRF layer.
    /// Must be between 0.0 and 1.0.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Dropout prevents overfitting by randomly turning off neurons
    /// during training. The default of 0.5 follows the original paper.
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
