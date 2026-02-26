using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.NER.Options;

/// <summary>
/// Configuration options for the BiLSTM-CRF Named Entity Recognition model.
/// </summary>
/// <remarks>
/// <para>
/// BiLSTM-CRF (Huang et al., 2015; Lample et al., NAACL 2016) combines bidirectional LSTM
/// with a Conditional Random Field layer for sequence labeling. The architecture consists of:
///
/// 1. <b>Word embeddings:</b> Pre-trained vectors (GloVe, Word2Vec) map words to dense vectors
///    that capture semantic meaning. Words with similar meanings have similar vectors.
///
/// 2. <b>Character embeddings (optional):</b> A small character-level LSTM processes each word's
///    characters to capture morphological features like capitalization (uppercase = likely entity),
///    prefixes (anti-, un-), and suffixes (-tion, -ing). This helps recognize unseen words.
///
/// 3. <b>BiLSTM encoder:</b> Processes the concatenated word+character embeddings in both forward
///    and backward directions. The forward LSTM reads left-to-right (capturing "John works at ...")
///    while the backward LSTM reads right-to-left (capturing "... at Google Inc."). Their outputs
///    are concatenated to give each token a representation informed by its full sentence context.
///
/// 4. <b>CRF decoder:</b> Models label transition dependencies. Instead of classifying each token
///    independently, the CRF finds the globally optimal label sequence using the Viterbi algorithm.
///    This ensures valid BIO transitions (e.g., I-PER can only follow B-PER or I-PER).
///
/// Default values follow the original paper (Lample et al., NAACL 2016):
/// - 100-dimensional GloVe word embeddings
/// - 25-dimensional character embeddings with 25-unit char LSTM
/// - Single BiLSTM layer with 100 hidden units per direction
/// - 50% dropout rate
/// - 9 CoNLL-2003 BIO labels
/// </para>
/// <para>
/// <b>For Beginners:</b> BiLSTM-CRF is the most widely-used neural NER architecture. It reads
/// text forwards and backwards to understand each word in context, then uses a CRF to pick the
/// best sequence of entity labels. These options let you configure the model's size, what
/// embeddings it expects, and how it trains.
///
/// The default settings are a good starting point for most NER tasks. If you need higher accuracy,
/// try increasing HiddenDimension to 256 or using 300-dimensional GloVe embeddings.
/// </para>
/// </remarks>
public class BiLSTMCRFOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values matching the original research paper.
    /// </summary>
    public BiLSTMCRFOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying all settings from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a copy of an existing options object. Useful when you
    /// want to create a variation of an existing configuration without modifying the original.
    /// </para>
    /// </remarks>
    public BiLSTMCRFOptions(BiLSTMCRFOptions other)
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
        UseCharEmbeddings = other.UseCharEmbeddings;
        CharEmbeddingDimension = other.CharEmbeddingDimension;
        CharHiddenDimension = other.CharHiddenDimension;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
        LabelNames = other.LabelNames;
    }

    #region Architecture

    /// <summary>
    /// Gets or sets the model size variant, which controls the overall capacity of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different variants provide different tradeoffs between accuracy and computational cost:
    /// - <b>Tiny:</b> 50 hidden units, fastest inference, suitable for mobile/edge deployment
    /// - <b>Small:</b> 100 hidden units, good balance for real-time applications
    /// - <b>Base:</b> 100 hidden units (default), matches the original paper configuration
    /// - <b>Large:</b> 256 hidden units, higher accuracy for offline processing
    /// - <b>XLarge:</b> 512 hidden units, maximum accuracy for research/evaluation
    ///
    /// The variant is primarily metadata; actual dimensions are controlled by HiddenDimension,
    /// NumLSTMLayers, etc. Use this to indicate which pre-configured size you're targeting.
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
    /// Common values:
    /// - <b>100:</b> GloVe-100d (used in the original Lample et al., 2016 paper)
    /// - <b>300:</b> GloVe-300d or Word2Vec-300d (standard for NER research)
    /// - <b>768:</b> BERT-base hidden states (for BERT+BiLSTM-CRF hybrid models)
    ///
    /// Higher dimensions capture more information about word meaning but increase model size
    /// and training time. 100d is sufficient for most NER tasks; 300d provides a modest improvement.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the word vectors you're using as input.
    /// If you downloaded GloVe-100d embeddings, set this to 100. If you're using GloVe-300d,
    /// set it to 300. The number must match exactly or the model will crash.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 100;

    /// <summary>
    /// Gets or sets the LSTM hidden state dimension per direction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the number of hidden units in each LSTM direction. Since BiLSTM-CRF uses
    /// bidirectional processing, the total output dimension after concatenation is
    /// 2 * HiddenDimension. The original paper uses 100 hidden units per direction (200 total).
    ///
    /// Recommended values:
    /// - <b>100:</b> Original paper default, good for CoNLL-2003 (F1 ~91%)
    /// - <b>200:</b> Common in practice, slightly better accuracy
    /// - <b>256:</b> Higher capacity, good for complex datasets like OntoNotes
    ///
    /// Larger hidden dimensions increase the model's capacity to represent complex patterns
    /// but also increase the risk of overfitting on small datasets.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how much information the model can remember about
    /// each word's context. A larger number means the model can capture more complex patterns
    /// but also takes longer to train. 100 is a good starting point; increase to 256 if you
    /// have a large dataset and want better accuracy.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of stacked BiLSTM layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The original Lample et al. (2016) paper uses a single BiLSTM layer, which is sufficient
    /// for most NER tasks. Stacking multiple layers can improve accuracy on complex datasets
    /// by learning hierarchical representations, but increases training time and overfitting risk.
    ///
    /// - <b>1:</b> Standard configuration (recommended for most use cases)
    /// - <b>2:</b> Can help on large, complex datasets (e.g., OntoNotes)
    /// - <b>3+:</b> Diminishing returns; use with strong regularization
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> More layers means the model can learn more complex patterns,
    /// but also needs more data to train well. Start with 1 layer (the default) and only
    /// increase if you have a large dataset and the model seems to underfit.
    /// </para>
    /// </remarks>
    public int NumLSTMLayers { get; set; } = 1;

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
    /// For example, 6 entity types would need 13 labels.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how many different labels the model outputs. The default
    /// of 9 handles 4 entity types (person, organization, location, miscellaneous). If you're
    /// building a model for a custom domain (e.g., medical entities), adjust this to match
    /// your label set.
    /// </para>
    /// </remarks>
    public int NumLabels { get; set; } = 9;

    /// <summary>
    /// Gets or sets the maximum input sequence length in tokens.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sentences exceeding this length will be truncated. The CRF layer's internal sequence
    /// dimension is set to this value. Most NER datasets have sentences under 50 tokens,
    /// so 256 provides generous headroom.
    ///
    /// - <b>128:</b> Sufficient for most sentence-level NER
    /// - <b>256:</b> Default, handles paragraph-level text
    /// - <b>512:</b> For document-level NER with very long sentences
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the maximum number of words the model can handle in one
    /// sentence. Most sentences are 10-50 words, so the default of 256 is plenty. Only increase
    /// this if you're processing very long paragraphs.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; set; } = 256;

    /// <summary>
    /// Gets or sets whether to use CRF (Conditional Random Field) decoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true (default), the model uses a CRF layer for structured prediction that enforces
    /// valid BIO label transitions. This typically improves F1 score by 1-2% compared to
    /// independent softmax classification.
    ///
    /// Set to false for faster inference when structured constraints are not needed,
    /// or for debugging purposes to isolate the BiLSTM's emission quality.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Keep this set to true for best accuracy. The CRF makes sure the
    /// model's predictions follow the rules of the BIO labeling scheme (e.g., "I-PER" can only
    /// come after "B-PER" or another "I-PER"). This prevents nonsensical label sequences.
    /// </para>
    /// </remarks>
    public bool UseCRF { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use character-level embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Character embeddings were introduced by Lample et al. (2016) to capture morphological
    /// features that word-level embeddings miss:
    /// - <b>Capitalization:</b> "Apple" (entity) vs "apple" (fruit)
    /// - <b>Prefixes/suffixes:</b> "-burg" (city), "-son" (person), "Dr." (title)
    /// - <b>Out-of-vocabulary words:</b> Rare names like "Krzyzewski" can be recognized by
    ///   character patterns even if they don't appear in the word embedding vocabulary
    ///
    /// A small character-level LSTM processes each word's characters and produces a fixed-size
    /// vector that is concatenated with the word embedding before feeding into the main BiLSTM.
    ///
    /// Note: Character embedding support is a configuration flag; the actual character-level
    /// LSTM layers are added during model initialization when this is enabled.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Character embeddings help the model understand words it has never
    /// seen before by looking at their letters. For example, even if "Krzyzewski" isn't in the
    /// model's vocabulary, the character patterns can help recognize it as a person name. This
    /// is especially useful for names from different languages or technical terms.
    /// </para>
    /// </remarks>
    public bool UseCharEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the character embedding vector dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each character (a-z, A-Z, 0-9, punctuation) is mapped to a dense vector of this dimension.
    /// The original paper uses 25-dimensional character embeddings, which is sufficient to capture
    /// character-level patterns. Larger values provide diminishing returns for NER.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Just like word embeddings represent words as numbers, character
    /// embeddings represent individual letters as numbers. This dimension controls how many
    /// numbers represent each character. The default of 30 works well; you rarely need to change this.
    /// </para>
    /// </remarks>
    public int CharEmbeddingDimension { get; set; } = 30;

    /// <summary>
    /// Gets or sets the hidden dimension of the character-level LSTM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This controls the capacity of the small LSTM that processes each word's characters.
    /// The character LSTM output (size = 2 * CharHiddenDimension for bidirectional) is
    /// concatenated with the word embedding to form the input to the main BiLSTM.
    ///
    /// The original paper uses 25 hidden units per direction (50 total after concatenation).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how much information the character-level model can
    /// extract from each word's spelling. The default of 50 is a good balance. Increasing this
    /// won't help much because character patterns are relatively simple.
    /// </para>
    /// </remarks>
    public int CharHiddenDimension { get; set; } = 50;

    /// <summary>
    /// Gets or sets the BIO label names for the tagging scheme.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This array maps label indices to human-readable label names. The length must equal
    /// <see cref="NumLabels"/>. The default set follows the CoNLL-2003 shared task.
    ///
    /// To customize for your domain, provide an array with your label names:
    /// <code>
    /// options.LabelNames = new[] { "O", "B-DRUG", "I-DRUG", "B-DISEASE", "I-DISEASE" };
    /// options.NumLabels = 5;
    /// </code>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These are the names of the entity types your model will recognize.
    /// The default set handles person names, organizations, locations, and miscellaneous entities.
    /// If you want to recognize different types (e.g., drug names and diseases for medical text),
    /// you can customize this list.
    /// </para>
    /// </remarks>
    public string[] LabelNames { get; set; } =
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
    /// ONNX models can be exported from PyTorch, TensorFlow, or other frameworks and provide
    /// optimized inference through the ONNX Runtime.
    ///
    /// Leave null for native training mode where the model is trained from scratch.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If you have a pre-trained model file (ending in .onnx), set this
    /// path to load it. The model will be ready for predictions immediately without training.
    /// If you want to train a new model from scratch, leave this as null.
    /// </para>
    /// </remarks>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX Runtime configuration options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls ONNX Runtime settings like execution providers (CPU, CUDA, DirectML),
    /// thread counts, memory allocation, and optimization levels. The defaults use CPU
    /// execution which works on all platforms.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These settings control how the ONNX model runs. The defaults
    /// work well for most cases. If you have an NVIDIA GPU, you can configure CUDA execution
    /// for faster inference.
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
    /// The learning rate controls how much the model's weights are adjusted during each
    /// training step. The original BiLSTM-CRF paper uses SGD with a learning rate of 0.01
    /// and gradient clipping. Modern implementations typically use Adam/AdamW with a lower
    /// learning rate.
    ///
    /// Recommended values:
    /// - <b>1e-3:</b> Default for AdamW optimizer (good starting point)
    /// - <b>1e-2:</b> For SGD optimizer with momentum
    /// - <b>5e-5:</b> For fine-tuning BERT+BiLSTM-CRF models
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate is like a step size when the model is learning.
    /// Too large and it overshoots the best answer; too small and it takes forever to learn.
    /// The default of 0.001 works well with the AdamW optimizer. If training is unstable
    /// (loss jumping up and down), try halving it.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the dropout rate for regularization during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Dropout randomly sets a fraction of neuron activations to zero during training,
    /// which prevents the model from relying too heavily on any single feature. This is
    /// the primary regularization technique for BiLSTM-CRF models.
    ///
    /// The original paper (Lample et al., 2016) uses 0.5 dropout, applied:
    /// - Between stacked LSTM layers (if more than 1 layer)
    /// - Before the linear projection layer
    /// - On word embeddings
    ///
    /// Higher dropout rates provide stronger regularization but may underfit on large datasets.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Dropout prevents the model from "memorizing" the training data
    /// instead of learning general patterns. A rate of 0.5 means half the neurons are randomly
    /// turned off during each training step. This forces the model to learn robust patterns
    /// that work even when some information is missing. The default of 0.5 follows the original
    /// paper and works well for most NER tasks.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.5;

    #endregion
}
