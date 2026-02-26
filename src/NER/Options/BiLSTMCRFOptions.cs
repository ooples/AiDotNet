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
/// with a Conditional Random Field layer for sequence labeling:
/// - BiLSTM processes tokens in both forward and backward directions to capture full context
/// - CRF layer models label transition dependencies for globally optimal label sequences
/// - Character-level embeddings capture morphological features (capitalization, prefixes, suffixes)
/// - Dropout regularization prevents overfitting
/// </para>
/// <para>
/// <b>For Beginners:</b> BiLSTM-CRF reads text forward and backward to understand context,
/// then uses a CRF to pick the best sequence of labels. It's the most widely-used neural
/// NER architecture and serves as the foundation for many modern NER systems.
/// </para>
/// </remarks>
public class BiLSTMCRFOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public BiLSTMCRFOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
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

    /// <summary>Gets or sets the model variant.</summary>
    public NERModelVariant Variant { get; set; } = NERModelVariant.Base;

    /// <summary>Gets or sets the input token embedding dimension.</summary>
    /// <remarks>Common values: 100 (GloVe-100d), 300 (GloVe-300d), 768 (BERT).</remarks>
    public int EmbeddingDimension { get; set; } = 100;

    /// <summary>Gets or sets the LSTM hidden dimension per direction.</summary>
    /// <remarks>The total BiLSTM output dimension is 2 * HiddenDimension.</remarks>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>Gets or sets the number of stacked BiLSTM layers.</summary>
    public int NumLSTMLayers { get; set; } = 1;

    /// <summary>Gets or sets the number of entity label classes.</summary>
    /// <remarks>Default 9 for CoNLL-2003 BIO: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC.</remarks>
    public int NumLabels { get; set; } = 9;

    /// <summary>Gets or sets the maximum sequence length.</summary>
    public int MaxSequenceLength { get; set; } = 256;

    /// <summary>Gets or sets whether to use CRF decoding.</summary>
    public bool UseCRF { get; set; } = true;

    /// <summary>Gets or sets whether to use character-level embeddings.</summary>
    /// <remarks>Character embeddings capture morphological features like capitalization and prefixes.</remarks>
    public bool UseCharEmbeddings { get; set; } = true;

    /// <summary>Gets or sets the character embedding dimension.</summary>
    public int CharEmbeddingDimension { get; set; } = 30;

    /// <summary>Gets or sets the character-level LSTM hidden dimension.</summary>
    public int CharHiddenDimension { get; set; } = 50;

    /// <summary>Gets or sets the label names for the BIO scheme.</summary>
    public string[] LabelNames { get; set; } =
    [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ];

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.5;

    #endregion
}
