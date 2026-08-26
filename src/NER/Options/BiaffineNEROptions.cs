namespace AiDotNet.NER.Options;

/// <summary>
/// Options for Biaffine-NER, defaulting to the values published in
/// Yu et al., ACL 2020, "Named Entity Recognition as Dependency Parsing" (arXiv:2005.07150).
/// </summary>
/// <remarks>
/// <para><see cref="SpanBasedNEROptions"/> is shared by Biaffine-NER, SpERT and PURE, which are
/// three different papers with genuinely conflicting hyperparameters — SpERT samples negative
/// spans, whereas Biaffine-NER classifies every span with a dedicated non-entity class. A single
/// set of shared defaults therefore cannot be faithful to all three, so each model carries its
/// own subclass whose defaults are literally its own paper's.</para>
/// <para>Values taken from the paper's hyperparameter table: BiLSTM size 200 with 3 layers and
/// dropout 0.4; FFNN size 150 with dropout 0.2; embeddings dropout 0.5; Adam at learning rate
/// 1e-3; a BERT-Large encoder using the last 4 layers (1024-dimensional).</para>
/// <para><b>For Beginners:</b> These are the settings the paper's authors actually used to get
/// their published results. You can change any of them, but the defaults are the paper.</para>
/// </remarks>
public class BiaffineNEROptions : SpanBasedNEROptions
{
    /// <summary>Initializes a new instance with the paper's published defaults.</summary>
    public BiaffineNEROptions()
    {
        // BERT-Large, last 4 layers concatenated down to a 1024-wide contextual representation.
        HiddenDimension = 1024;
        NumAttentionHeads = 16;
        NumTransformerLayers = 24;
        IntermediateDimension = 4096;

        // "FFNN size 150" with "0.2" dropout — the two span-boundary FFNNs feeding the biaffine
        // scorer. The shared base defaults to a 256-wide span embedding, which is not a value
        // from this paper.
        SpanEmbeddingDimension = 150;
        DropoutRate = 0.2;

        // "Adam" at "1e-3". The shared base defaults to 5e-5, a BERT fine-tuning rate that this
        // paper does not use.
        LearningRate = 1e-3;

        // Biaffine-NER enumerates ALL spans with start <= end and classifies each one, using a
        // non-entity category rather than sampling negatives. The base class defaults to SpERT's
        // sampling recipe, which is not this paper's.
        UseNegativeSampling = false;
    }

    /// <summary>Initializes a new instance by copying shared span options.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public BiaffineNEROptions(SpanBasedNEROptions other)
        : base(other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));

        if (other is BiaffineNEROptions biaffine)
        {
            BiLstmHiddenSize = biaffine.BiLstmHiddenSize;
            BiLstmLayers = biaffine.BiLstmLayers;
            BiLstmDropout = biaffine.BiLstmDropout;
            EmbeddingsDropout = biaffine.EmbeddingsDropout;
        }
    }

    /// <summary>Initializes a new instance by copying another Biaffine-NER options instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    public BiaffineNEROptions(BiaffineNEROptions other)
        : this((SpanBasedNEROptions)other)
    {
    }

    /// <summary>
    /// Gets or sets the per-direction hidden size of the BiLSTM stacked on the encoder.
    /// </summary>
    /// <value>Defaults to 200, the paper's "BiLSTM size".</value>
    /// <remarks>
    /// <b>For Beginners:</b> After the transformer produces one vector per word, a bidirectional
    /// LSTM re-reads the sentence forwards and backwards. This is how wide that re-reading pass
    /// is; larger values give it more capacity at proportionally more cost.
    /// </remarks>
    public int BiLstmHiddenSize { get; set; } = 200;

    /// <summary>
    /// Gets or sets the number of stacked BiLSTM layers.
    /// </summary>
    /// <value>Defaults to 3, the paper's value.</value>
    public int BiLstmLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the dropout applied inside the BiLSTM stack.
    /// </summary>
    /// <value>Defaults to 0.4, the paper's value.</value>
    public double BiLstmDropout { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets the dropout applied to the input embeddings.
    /// </summary>
    /// <value>Defaults to 0.5, the paper's "embeddings dropout".</value>
    public double EmbeddingsDropout { get; set; } = 0.5;
}
