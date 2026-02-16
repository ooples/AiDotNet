namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Configuration options for training knowledge graph embedding models.
/// </summary>
/// <remarks>
/// <para>
/// These options control the training process for all KG embedding models (TransE, RotatE, ComplEx, DistMult).
/// All values use nullable types with industry-standard defaults applied internally when null.
/// </para>
/// <para><b>For Beginners:</b> These settings control how the model learns. The defaults work well
/// for most knowledge graphs, but you can tune them for better performance:
/// - EmbeddingDimension: Larger = more expressive but slower to train
/// - Epochs: More = better fit but risk of overfitting
/// - LearningRate: How fast the model updates (too high = unstable, too low = slow)
/// - NegativeSamples: Corrupted triples generated per real triple for contrast
/// </para>
/// </remarks>
public class KGEmbeddingOptions
{
    /// <summary>
    /// Dimensionality of entity and relation embedding vectors. Default: 100.
    /// </summary>
    public int? EmbeddingDimension { get; set; }

    /// <summary>
    /// Number of training epochs (full passes over all triples). Default: 100.
    /// </summary>
    public int? Epochs { get; set; }

    /// <summary>
    /// Mini-batch size for stochastic gradient descent. Default: 128.
    /// </summary>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Learning rate for SGD updates. Default: 0.01.
    /// </summary>
    public double? LearningRate { get; set; }

    /// <summary>
    /// Margin for margin-based ranking loss (used by TransE and RotatE). Default: 1.0.
    /// </summary>
    public double? Margin { get; set; }

    /// <summary>
    /// Number of negative (corrupted) samples per positive triple. Default: 1.
    /// </summary>
    public int? NegativeSamples { get; set; }

    /// <summary>
    /// L2 regularization coefficient. Default: 0.0 (no regularization).
    /// </summary>
    public double? L2Regularization { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null (non-deterministic).
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Number of time bins for temporal embedding models (TemporalTransE). Default: 100.
    /// Time is discretized into this many bins, each with its own learned embedding vector.
    /// Only used when the embedding model supports temporal data.
    /// </summary>
    public int? NumTimeBins { get; set; }

    internal int GetEffectiveEmbeddingDimension() => EmbeddingDimension ?? 100;
    internal int GetEffectiveEpochs() => Epochs ?? 100;
    internal int GetEffectiveBatchSize() => BatchSize ?? 128;
    internal double GetEffectiveLearningRate() => LearningRate ?? 0.01;
    internal double GetEffectiveMargin() => Margin ?? 1.0;
    internal int GetEffectiveNegativeSamples() => NegativeSamples ?? 1;
    internal double GetEffectiveL2Regularization() => L2Regularization ?? 0.0;
    internal int GetEffectiveNumTimeBins() => NumTimeBins ?? 100;
}
