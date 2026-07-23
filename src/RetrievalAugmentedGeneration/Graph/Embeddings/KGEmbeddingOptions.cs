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

    internal int GetEffectiveEmbeddingDimension()
    {
        var value = EmbeddingDimension ?? 100;
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(EmbeddingDimension), "EmbeddingDimension must be > 0.");
        return value;
    }

    internal int GetEffectiveEpochs()
    {
        var value = Epochs ?? 100;
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(Epochs), "Epochs must be > 0.");
        return value;
    }

    internal int GetEffectiveBatchSize()
    {
        var value = BatchSize ?? 128;
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(BatchSize), "BatchSize must be > 0.");
        return value;
    }

    internal double GetEffectiveLearningRate()
    {
        var value = LearningRate ?? 0.01;
        if (value <= 0 || double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(nameof(LearningRate), "LearningRate must be a finite positive number.");
        return value;
    }

    internal double GetEffectiveMargin()
    {
        var value = Margin ?? 1.0;
        if (value <= 0 || double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(nameof(Margin), "Margin must be a finite positive number.");
        return value;
    }

    internal int GetEffectiveNegativeSamples()
    {
        var value = NegativeSamples ?? 1;
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(NegativeSamples), "NegativeSamples must be > 0.");
        return value;
    }

    internal double GetEffectiveL2Regularization()
    {
        var value = L2Regularization ?? 0.0;
        if (value < 0 || double.IsNaN(value) || double.IsInfinity(value))
            throw new ArgumentOutOfRangeException(nameof(L2Regularization), "L2Regularization must be a finite non-negative number.");
        return value;
    }

    internal int? GetEffectiveSeed() => Seed;

    internal int GetEffectiveNumTimeBins()
    {
        var value = NumTimeBins ?? 100;
        if (value <= 0) throw new ArgumentOutOfRangeException(nameof(NumTimeBins), "NumTimeBins must be > 0.");
        return value;
    }
}
