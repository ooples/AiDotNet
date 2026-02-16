using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Abstract base class for knowledge graph embedding models providing shared training infrastructure.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// This base class implements the common training loop with mini-batch SGD and negative sampling.
/// Subclasses only need to implement scoring, gradient computation, and post-epoch normalization.
/// </para>
/// <para><b>For Beginners:</b> This class handles the "plumbing" of training:
/// 1. Builds entity/relation vocabularies from the graph
/// 2. Initializes random embedding vectors
/// 3. For each epoch, shuffles triples into mini-batches
/// 4. For each positive triple, generates corrupted (negative) triples
/// 5. Computes loss and gradients, then updates embeddings via SGD
/// 6. Subclasses define how to score triples and compute gradients
/// </para>
/// </remarks>
public abstract class KGEmbeddingBase<T> : IKnowledgeGraphEmbedding<T>
{
    private protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private protected T[][] _entityEmbeddings = [];
    private protected T[][] _relationEmbeddings = [];
    private protected Dictionary<string, int> _entityIndex = [];
    private protected Dictionary<string, int> _relationIndex = [];
    private protected string[] _entityIds = [];

    private int _embeddingDimension;

    /// <inheritdoc />
    public bool IsTrained { get; private protected set; }

    /// <inheritdoc />
    public abstract bool IsDistanceBased { get; }

    /// <inheritdoc />
    public int EmbeddingDimension => _embeddingDimension;

    /// <inheritdoc />
    public KGEmbeddingTrainingResult Train(KnowledgeGraph<T> graph, KGEmbeddingOptions? options = null)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        var opts = options ?? new KGEmbeddingOptions();
        _embeddingDimension = opts.GetEffectiveEmbeddingDimension();

        var stopwatch = Stopwatch.StartNew();

        // Build vocabularies
        var triples = BuildVocabularyAndTriples(graph);
        if (triples.Count == 0)
            throw new InvalidOperationException("Knowledge graph contains no edges to train on.");

        var rng = opts.Seed.HasValue ? new Random(opts.Seed.Value) : new Random();

        // Initialize embeddings
        InitializeEmbeddings(rng);
        OnInitialize(opts, rng, graph);

        // Training loop
        var epochs = opts.GetEffectiveEpochs();
        var batchSize = opts.GetEffectiveBatchSize();
        var lr = opts.GetEffectiveLearningRate();
        var negSamples = opts.GetEffectiveNegativeSamples();
        var l2Reg = opts.GetEffectiveL2Regularization();
        var epochLosses = new List<double>(epochs);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(triples, rng);
            double epochLoss = 0.0;
            int batchCount = 0;

            for (int batchStart = 0; batchStart < triples.Count; batchStart += batchSize)
            {
                int batchEnd = Math.Min(batchStart + batchSize, triples.Count);
                double batchLoss = 0.0;

                for (int i = batchStart; i < batchEnd; i++)
                {
                    var (h, r, t) = triples[i];

                    for (int n = 0; n < negSamples; n++)
                    {
                        // Corrupt head or tail uniformly
                        int nh, nt;
                        if (rng.NextDouble() < 0.5)
                        {
                            nh = rng.Next(_entityEmbeddings.Length);
                            nt = t;
                        }
                        else
                        {
                            nh = h;
                            nt = rng.Next(_entityEmbeddings.Length);
                        }

                        batchLoss += ComputeLossAndUpdateGradients(h, r, t, nh, nt, lr, opts);
                    }
                }

                // Apply L2 regularization if configured
                if (l2Reg > 0.0)
                {
                    ApplyL2Regularization(lr, l2Reg);
                }

                batchCount++;
                epochLoss += batchLoss;
            }

            OnPostEpoch(epoch);
            epochLosses.Add(batchCount > 0 ? epochLoss / (triples.Count * negSamples) : 0.0);
        }

        IsTrained = true;
        stopwatch.Stop();

        return new KGEmbeddingTrainingResult
        {
            EpochLosses = epochLosses,
            TotalEpochs = epochs,
            EntityCount = _entityEmbeddings.Length,
            RelationCount = _relationEmbeddings.Length,
            TripleCount = triples.Count,
            TrainingDuration = stopwatch.Elapsed
        };
    }

    /// <inheritdoc />
    public T[]? GetEntityEmbedding(string entityId)
    {
        if (!IsTrained || !_entityIndex.TryGetValue(entityId, out var idx))
            return null;
        int size = GetEntityEmbeddingSize();
        var copy = new T[size];
        Array.Copy(_entityEmbeddings[idx], copy, size);
        return copy;
    }

    /// <inheritdoc />
    public T[]? GetRelationEmbedding(string relationType)
    {
        if (!IsTrained || !_relationIndex.TryGetValue(relationType, out var idx))
            return null;
        var copy = new T[GetRelationEmbeddingSize()];
        Array.Copy(_relationEmbeddings[idx], copy, GetRelationEmbeddingSize());
        return copy;
    }

    /// <inheritdoc />
    public T ScoreTriple(string headId, string relationType, string tailId)
    {
        if (!IsTrained)
            throw new InvalidOperationException("Model must be trained before scoring triples.");

        if (!_entityIndex.TryGetValue(headId, out var h) ||
            !_relationIndex.TryGetValue(relationType, out var r) ||
            !_entityIndex.TryGetValue(tailId, out var t))
        {
            // Distance-based: higher = worse, so return MaxValue
            // Semantic: lower = worse, so return MinValue
            return NumOps.FromDouble(IsDistanceBased ? double.MaxValue : double.MinValue);
        }

        return ScoreTripleInternal(h, r, t);
    }

    /// <summary>
    /// Gets the size of entity embedding vectors. Override for models that use different sizes (e.g., complex-valued with real+imaginary parts).
    /// </summary>
    private protected virtual int GetEntityEmbeddingSize() => _embeddingDimension;

    /// <summary>
    /// Gets the size of relation embedding vectors. Override for models that use different sizes (e.g., complex-valued).
    /// </summary>
    private protected virtual int GetRelationEmbeddingSize() => _embeddingDimension;

    /// <summary>
    /// Called after embedding arrays are allocated but before the training loop begins.
    /// Subclasses can initialize additional arrays or perform model-specific setup.
    /// The graph is provided so models can extract metadata (e.g., temporal ranges).
    /// </summary>
    private protected virtual void OnInitialize(KGEmbeddingOptions options, Random rng, KnowledgeGraph<T> graph) { }

    /// <summary>
    /// Called after each epoch. Subclasses can perform normalization (e.g., TransE normalizes entities to unit ball).
    /// </summary>
    private protected virtual void OnPostEpoch(int epoch) { }

    /// <summary>
    /// Computes the score for a triple given entity/relation indices.
    /// </summary>
    private protected abstract T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx);

    /// <summary>
    /// Computes loss for a positive triple vs. a negative triple and applies gradient updates in-place.
    /// Returns the loss contribution.
    /// </summary>
    private protected abstract double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options);

    // --- Shared helpers ---

    private List<(int head, int relation, int tail)> BuildVocabularyAndTriples(KnowledgeGraph<T> graph)
    {
        _entityIndex = [];
        _relationIndex = [];
        var triples = new List<(int, int, int)>();

        foreach (var node in graph.GetAllNodes())
        {
            if (!_entityIndex.ContainsKey(node.Id))
            {
                _entityIndex[node.Id] = _entityIndex.Count;
            }
        }

        foreach (var edge in graph.GetAllEdges())
        {
            if (!_entityIndex.ContainsKey(edge.SourceId))
                _entityIndex[edge.SourceId] = _entityIndex.Count;
            if (!_entityIndex.ContainsKey(edge.TargetId))
                _entityIndex[edge.TargetId] = _entityIndex.Count;
            if (!_relationIndex.ContainsKey(edge.RelationType))
                _relationIndex[edge.RelationType] = _relationIndex.Count;

            triples.Add((_entityIndex[edge.SourceId], _relationIndex[edge.RelationType], _entityIndex[edge.TargetId]));
        }

        _entityIds = new string[_entityIndex.Count];
        foreach (var kvp in _entityIndex)
            _entityIds[kvp.Value] = kvp.Key;

        return triples;
    }

    private void InitializeEmbeddings(Random rng)
    {
        int entityCount = _entityIndex.Count;
        int relationCount = _relationIndex.Count;
        int entDim = GetEntityEmbeddingSize();
        int relDim = GetRelationEmbeddingSize();

        double scale = 6.0 / Math.Sqrt(_embeddingDimension);

        _entityEmbeddings = new T[entityCount][];
        for (int i = 0; i < entityCount; i++)
        {
            _entityEmbeddings[i] = new T[entDim];
            for (int d = 0; d < entDim; d++)
            {
                _entityEmbeddings[i][d] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * scale);
            }
            NormalizeL2(_entityEmbeddings[i]);
        }

        _relationEmbeddings = new T[relationCount][];
        for (int i = 0; i < relationCount; i++)
        {
            _relationEmbeddings[i] = new T[relDim];
            for (int d = 0; d < relDim; d++)
            {
                _relationEmbeddings[i][d] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * scale);
            }
        }
    }

    private protected static void NormalizeL2(T[] vector)
    {
        T sumSq = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(vector[i], vector[i]));
        }

        double norm = NumOps.ToDouble(NumOps.Sqrt(sumSq));
        if (norm < 1e-12) return;

        T invNorm = NumOps.FromDouble(1.0 / norm);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = NumOps.Multiply(vector[i], invNorm);
        }
    }

    private void ApplyL2Regularization(double learningRate, double l2Reg)
    {
        double factor = 1.0 - learningRate * l2Reg;
        T scaleFactor = NumOps.FromDouble(factor);

        foreach (var emb in _entityEmbeddings)
        {
            for (int d = 0; d < emb.Length; d++)
                emb[d] = NumOps.Multiply(emb[d], scaleFactor);
        }

        foreach (var emb in _relationEmbeddings)
        {
            for (int d = 0; d < emb.Length; d++)
                emb[d] = NumOps.Multiply(emb[d], scaleFactor);
        }
    }

    private static void Shuffle<TItem>(List<TItem> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
