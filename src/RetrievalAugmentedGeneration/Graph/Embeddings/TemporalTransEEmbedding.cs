using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Temporal TransE embedding: extends TransE with time-aware scoring via discretized time bins.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// TE-TransE models temporal knowledge graphs by adding a time embedding to the scoring function:
/// d(h, r, t, τ) = ||h + r + t_time(τ) - t||, where t_time(τ) is a learned time bin embedding.
/// Time is discretized into bins (e.g., yearly), and each bin learns its own embedding vector.
/// </para>
/// <para><b>For Beginners:</b> Regular TransE doesn't know about time — it treats all facts as eternal.
/// TE-TransE adds a time dimension:
/// - Facts like "Obama PRESIDENT_OF USA" have a time window (2009-2017)
/// - The model learns that "Obama + president_of + time_2012" ≈ "USA"
/// - But "Obama + president_of + time_2020" should NOT point to "USA"
///
/// Time is grouped into bins (e.g., one per year). Each bin has its own learned vector.
/// </para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class TemporalTransEEmbedding<T> : KGEmbeddingBase<T>
{
    /// <inheritdoc />
    public override bool IsDistanceBased => true;

    private T[][] _timeBinEmbeddings = [];
    private DateTime _minTime;
    private TimeSpan _binWidth;
    private int _numTimeBins;

    // Maps (headIdx, relIdx, tailIdx) -> time bin for training
    private Dictionary<(int, int, int), int> _tripleTimeBins = [];

    /// <summary>
    /// Gets or sets the number of time bins for discretization. Default: 100.
    /// </summary>
    public int? NumTimeBins { get; set; }

    /// <summary>
    /// Gets the actual number of time bins after training (resolved from <see cref="NumTimeBins"/> or options default).
    /// </summary>
    public int ResolvedNumTimeBins => _numTimeBins;

    private protected override void OnInitialize(KGEmbeddingOptions options, Random rng, KnowledgeGraph<T> graph)
    {
        // Determine time range from edges
        var temporalEdges = graph.GetAllEdges()
            .Where(e => e.ValidFrom.HasValue || e.ValidUntil.HasValue)
            .ToList();

        if (temporalEdges.Count > 0)
        {
            var minTimes = temporalEdges.Where(e => e.ValidFrom.HasValue).Select(e => e.ValidFrom.GetValueOrDefault());
            var maxTimes = temporalEdges.Where(e => e.ValidUntil.HasValue).Select(e => e.ValidUntil.GetValueOrDefault());

            _minTime = minTimes.Any() ? minTimes.Min() : DateTime.MinValue;
            var maxTime = maxTimes.Any() ? maxTimes.Max() : DateTime.UtcNow;

            _numTimeBins = NumTimeBins ?? options.GetEffectiveNumTimeBins();
            if (_numTimeBins <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumTimeBins), "NumTimeBins must be > 0.");
            var totalSpan = maxTime - _minTime;
            _binWidth = totalSpan.TotalSeconds > 0
                ? TimeSpan.FromSeconds(totalSpan.TotalSeconds / _numTimeBins)
                : TimeSpan.FromDays(365);
        }
        else
        {
            _minTime = DateTime.MinValue;
            _numTimeBins = NumTimeBins ?? options.GetEffectiveNumTimeBins();
            if (_numTimeBins <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumTimeBins), "NumTimeBins must be > 0.");
            _binWidth = TimeSpan.FromDays(365);
        }

        int dim = options.GetEffectiveEmbeddingDimension();
        double scale = 6.0 / Math.Sqrt(dim);

        _timeBinEmbeddings = new T[_numTimeBins][];
        for (int i = 0; i < _numTimeBins; i++)
        {
            _timeBinEmbeddings[i] = new T[dim];
            for (int d = 0; d < dim; d++)
            {
                _timeBinEmbeddings[i][d] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * scale);
            }
        }

        // Relations are normalized once at initialization, as in TransE (Bordes et al. 2013, Algorithm 1).
        foreach (var emb in _relationEmbeddings)
        {
            NormalizeL2(emb);
        }

        // Build triple -> time bin mapping from edge temporal data
        _tripleTimeBins = [];
        foreach (var edge in graph.GetAllEdges())
        {
            if (!_entityIndex.TryGetValue(edge.SourceId, out var hIdx) ||
                !_relationIndex.TryGetValue(edge.RelationType, out var rIdx) ||
                !_entityIndex.TryGetValue(edge.TargetId, out var tIdx))
                continue;

            // Use midpoint of validity window, or ValidFrom, or ValidUntil
            DateTime? timestamp = null;
            if (edge.ValidFrom.HasValue && edge.ValidUntil.HasValue)
            {
                var start = edge.ValidFrom.Value;
                var end = edge.ValidUntil.Value;

                // Skip edges with invalid temporal windows (ValidUntil < ValidFrom)
                // which can arise if callers set the properties directly.
                if (end < start)
                    continue;

                // Overflow-safe midpoint: delta is non-negative for valid ranges,
                // and start + delta/2 stays within DateTime bounds.
                var startTicks = start.Ticks;
                var deltaTicks = end.Ticks - startTicks;
                var midTicks = startTicks + deltaTicks / 2;
                timestamp = new DateTime(midTicks, DateTimeKind.Utc);
            }
            else if (edge.ValidFrom.HasValue)
            {
                timestamp = edge.ValidFrom.Value;
            }
            else if (edge.ValidUntil.HasValue)
            {
                timestamp = edge.ValidUntil.Value;
            }

            if (timestamp.HasValue)
            {
                _tripleTimeBins[(hIdx, rIdx, tIdx)] = GetTimeBin(timestamp.Value);
            }
        }
    }

    private protected override T ScoreTripleInternal(int headIdx, int relationIdx, int tailIdx)
    {
        // Use the stored time bin if available, otherwise bin 0
        int timeBin = _tripleTimeBins.TryGetValue((headIdx, relationIdx, tailIdx), out var bin) ? bin : 0;
        return ScoreTripleWithTime(headIdx, relationIdx, tailIdx, timeBin);
    }

    /// <summary>
    /// Scores a triple at a specific time point.
    /// </summary>
    /// <param name="headId">Head entity ID.</param>
    /// <param name="relationType">Relation type.</param>
    /// <param name="tailId">Tail entity ID.</param>
    /// <param name="timestamp">The time point.</param>
    /// <returns>Score (distance) — lower means more plausible.</returns>
    public T ScoreTripleAtTime(string headId, string relationType, string tailId, DateTime timestamp)
    {
        if (!IsTrained)
            throw new InvalidOperationException("Model must be trained before scoring triples.");

        if (!_entityIndex.TryGetValue(headId, out var h) ||
            !_relationIndex.TryGetValue(relationType, out var r) ||
            !_entityIndex.TryGetValue(tailId, out var t))
        {
            return NumOps.MaxValue;
        }

        int timeBin = GetTimeBin(timestamp);
        return ScoreTripleWithTime(h, r, t, timeBin);
    }

    private T ScoreTripleWithTime(int headIdx, int relationIdx, int tailIdx, int timeBin)
    {
        // ||h + r + t_time - t||
        return Engine.Norm(Residual(headIdx, relationIdx, tailIdx, timeBin));
    }

    /// <summary>h + r + t_time - t.</summary>
    private Vector<T> Residual(int headIdx, int relationIdx, int tailIdx, int timeBin)
        => Engine.Subtract(
            Engine.Add(
                Engine.Add(Row(_entityEmbeddings[headIdx]), Row(_relationEmbeddings[relationIdx])),
                Row(_timeBinEmbeddings[timeBin])),
            Row(_entityEmbeddings[tailIdx]));

    /// <summary>
    /// Entities are renormalized at the start of every mini-batch, as in TransE (Bordes et al. 2013, Algorithm 1).
    /// </summary>
    private protected override void OnBatchStart()
    {
        foreach (var emb in _entityEmbeddings)
        {
            NormalizeL2(emb);
        }
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double margin = options.GetEffectiveMargin();

        // Look up the time bin for this positive triple from edge temporal data
        int timeBin = _tripleTimeBins.TryGetValue((posHead, relation, posTail), out var bin) ? bin : 0;

        // TransE margin loss with the L2 norm (Bordes et al. 2013), the distance the model scores with.
        var posDiff = Residual(posHead, relation, posTail, timeBin);
        var negDiff = Residual(negHead, relation, negTail, timeBin);
        T posDist = Engine.Norm(posDiff);
        T negDist = Engine.Norm(negDiff);

        double loss = Math.Max(0.0, margin + NumOps.ToDouble(posDist) - NumOps.ToDouble(negDist));
        if (loss <= 0.0) return 0.0;

        // d||x||/dx = x / ||x||; every step is formed from the pre-update embeddings.
        T lr = NumOps.FromDouble(learningRate);
        var posStep = Engine.Multiply(UnitDirection(posDiff, posDist), lr);
        var negStep = Engine.Multiply(UnitDirection(negDiff, negDist), lr);

        // Relation and time embedding appear in both distances.
        var sharedStep = Engine.Subtract(posStep, negStep);
        SubtractFrom(_entityEmbeddings[posHead], posStep);
        AddTo(_entityEmbeddings[posTail], posStep);
        SubtractFrom(_relationEmbeddings[relation], sharedStep);
        SubtractFrom(_timeBinEmbeddings[timeBin], sharedStep);
        AddTo(_entityEmbeddings[negHead], negStep);
        SubtractFrom(_entityEmbeddings[negTail], negStep);

        return loss;
    }

    private protected override void OnPostEpoch(int epoch)
    {
        // Normalize entity embeddings to unit ball (TransE constraint)
        foreach (var emb in _entityEmbeddings)
        {
            NormalizeL2(emb);
        }
    }

    private int GetTimeBin(DateTime timestamp)
    {
        if (_binWidth.TotalSeconds <= 0) return 0;
        int bin = (int)((timestamp - _minTime).TotalSeconds / _binWidth.TotalSeconds);
        return Math.Max(0, Math.Min(bin, _numTimeBins - 1));
    }
}
