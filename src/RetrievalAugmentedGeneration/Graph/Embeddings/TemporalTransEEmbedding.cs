using System;
using System.Collections.Generic;
using System.Linq;

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
            var totalSpan = maxTime - _minTime;
            _binWidth = totalSpan.TotalSeconds > 0
                ? TimeSpan.FromSeconds(totalSpan.TotalSeconds / _numTimeBins)
                : TimeSpan.FromDays(365);
        }
        else
        {
            _minTime = DateTime.MinValue;
            _numTimeBins = NumTimeBins ?? options.GetEffectiveNumTimeBins();
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
                var midTicks = edge.ValidFrom.Value.Ticks +
                    (edge.ValidUntil.Value.Ticks - edge.ValidFrom.Value.Ticks) / 2;
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
            return NumOps.FromDouble(double.MaxValue);
        }

        int timeBin = GetTimeBin(timestamp);
        return ScoreTripleWithTime(h, r, t, timeBin);
    }

    private T ScoreTripleWithTime(int headIdx, int relationIdx, int tailIdx, int timeBin)
    {
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        var timeEmb = _timeBinEmbeddings[timeBin];
        int dim = EmbeddingDimension;

        // ||h + r + t_time - t||
        T sumSq = NumOps.Zero;
        for (int d = 0; d < dim; d++)
        {
            T diff = NumOps.Subtract(NumOps.Add(NumOps.Add(h[d], r[d]), timeEmb[d]), t[d]);
            sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sumSq);
    }

    private protected override double ComputeLossAndUpdateGradients(
        int posHead, int relation, int posTail,
        int negHead, int negTail,
        double learningRate, KGEmbeddingOptions options)
    {
        double margin = options.GetEffectiveMargin();
        int dim = EmbeddingDimension;

        // Look up the time bin for this positive triple from edge temporal data
        int timeBin = _tripleTimeBins.TryGetValue((posHead, relation, posTail), out var bin) ? bin : 0;

        double posDist = ComputeDistanceSqWithTime(posHead, relation, posTail, timeBin, dim);
        double negDist = ComputeDistanceSqWithTime(negHead, relation, negTail, timeBin, dim);

        double loss = Math.Max(0.0, margin + posDist - negDist);
        if (loss <= 0.0) return 0.0;

        T lr = NumOps.FromDouble(learningRate);
        T two = NumOps.FromDouble(2.0);
        var timeEmb = _timeBinEmbeddings[timeBin];

        var ph = _entityEmbeddings[posHead];
        var r_ = _relationEmbeddings[relation];
        var pt = _entityEmbeddings[posTail];
        var nh_ = _entityEmbeddings[negHead];
        var nt_ = _entityEmbeddings[negTail];

        for (int d = 0; d < dim; d++)
        {
            // Positive gradient: 2(h + r + time - t)
            T posGrad = NumOps.Multiply(two,
                NumOps.Subtract(NumOps.Add(NumOps.Add(ph[d], r_[d]), timeEmb[d]), pt[d]));

            // Negative gradient: 2(nh + r + time - nt)
            T negGrad = NumOps.Multiply(two,
                NumOps.Subtract(NumOps.Add(NumOps.Add(nh_[d], r_[d]), timeEmb[d]), nt_[d]));

            // Update positive triple (decrease distance)
            ph[d] = NumOps.Subtract(ph[d], NumOps.Multiply(lr, posGrad));
            pt[d] = NumOps.Add(pt[d], NumOps.Multiply(lr, posGrad));

            // Relation gradient accounts for both positive and negative triples
            T rGrad = NumOps.Subtract(posGrad, negGrad);
            r_[d] = NumOps.Subtract(r_[d], NumOps.Multiply(lr, rGrad));

            // Time embedding gradient also accounts for both
            timeEmb[d] = NumOps.Subtract(timeEmb[d], NumOps.Multiply(lr, rGrad));

            // Update negative triple (increase distance)
            nh_[d] = NumOps.Add(nh_[d], NumOps.Multiply(lr, negGrad));
            nt_[d] = NumOps.Subtract(nt_[d], NumOps.Multiply(lr, negGrad));
        }

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

    private double ComputeDistanceSqWithTime(int headIdx, int relationIdx, int tailIdx, int timeBin, int dim)
    {
        var h = _entityEmbeddings[headIdx];
        var r = _relationEmbeddings[relationIdx];
        var t = _entityEmbeddings[tailIdx];
        var timeEmb = _timeBinEmbeddings[timeBin];

        double sumSq = 0.0;
        for (int d = 0; d < dim; d++)
        {
            double diff = NumOps.ToDouble(h[d]) + NumOps.ToDouble(r[d]) + NumOps.ToDouble(timeEmb[d]) - NumOps.ToDouble(t[d]);
            sumSq += diff * diff;
        }

        return sumSq;
    }

    private int GetTimeBin(DateTime timestamp)
    {
        if (_binWidth.TotalSeconds <= 0) return 0;
        int bin = (int)((timestamp - _minTime).TotalSeconds / _binWidth.TotalSeconds);
        return Math.Max(0, Math.Min(bin, _numTimeBins - 1));
    }
}
