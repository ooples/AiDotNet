using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Link prediction engine that uses trained KG embeddings to predict missing triples
/// and evaluate model quality.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// Link prediction answers questions like "Given (Einstein, born_in, ?), what is the most likely tail entity?"
/// It works by scoring all candidate entities and ranking them by plausibility.
/// </para>
/// <para><b>For Beginners:</b> Link prediction fills in missing facts in a knowledge graph.
///
/// Example: Your graph has "Einstein born_in ?" — the predictor:
/// 1. Scores every entity as a possible answer: (Einstein, born_in, Germany) = 0.95, (Einstein, born_in, France) = 0.12
/// 2. Ranks them: Germany #1, Switzerland #2, Austria #3...
/// 3. Returns the top-K most plausible completions
///
/// Evaluation (EvaluateModel) tests how well the model can "guess" known facts by temporarily
/// hiding them and checking if they rank highly.
/// </para>
/// </remarks>
public class LinkPredictor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IKnowledgeGraphEmbedding<T> _embedding;

    /// <summary>
    /// Creates a new link predictor using a trained embedding model.
    /// </summary>
    /// <param name="embedding">A trained KG embedding model.</param>
    public LinkPredictor(IKnowledgeGraphEmbedding<T> embedding)
    {
        if (embedding == null) throw new ArgumentNullException(nameof(embedding));
        if (!embedding.IsTrained)
            throw new InvalidOperationException("Embedding model must be trained before link prediction.");
        _embedding = embedding;
    }

    /// <summary>
    /// Predicts the most plausible tail entities for a given (head, relation, ?) query.
    /// </summary>
    /// <param name="graph">The knowledge graph (used for entity candidates and filtering existing triples).</param>
    /// <param name="headId">The head entity ID.</param>
    /// <param name="relationType">The relation type.</param>
    /// <param name="topK">Number of top predictions to return.</param>
    /// <param name="filterExisting">Whether to exclude already-existing triples from results.</param>
    /// <returns>Top-K predicted triples sorted by plausibility.</returns>
    public List<PredictedTriple> PredictTails(
        KnowledgeGraph<T> graph, string headId, string relationType,
        int topK = 10, bool filterExisting = true)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        var existingTails = filterExisting
            ? new HashSet<string>(graph.GetOutgoingEdges(headId)
                .Where(e => e.RelationType == relationType)
                .Select(e => e.TargetId))
            : new HashSet<string>();

        var candidates = new List<(string tailId, double score)>();
        foreach (var node in graph.GetAllNodes())
        {
            if (node.Id == headId) continue;
            if (filterExisting && existingTails.Contains(node.Id)) continue;

            var score = _embedding.ScoreTriple(headId, relationType, node.Id);
            candidates.Add((node.Id, NumOps.ToDouble(score)));
        }

        // Distance-based: lower=better → sort ascending. Semantic: higher=better → sort descending.
        if (_embedding.IsDistanceBased)
            candidates.Sort((a, b) => a.score.CompareTo(b.score));
        else
            candidates.Sort((a, b) => b.score.CompareTo(a.score));

        return candidates.Take(topK).Select(c => new PredictedTriple
        {
            HeadId = headId,
            RelationType = relationType,
            TailId = c.tailId,
            Score = c.score,
            Confidence = ComputeConfidence(c.score, candidates)
        }).ToList();
    }

    /// <summary>
    /// Predicts the most plausible head entities for a given (?, relation, tail) query.
    /// </summary>
    /// <param name="graph">The knowledge graph.</param>
    /// <param name="relationType">The relation type.</param>
    /// <param name="tailId">The tail entity ID.</param>
    /// <param name="topK">Number of top predictions to return.</param>
    /// <param name="filterExisting">Whether to exclude already-existing triples.</param>
    /// <returns>Top-K predicted triples sorted by plausibility.</returns>
    public List<PredictedTriple> PredictHeads(
        KnowledgeGraph<T> graph, string relationType, string tailId,
        int topK = 10, bool filterExisting = true)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        var existingHeads = filterExisting
            ? new HashSet<string>(graph.GetIncomingEdges(tailId)
                .Where(e => e.RelationType == relationType)
                .Select(e => e.SourceId))
            : new HashSet<string>();

        var candidates = new List<(string headId, double score)>();
        foreach (var node in graph.GetAllNodes())
        {
            if (node.Id == tailId) continue;
            if (filterExisting && existingHeads.Contains(node.Id)) continue;

            var score = _embedding.ScoreTriple(node.Id, relationType, tailId);
            candidates.Add((node.Id, NumOps.ToDouble(score)));
        }

        if (_embedding.IsDistanceBased)
            candidates.Sort((a, b) => a.score.CompareTo(b.score));
        else
            candidates.Sort((a, b) => b.score.CompareTo(a.score));

        return candidates.Take(topK).Select(c => new PredictedTriple
        {
            HeadId = c.headId,
            RelationType = relationType,
            TailId = tailId,
            Score = c.score,
            Confidence = ComputeConfidence(c.score, candidates)
        }).ToList();
    }

    /// <summary>
    /// Evaluates the embedding model using standard link prediction metrics (MRR, Hits@K, MeanRank)
    /// in the filtered setting.
    /// </summary>
    /// <param name="graph">The full knowledge graph (used for filtering).</param>
    /// <param name="testTriples">Test triples to evaluate: (headId, relationType, tailId).</param>
    /// <param name="hitsAtKValues">K values for Hits@K metrics (default: 1, 3, 10).</param>
    /// <returns>Evaluation results with MRR, Hits@K, and MeanRank.</returns>
    public LinkPredictionEvaluation EvaluateModel(
        KnowledgeGraph<T> graph,
        IEnumerable<(string headId, string relationType, string tailId)> testTriples,
        int[]? hitsAtKValues = null)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));
        if (testTriples == null) throw new ArgumentNullException(nameof(testTriples));

        hitsAtKValues ??= [1, 3, 10];

        var allNodes = graph.GetAllNodes().ToList();
        var existingTriples = new HashSet<(string headId, string relationType, string tailId)>(
            graph.GetAllEdges().Select(e => (e.SourceId, e.RelationType, e.TargetId)));

        double sumReciprocalRank = 0.0;
        double sumRank = 0.0;
        var hitsCount = new Dictionary<int, int>();
        foreach (var k in hitsAtKValues)
            hitsCount[k] = 0;

        int tripleCount = 0;

        foreach (var (headId, relationType, tailId) in testTriples)
        {
            tripleCount++;

            // Tail prediction: rank all entities as possible tails
            int tailRank = ComputeFilteredRank(
                allNodes, headId, relationType, tailId,
                existingTriples, isTailPrediction: true);

            // Head prediction: rank all entities as possible heads
            int headRank = ComputeFilteredRank(
                allNodes, headId, relationType, tailId,
                existingTriples, isTailPrediction: false);

            // Average head and tail ranks (standard protocol)
            double avgRank = (tailRank + headRank) / 2.0;
            double avgRR = (1.0 / tailRank + 1.0 / headRank) / 2.0;

            sumRank += avgRank;
            sumReciprocalRank += avgRR;

            // Hits@K counts each prediction (head and tail) independently,
            // then divides by totalPredictions (= 2 * tripleCount). Standard protocol per PyKEEN.
            foreach (var k in hitsAtKValues)
            {
                if (tailRank <= k) hitsCount[k]++;
                if (headRank <= k) hitsCount[k]++;
            }
        }

        if (tripleCount == 0)
        {
            return new LinkPredictionEvaluation { TestTripleCount = 0 };
        }

        int totalPredictions = tripleCount * 2; // head + tail for each
        var hitsAtK = new Dictionary<int, double>();
        foreach (var k in hitsAtKValues)
        {
            hitsAtK[k] = (double)hitsCount[k] / totalPredictions;
        }

        return new LinkPredictionEvaluation
        {
            MeanReciprocalRank = sumReciprocalRank / tripleCount,
            HitsAtK = hitsAtK,
            MeanRank = sumRank / tripleCount,
            TestTripleCount = tripleCount
        };
    }

    private int ComputeFilteredRank(
        List<GraphNode<T>> allNodes,
        string headId, string relationType, string tailId,
        HashSet<(string headId, string relationType, string tailId)> existingTriples,
        bool isTailPrediction)
    {
        string targetId = isTailPrediction ? tailId : headId;
        double targetScore = NumOps.ToDouble(_embedding.ScoreTriple(headId, relationType, tailId));

        int rank = 1;
        foreach (var node in allNodes)
        {
            string candidateId = node.Id;
            if (candidateId == targetId) continue;

            // Filtered setting: skip existing true triples (except the test triple itself)
            var tripleKey = isTailPrediction
                ? (headId, relationType, candidateId)
                : (candidateId, relationType, tailId);

            if (existingTriples.Contains(tripleKey)) continue;

            double candidateScore = isTailPrediction
                ? NumOps.ToDouble(_embedding.ScoreTriple(headId, relationType, candidateId))
                : NumOps.ToDouble(_embedding.ScoreTriple(candidateId, relationType, tailId));

            // Distance-based: candidate ranks higher if its score is LOWER
            // Semantic: candidate ranks higher if its score is HIGHER
            bool candidateRanksHigher = _embedding.IsDistanceBased
                ? candidateScore < targetScore
                : candidateScore > targetScore;

            if (candidateRanksHigher)
                rank++;
        }

        return rank;
    }

    private double ComputeConfidence(double score, List<(string id, double score)> sortedCandidates)
    {
        if (sortedCandidates.Count == 0) return 1.0;

        double bestScore = sortedCandidates[0].score;
        double worstScore = sortedCandidates[^1].score;
        double range = Math.Abs(worstScore - bestScore);

        if (range < 1e-12) return 1.0;

        // After sorting, index 0 is always the "best" regardless of model type
        // Confidence = how close this score is to the best vs. the worst
        return 1.0 - Math.Abs(score - bestScore) / range;
    }
}
