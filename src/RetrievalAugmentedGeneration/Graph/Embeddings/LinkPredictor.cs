using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Engines;

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
[ComponentType(ComponentType.Retriever)]
[PipelineStage(PipelineStage.Retrieval)]
public class LinkPredictor<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

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

        var candidateIds = graph.GetAllNodes()
            .Select(node => node.Id)
            .Where(id => id != headId && !(filterExisting && existingTails.Contains(id)))
            .ToList();
        var candidates = ScoreCandidates(candidateIds, id => _embedding.ScoreTriple(headId, relationType, id));

        // Distance-based: lower=better → sort ascending. Semantic: higher=better → sort descending.
        if (_embedding.IsDistanceBased)
            candidates.Sort((a, b) => a.score.CompareTo(b.score));
        else
            candidates.Sort((a, b) => b.score.CompareTo(a.score));

        return candidates.Take(topK).Select(c => new PredictedTriple
        {
            HeadId = headId,
            RelationType = relationType,
            TailId = c.id,
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

        var candidateIds = graph.GetAllNodes()
            .Select(node => node.Id)
            .Where(id => id != tailId && !(filterExisting && existingHeads.Contains(id)))
            .ToList();
        var candidates = ScoreCandidates(candidateIds, id => _embedding.ScoreTriple(id, relationType, tailId));

        if (_embedding.IsDistanceBased)
            candidates.Sort((a, b) => a.score.CompareTo(b.score));
        else
            candidates.Sort((a, b) => b.score.CompareTo(a.score));

        return candidates.Take(topK).Select(c => new PredictedTriple
        {
            HeadId = c.id,
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
        T targetScore = _embedding.ScoreTriple(headId, relationType, tailId);

        // Filtered setting: skip the target and existing true triples (except the test triple itself)
        var candidateScores = allNodes
            .Select(node => node.Id)
            .Where(candidateId => candidateId != targetId && !existingTriples.Contains(isTailPrediction
                ? (headId, relationType, candidateId)
                : (candidateId, relationType, tailId)))
            .Select(candidateId => isTailPrediction
                ? _embedding.ScoreTriple(headId, relationType, candidateId)
                : _embedding.ScoreTriple(candidateId, relationType, tailId))
            .ToArray();
        if (candidateScores.Length == 0) return 1;

        // Distance-based: candidate ranks higher if its score is LOWER
        // Semantic: candidate ranks higher if its score is HIGHER
        var scores = new Tensor<T>(candidateScores, [candidateScores.Length]);
        var ranksHigher = _embedding.IsDistanceBased
            ? Engine.TensorLessThan(scores, targetScore)
            : Engine.TensorGreaterThan(scores, targetScore);

        return 1 + Engine.TensorCountNonzero(ranksHigher);
    }

    /// <summary>
    /// Scores each candidate with the embedding model and converts the scores to double in one pass
    /// (<see cref="PredictedTriple.Score"/> is double).
    /// </summary>
    private static List<(string id, double score)> ScoreCandidates(List<string> candidateIds, Func<string, T> score)
    {
        var candidates = new List<(string id, double score)>(candidateIds.Count);
        if (candidateIds.Count == 0) return candidates;

        var scores = new Tensor<T>(candidateIds.Select(score).ToArray(), [candidateIds.Count]).Cast<double>().ToArray();
        candidates.AddRange(candidateIds.Zip(scores, (id, value) => (id, value)));
        return candidates;
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
