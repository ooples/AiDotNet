using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;
using System.Collections.Generic;

namespace AiDotNet.Models.Results;

/// <summary>
/// Contains the results of knowledge graph processing, including trained embeddings,
/// community structure, and link prediction evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for graph operations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> After building a model with <c>ConfigureKnowledgeGraph()</c>,
/// this result contains everything that was computed:
/// - EmbeddingTrainingResult: Training statistics if embeddings were trained
/// - TrainedEmbedding: The trained embedding model for scoring triples
/// - CommunityStructure: Detected communities if Leiden was run
/// - CommunitySummaries: Human-readable descriptions of each community
/// - LinkPredictionEvaluation: Quality metrics if link prediction was evaluated
/// - EnhancedGraphRAG: The configured GraphRAG instance for querying
/// </para>
/// </remarks>
public class KnowledgeGraphResult<T>
{
    /// <summary>
    /// Training result from embedding model training, if embeddings were trained.
    /// </summary>
    public KGEmbeddingTrainingResult? EmbeddingTrainingResult { get; set; }

    /// <summary>
    /// The trained embedding model, if training was performed.
    /// </summary>
    [Newtonsoft.Json.JsonIgnore]
    public IKnowledgeGraphEmbedding<T>? TrainedEmbedding { get; set; }

    /// <summary>
    /// Community detection results from the Leiden algorithm, if community detection was run.
    /// </summary>
    public LeidenResult? CommunityStructure { get; set; }

    /// <summary>
    /// Human-readable summaries of detected communities.
    /// </summary>
    public List<CommunitySummary>? CommunitySummaries { get; set; }

    /// <summary>
    /// Link prediction evaluation metrics, if evaluation was performed.
    /// </summary>
    public LinkPredictionEvaluation? LinkPredictionEvaluation { get; set; }

    /// <summary>
    /// The configured EnhancedGraphRAG instance for querying.
    /// </summary>
    [Newtonsoft.Json.JsonIgnore]
    public EnhancedGraphRAG<T>? EnhancedGraphRAG { get; set; }
}
