using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Configuration options for advanced knowledge graph capabilities including embeddings,
/// community detection, link prediction, temporal queries, and KG construction.
/// </summary>
/// <remarks>
/// <para>
/// These options are separate from <c>ConfigureRetrievalAugmentedGeneration()</c>, which handles
/// low-level plumbing (IGraphStore, KnowledgeGraph, HybridGraphRetriever). This class configures
/// higher-level features built on top of the existing infrastructure.
/// </para>
/// <para><b>For Beginners:</b> After setting up your knowledge graph via <c>ConfigureRetrievalAugmentedGeneration()</c>,
/// use <c>ConfigureKnowledgeGraph()</c> to enable advanced features:
/// - Train embeddings to enable link prediction and entity similarity
/// - Detect communities for global search capabilities
/// - Enable temporal queries for time-aware reasoning
/// - Construct a KG automatically from text input
/// </para>
/// </remarks>
public class KnowledgeGraphOptions
{
    /// <summary>
    /// Whether to train knowledge graph embeddings. Default: false.
    /// </summary>
    public bool? TrainEmbeddings { get; set; }

    /// <summary>
    /// Type of embedding model to use. Default: TransE.
    /// </summary>
    public KGEmbeddingType? EmbeddingType { get; set; }

    /// <summary>
    /// Options for embedding model training.
    /// </summary>
    public KGEmbeddingOptions? EmbeddingOptions { get; set; }

    /// <summary>
    /// GraphRAG retrieval mode. Default: Local.
    /// </summary>
    public GraphRAGMode? GraphRAGMode { get; set; }

    /// <summary>
    /// Options for GraphRAG retrieval.
    /// </summary>
    public GraphRAGOptions? GraphRAGOptions { get; set; }

    /// <summary>
    /// Whether to enable link prediction. Default: false.
    /// </summary>
    public bool? EnableLinkPrediction { get; set; }

    /// <summary>
    /// Number of top predictions to return in link prediction. Default: 10.
    /// </summary>
    public int? LinkPredictionTopK { get; set; }

    /// <summary>
    /// Whether to enable temporal query support. Default: false.
    /// </summary>
    public bool? EnableTemporalQueries { get; set; }

    /// <summary>
    /// Options for KG construction from text. If null, KG construction is not performed.
    /// </summary>
    public KGConstructionOptions? ConstructionOptions { get; set; }

    /// <summary>
    /// Text documents to construct the knowledge graph from. Each string is processed
    /// independently through the entity/relation extraction pipeline.
    /// Set this along with <see cref="ConstructionOptions"/> to enable automatic KG construction.
    /// </summary>
    public List<string>? ConstructionTexts { get; set; }

    internal bool GetEffectiveTrainEmbeddings() => TrainEmbeddings ?? false;
    internal KGEmbeddingType GetEffectiveEmbeddingType() => EmbeddingType ?? Enums.KGEmbeddingType.TransE;
    internal bool GetEffectiveEnableLinkPrediction() => EnableLinkPrediction ?? false;
    internal int GetEffectiveLinkPredictionTopK() => LinkPredictionTopK ?? 10;
    internal bool GetEffectiveEnableTemporalQueries() => EnableTemporalQueries ?? false;
}
