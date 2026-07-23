namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Defines the contract for knowledge graph embedding models that learn vector representations
/// of entities and relations.
/// </summary>
/// <typeparam name="T">The numeric type used for embedding calculations.</typeparam>
/// <remarks>
/// <para>
/// Knowledge graph embeddings map entities and relations into a continuous vector space,
/// enabling operations like link prediction, entity clustering, and relation inference.
/// </para>
/// <para><b>For Beginners:</b> A knowledge graph stores facts as (head, relation, tail) triples,
/// e.g., (Einstein, born_in, Germany). Embedding models learn numeric vectors for each entity
/// and relation so that valid triples score higher than invalid ones. This lets you:
/// - Predict missing links: "What city was Tesla born in?"
/// - Find similar entities: Entities with similar vectors are semantically related
/// - Evaluate triple plausibility: Score how likely a new fact is to be true
/// </para>
/// </remarks>
public interface IKnowledgeGraphEmbedding<T>
{
    /// <summary>
    /// Gets whether the model has been trained.
    /// </summary>
    bool IsTrained { get; }

    /// <summary>
    /// Gets whether this model uses distance-based scoring (lower score = more plausible triple)
    /// or semantic matching scoring (higher score = more plausible triple).
    /// </summary>
    /// <remarks>
    /// <para>Distance-based models (TransE, RotatE, TemporalTransE): lower scores are better.</para>
    /// <para>Semantic matching models (ComplEx, DistMult): higher scores are better.</para>
    /// </remarks>
    bool IsDistanceBased { get; }

    /// <summary>
    /// Gets the dimensionality of the embedding vectors.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Trains the embedding model on the triples from the given knowledge graph.
    /// </summary>
    /// <param name="graph">The knowledge graph containing the training triples.</param>
    /// <param name="options">Training options (learning rate, epochs, etc.).</param>
    /// <returns>Training result with loss history and statistics.</returns>
    KGEmbeddingTrainingResult Train(KnowledgeGraph<T> graph, KGEmbeddingOptions? options = null);

    /// <summary>
    /// Gets the learned embedding vector for an entity.
    /// </summary>
    /// <param name="entityId">The entity's node ID in the knowledge graph.</param>
    /// <returns>The embedding vector, or null if the entity was not in the training data.</returns>
    T[]? GetEntityEmbedding(string entityId);

    /// <summary>
    /// Gets the learned embedding vector for a relation type.
    /// </summary>
    /// <param name="relationType">The relation type string (e.g., "born_in").</param>
    /// <returns>The embedding vector, or null if the relation was not in the training data.</returns>
    T[]? GetRelationEmbedding(string relationType);

    /// <summary>
    /// Scores a triple (head, relation, tail) — lower scores indicate more plausible triples
    /// for distance-based models, higher scores for semantic matching models.
    /// </summary>
    /// <param name="headId">The head entity ID.</param>
    /// <param name="relationType">The relation type.</param>
    /// <param name="tailId">The tail entity ID.</param>
    /// <returns>The plausibility score for the triple.</returns>
    T ScoreTriple(string headId, string relationType, string tailId);
}
