namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of knowledge graph embedding model to use.
/// </summary>
/// <remarks>
/// <para>
/// Knowledge graph embeddings map entities and relations to continuous vector spaces,
/// enabling mathematical reasoning about graph structure. Different models capture
/// different types of relational patterns.
/// </para>
/// <para><b>For Beginners:</b> Each embedding type has strengths for different relationship patterns:
/// - TransE: Good for one-to-one relations (born_in, capital_of)
/// - RotatE: Handles symmetric, antisymmetric, inversion, and composition patterns
/// - ComplEx: Best for symmetric and antisymmetric relations
/// - DistMult: Best for symmetric relations (similar_to, married_to)
/// - TemporalTransE: TransE with time-awareness for facts that change over time
/// </para>
/// </remarks>
public enum KGEmbeddingType
{
    /// <summary>
    /// TransE: Translational embedding where h + r ≈ t.
    /// </summary>
    TransE,

    /// <summary>
    /// RotatE: Rotation-based embedding in complex space where t = h ∘ r.
    /// </summary>
    RotatE,

    /// <summary>
    /// ComplEx: Complex-valued embedding using Hermitian dot product.
    /// </summary>
    ComplEx,

    /// <summary>
    /// DistMult: Bilinear diagonal model scoring Σ(h_k · r_k · t_k).
    /// </summary>
    DistMult,

    /// <summary>
    /// TemporalTransE: Time-aware TransE with discretized time bins.
    /// </summary>
    TemporalTransE
}
