namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;

/// <summary>
/// Represents a claim (covariate) extracted from text about an entity, following the
/// Microsoft GraphRAG claim-extraction model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A claim is a factual statement asserted about an entity that
/// carries extra context beyond a simple relationship. For example, in
/// "Acme Corp was fined $2M for pollution in 2021":
/// - Subject: "Acme Corp"
/// - Object: "Environmental Agency" (or NONE if unspecified)
/// - ClaimType: "REGULATORY_VIOLATION"
/// - Description: "Acme Corp was fined $2M for pollution"
/// - Status: "TRUE"
///
/// Claims let a knowledge graph capture assertions (fines, awards, allegations) that don't fit
/// neatly into subject-relation-object triples.
/// </para>
/// </remarks>
public class ExtractedClaim
{
    /// <summary>
    /// The entity the claim is about (subject).
    /// </summary>
    public string Subject { get; set; } = string.Empty;

    /// <summary>
    /// The entity the claim is directed at, or empty/"NONE" if not applicable.
    /// </summary>
    public string Object { get; set; } = string.Empty;

    /// <summary>
    /// The category of the claim (e.g., REGULATORY_VIOLATION, AWARD, ALLEGATION).
    /// </summary>
    public string ClaimType { get; set; } = string.Empty;

    /// <summary>
    /// A human-readable description of the claim.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Truth status of the claim (e.g., TRUE, FALSE, SUSPECTED).
    /// </summary>
    public string Status { get; set; } = string.Empty;
}
