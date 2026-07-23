using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Defines the interface for Private Set Intersection protocols.
/// </summary>
/// <remarks>
/// <para>PSI allows two or more parties to compute the intersection of their sets
/// without revealing elements not in the intersection. This is the foundational
/// building block for entity alignment in vertical federated learning.</para>
///
/// <para><b>For Beginners:</b> Imagine two hospitals that want to find shared patients
/// without revealing their full patient lists to each other. PSI solves this:
/// each hospital inputs its patient IDs, and the protocol outputs only the IDs
/// that appear in both hospitals' records, without either hospital learning about
/// patients unique to the other.</para>
///
/// <para>All implementations are simulation-safe: they operate on in-memory data
/// structures rather than requiring actual network communication, making them
/// suitable for testing and single-machine VFL experiments.</para>
/// </remarks>
public interface IPrivateSetIntersection
{
    /// <summary>
    /// Gets the name of this PSI protocol.
    /// </summary>
    string ProtocolName { get; }

    /// <summary>
    /// Computes the intersection between the local party's IDs and the remote party's IDs.
    /// </summary>
    /// <param name="localIds">The local party's set of entity identifiers.</param>
    /// <param name="remoteIds">The remote party's set of entity identifiers.</param>
    /// <param name="options">Protocol configuration options.</param>
    /// <returns>A <see cref="PsiResult"/> containing the intersection and alignment mappings.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this method with both parties' ID lists.
    /// The result tells you which IDs are shared and how to align the data rows
    /// for joint training.</para>
    ///
    /// <para>In a real deployment, this would involve cryptographic message exchange
    /// between parties. In simulation mode (used here), both sets are provided directly
    /// and the protocol computes the intersection while demonstrating the algorithmic
    /// approach each protocol would use.</para>
    /// </remarks>
    PsiResult ComputeIntersection(IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options);

    /// <summary>
    /// Computes only the cardinality (count) of the intersection without revealing the actual elements.
    /// </summary>
    /// <param name="localIds">The local party's set of entity identifiers.</param>
    /// <param name="remoteIds">The remote party's set of entity identifiers.</param>
    /// <param name="options">Protocol configuration options.</param>
    /// <returns>The number of elements in the intersection.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sometimes you only need to know how many IDs are shared,
    /// not which ones. This is faster and reveals less information. Useful for deciding
    /// whether there's enough overlap to justify VFL training.</para>
    /// </remarks>
    int ComputeCardinality(IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options);
}
