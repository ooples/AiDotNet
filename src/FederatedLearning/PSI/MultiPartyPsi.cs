using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.PSI;

/// <summary>
/// Implements multi-party Private Set Intersection for 3 or more parties.
/// </summary>
/// <remarks>
/// <para>Multi-party PSI extends two-party protocols to find the intersection common
/// to all participating parties. The result contains only elements present in every
/// party's set.</para>
///
/// <para><b>For Beginners:</b> Two-party PSI is like two people comparing guest lists.
/// Multi-party PSI is like a group of people finding guests who are on everyone's list.
/// For example, 3 hospitals finding patients who appear in all 3 systems.</para>
///
/// <para>This implementation uses a star topology: a designated leader runs pairwise
/// two-party PSI with each other party, then intersects all pairwise results to produce
/// the global intersection.</para>
///
/// <para><b>Complexity:</b> (P-1) two-party PSI executions where P is the number of parties,
/// plus O(n) intersection of pairwise results.</para>
///
/// <para><b>Security:</b> In the star topology, the leader learns all pairwise intersections
/// (which is more than just the global intersection). For stronger security, tree or
/// ring topologies can be used at the cost of more rounds.</para>
///
/// <para><b>Reference:</b> Kolesnikov et al., "Efficient Batched Oblivious PRF with Applications
/// to Private Set Intersection", ACM CCS 2016. Li et al., "Lightweight MP-PSI", 2025.</para>
/// </remarks>
public class MultiPartyPsi : PsiBase
{
    private readonly IPrivateSetIntersection _twoPartyProtocol;

    /// <summary>
    /// Initializes a new instance of <see cref="MultiPartyPsi"/> using a specified two-party PSI protocol.
    /// </summary>
    /// <param name="twoPartyProtocol">The two-party PSI protocol to use for pairwise intersections.</param>
    public MultiPartyPsi(IPrivateSetIntersection twoPartyProtocol)
    {
        _twoPartyProtocol = twoPartyProtocol ?? throw new ArgumentNullException(nameof(twoPartyProtocol));
    }

    /// <summary>
    /// Initializes a new instance of <see cref="MultiPartyPsi"/> using Diffie-Hellman PSI for pairwise intersections.
    /// </summary>
    public MultiPartyPsi() : this(new DiffieHellmanPsi())
    {
    }

    /// <inheritdoc/>
    public override string ProtocolName => $"MultiParty-{_twoPartyProtocol.ProtocolName}";

    /// <inheritdoc/>
    protected override PsiResult ComputeExactIntersection(
        IReadOnlyList<string> localIds, IReadOnlyList<string> remoteIds, PsiOptions options)
    {
        // Two-party case: delegate to underlying protocol
        return _twoPartyProtocol.ComputeIntersection(localIds, remoteIds, options);
    }

    /// <summary>
    /// Computes the intersection across multiple parties using the star topology.
    /// </summary>
    /// <param name="partySets">The ID sets for each party. The first party is the leader.</param>
    /// <param name="options">Protocol configuration options.</param>
    /// <returns>A <see cref="PsiResult"/> containing the global intersection and per-party alignment mappings.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a list of ID sets (one per party) and finds
    /// the IDs that appear in ALL parties' sets. The first party acts as the coordinator.</para>
    ///
    /// <para><b>Algorithm (Star Topology):</b></para>
    /// <list type="number">
    /// <item><description>Leader (party 0) runs two-party PSI with each other party.</description></item>
    /// <item><description>Leader intersects all pairwise results to get the global intersection.</description></item>
    /// <item><description>Leader builds alignment mappings for all parties.</description></item>
    /// </list>
    /// </remarks>
    public MultiPartyPsiResult ComputeMultiPartyIntersection(
        IReadOnlyList<IReadOnlyList<string>> partySets, PsiOptions options)
    {
        if (partySets is null)
        {
            throw new ArgumentNullException(nameof(partySets));
        }

        if (partySets.Count < 2)
        {
            throw new ArgumentException("Multi-party PSI requires at least 2 parties.", nameof(partySets));
        }

        var leaderSet = partySets[0];

        // Step 1: Leader runs pairwise PSI with each other party
        var pairwiseResults = new PsiResult[partySets.Count - 1];
        for (int i = 1; i < partySets.Count; i++)
        {
            pairwiseResults[i - 1] = _twoPartyProtocol.ComputeIntersection(leaderSet, partySets[i], options);
        }

        // Step 2: Intersect all pairwise results
        // Start with the first pairwise intersection
        var globalIntersection = new HashSet<string>(
            pairwiseResults[0].IntersectionIds, StringComparer.Ordinal);

        for (int i = 1; i < pairwiseResults.Length; i++)
        {
            var pairwiseSet = new HashSet<string>(
                pairwiseResults[i].IntersectionIds, StringComparer.Ordinal);
            globalIntersection.IntersectWith(pairwiseSet);
        }

        var sortedIntersection = globalIntersection.OrderBy(id => id, StringComparer.Ordinal).ToList();

        // Step 3: Build per-party alignment mappings
        var partyMappings = new Dictionary<int, int>[partySets.Count];
        for (int p = 0; p < partySets.Count; p++)
        {
            var mapping = new Dictionary<int, int>();
            var partyLookup = new Dictionary<string, int>(partySets[p].Count, StringComparer.Ordinal);
            for (int i = 0; i < partySets[p].Count; i++)
            {
                if (!partyLookup.ContainsKey(partySets[p][i]))
                {
                    partyLookup[partySets[p][i]] = i;
                }
            }

            for (int sharedIdx = 0; sharedIdx < sortedIntersection.Count; sharedIdx++)
            {
                if (partyLookup.TryGetValue(sortedIntersection[sharedIdx], out int localIdx))
                {
                    mapping[localIdx] = sharedIdx;
                }
            }

            partyMappings[p] = mapping;
        }

        return new MultiPartyPsiResult
        {
            IntersectionIds = sortedIntersection,
            IntersectionSize = sortedIntersection.Count,
            PartyAlignmentMappings = partyMappings,
            NumberOfParties = partySets.Count
        };
    }
}

/// <summary>
/// Contains results of a multi-party PSI computation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Similar to <see cref="PsiResult"/> but with alignment mappings
/// for every participating party, not just two.</para>
/// </remarks>
public class MultiPartyPsiResult
{
    /// <summary>
    /// Gets or sets the intersecting entity IDs found across all parties.
    /// </summary>
    public IReadOnlyList<string> IntersectionIds { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the number of intersecting elements.
    /// </summary>
    public int IntersectionSize { get; set; }

    /// <summary>
    /// Gets or sets per-party alignment mappings (localIndex -> sharedIndex).
    /// Index 0 is the leader, subsequent indices correspond to other parties.
    /// </summary>
    public IReadOnlyDictionary<int, int>[] PartyAlignmentMappings { get; set; } = Array.Empty<IReadOnlyDictionary<int, int>>();

    /// <summary>
    /// Gets or sets the number of parties that participated.
    /// </summary>
    public int NumberOfParties { get; set; }

    /// <summary>
    /// Gets or sets the total execution time.
    /// </summary>
    public TimeSpan ExecutionTime { get; set; }
}
