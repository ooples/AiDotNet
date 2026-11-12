namespace AiDotNet.Enums;

/// <summary>
/// How an R-GCN layer normalizes the messages a node receives along one relation type: the c_{i,r} of
/// Schlichtkrull et al. 2018, Eq. 2, h_i' = sigma(W_0 h_i + sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each node adds up the transformed states of its neighbours under every relation.
/// <see cref="Sum"/> uses that sum as it is; <see cref="Mean"/> divides it by how many neighbours the node has
/// under that relation, so a node with many neighbours is not drowned out by sheer count.
/// </para>
/// </remarks>
public enum RelationalAggregation
{
    /// <summary>
    /// c_{i,r} = 1: the adjacency weights are used as given, so pass a pre-normalized adjacency for any other
    /// normalization.
    /// </summary>
    Sum = 0,

    /// <summary>
    /// c_{i,r} = the node's weighted in-degree under relation r, which for a 0/1 adjacency is the |N_i^r| the
    /// paper uses. A node with no neighbours under r receives nothing from it.
    /// </summary>
    Mean = 1,
}
