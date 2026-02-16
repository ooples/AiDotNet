namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how a graph is partitioned across federated clients.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In graph FL, a large graph must be split across clients. The partition
/// strategy determines how nodes are assigned:</para>
/// <list type="bullet">
/// <item><description><b>Random:</b> Assign nodes randomly. Simple but creates many cross-client edges.</description></item>
/// <item><description><b>Metis:</b> Use the METIS algorithm to minimize edge cuts. Best partition quality
/// but requires seeing the full graph structure.</description></item>
/// <item><description><b>StreamPartition:</b> Assign nodes as they arrive in a stream. Good for dynamic graphs.</description></item>
/// <item><description><b>CommunityBased:</b> Detect communities (Louvain, label propagation) and assign each
/// community to a client. Preserves local structure well.</description></item>
/// <item><description><b>Preassigned:</b> Nodes are already assigned to clients (e.g., each hospital has its own
/// patient network). No partitioning needed.</description></item>
/// </list>
/// </remarks>
public enum GraphPartitionStrategy
{
    /// <summary>Random node assignment.</summary>
    Random,

    /// <summary>METIS-based minimum edge-cut partitioning.</summary>
    Metis,

    /// <summary>Stream-based partitioning for dynamic graphs.</summary>
    StreamPartition,

    /// <summary>Community detection-based partitioning.</summary>
    CommunityBased,

    /// <summary>Nodes are pre-assigned to clients.</summary>
    Preassigned
}
