namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how to handle missing cross-client neighbor nodes in subgraph-level FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a graph is split across clients, some edges point to nodes
/// on other clients. A GNN needs neighbor features for message passing, but it can't directly
/// access nodes on other clients. Pseudo-node strategies fill in these missing neighbors:</para>
/// <list type="bullet">
/// <item><description><b>None:</b> Ignore missing neighbors entirely. Simplest but hurts GNN expressiveness.</description></item>
/// <item><description><b>FeatureAverage:</b> Replace each missing neighbor with the average feature vector of known
/// nodes. Low-cost approximation.</description></item>
/// <item><description><b>GeneratorBased:</b> Train a small neural network to generate realistic pseudo-node
/// features based on the local subgraph structure. Best quality but more compute.</description></item>
/// <item><description><b>ZeroFill:</b> Replace missing neighbors with zero vectors. Fast but loses information.</description></item>
/// </list>
/// </remarks>
public enum PseudoNodeStrategy
{
    /// <summary>Ignore missing cross-client neighbors.</summary>
    None,

    /// <summary>Use average feature vectors for missing neighbors.</summary>
    FeatureAverage,

    /// <summary>Use a trained generator to produce pseudo-node features.</summary>
    GeneratorBased,

    /// <summary>Fill missing neighbors with zero vectors.</summary>
    ZeroFill
}
