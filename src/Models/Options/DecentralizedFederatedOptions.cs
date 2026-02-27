namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the decentralized topology for peer-to-peer federated learning.
/// </summary>
public enum DecentralizedTopologyType
{
    /// <summary>Gossip — randomized peer selection each round.</summary>
    Gossip,
    /// <summary>Ring AllReduce — bandwidth-optimal ring-based averaging.</summary>
    RingAllReduce,
    /// <summary>DFedAvgM — decentralized averaging with momentum for faster convergence. (Sun et al., TMLR 2023)</summary>
    DFedAvgM,
    /// <summary>DFedBCA — block coordinate ascent with partial model sharing per round. (2024)</summary>
    DFedBCA,
    /// <summary>DeTAG — gradient tracking for exact convergence in decentralized non-convex optimization. (Li et al., 2023)</summary>
    DeTAG,
    /// <summary>Segmented gossip — exchange only model segments per round for bandwidth efficiency. (Bellet et al., 2024)</summary>
    SegmentedGossip,
    /// <summary>Time-varying topology — dynamic graph that changes each round for better mixing. (2024)</summary>
    TimeVarying
}

/// <summary>
/// Configuration options for decentralized (serverless) federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Standard FL uses a central server to aggregate models. Decentralized FL
/// removes this server — nodes communicate directly with each other using peer-to-peer protocols.
/// This eliminates single point of failure and can be more robust in edge/IoT deployments.</para>
/// </remarks>
public class DecentralizedFederatedOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets whether decentralized mode is enabled. Default: false.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the topology type. Default: Gossip.
    /// </summary>
    public DecentralizedTopologyType Topology { get; set; } = DecentralizedTopologyType.Gossip;

    /// <summary>
    /// Gets or sets the gossip fanout (number of random peers per round). Default: 2.
    /// </summary>
    public int GossipFanout { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of mixing rounds per training round. Default: 3.
    /// </summary>
    /// <remarks>
    /// More mixing rounds improve convergence but increase communication cost.
    /// </remarks>
    public int MixingRoundsPerTrainingRound { get; set; } = 3;

}
