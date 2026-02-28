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

    // --- DFedAvgM ---

    /// <summary>
    /// Gets or sets the momentum coefficient for DFedAvgM.
    /// </summary>
    /// <value>The momentum factor applied to decentralized model averaging. Default: 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> DFedAvgM adds momentum to decentralized averaging, similar
    /// to how SGD with momentum accelerates training. Higher values (closer to 1.0) give more
    /// smoothing across rounds, improving convergence speed on non-IID data.</para>
    /// </remarks>
    public double DFedAvgMMomentum { get; set; } = 0.9;

    // --- DFedBCA ---

    /// <summary>
    /// Gets or sets the number of parameter blocks for DFedBCA.
    /// </summary>
    /// <value>The number of disjoint blocks the model is partitioned into. Default: 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> DFedBCA splits the model into blocks and only exchanges one
    /// block per round, reducing per-round communication by a factor of NumBlocks. More blocks
    /// mean less bandwidth per round but slower convergence per round.</para>
    /// </remarks>
    public int DFedBCANumBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the block selection strategy for DFedBCA.
    /// </summary>
    /// <value>How blocks are chosen each round. Default: <see cref="BlockSelectionMode.Cyclic"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cyclic rotates through blocks in order (simple, predictable).
    /// Random picks a block at random each round. ImportanceBased selects the block with the
    /// largest gradient magnitude, focusing bandwidth on the most-changing parameters.</para>
    /// </remarks>
    public BlockSelectionMode DFedBCASelectionStrategy { get; set; } = BlockSelectionMode.Cyclic;

    // --- DeTAG ---

    /// <summary>
    /// Gets or sets the learning rate for DeTAG gradient tracking.
    /// </summary>
    /// <value>The step size for the gradient tracking correction term. Default: 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> DeTAG uses gradient tracking to correct for the bias introduced
    /// by decentralized averaging. The learning rate controls how aggressively the tracking variable
    /// is updated. Too large may cause instability; too small slows convergence.</para>
    /// </remarks>
    public double DeTAGLearningRate { get; set; } = 0.01;

    // --- Segmented Gossip ---

    /// <summary>
    /// Gets or sets the number of model segments for SegmentedGossip.
    /// </summary>
    /// <value>The number of segments the model is divided into for partial gossip exchanges. Default: 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of exchanging the full model with a peer, segmented gossip
    /// only exchanges one segment per round. This reduces bandwidth by a factor of NumSegments while
    /// eventually exchanging all parameters over multiple rounds.</para>
    /// </remarks>
    public int SegmentedGossipNumSegments { get; set; } = 4;

    // --- Time-Varying Topology ---

    /// <summary>
    /// Gets or sets the random seed for time-varying topology generation.
    /// </summary>
    /// <value>The seed for reproducible random topology generation. Default: 42.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A time-varying topology changes the communication graph each round
    /// (which peers talk to which). This improves mixing — information spreads faster across the
    /// network compared to a fixed topology. The seed ensures reproducibility.</para>
    /// </remarks>
    public int TimeVaryingSeed { get; set; } = 42;
}

/// <summary>
/// Block selection mode for DFedBCA protocol.
/// </summary>
public enum BlockSelectionMode
{
    /// <summary>Rotate through blocks in order.</summary>
    Cyclic = 0,
    /// <summary>Select blocks randomly each round.</summary>
    Random = 1,
    /// <summary>Select blocks based on gradient importance.</summary>
    ImportanceBased = 2
}
