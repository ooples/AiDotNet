namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies how to choose the split point in a split neural network for vertical FL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, the neural network is "split" into two parts:
/// a bottom model (runs locally at each party) and a top model (runs at the coordinator).
/// The split point determines where the network is divided. Choosing the right split point
/// affects both privacy (deeper splits leak less information) and efficiency (deeper splits
/// require more local computation but less communication).</para>
/// </remarks>
public enum SplitPointStrategy
{
    /// <summary>
    /// The user specifies exactly which layer to split at. Provides full control but
    /// requires knowledge of the model architecture.
    /// </summary>
    Manual,

    /// <summary>
    /// Automatically selects the split point that minimizes information leakage while
    /// maintaining model accuracy. Uses mutual information estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The system automatically finds the best layer to split at,
    /// balancing privacy and accuracy. This is the recommended option for most use cases.</para>
    /// </remarks>
    AutoOptimal,

    /// <summary>
    /// Selects the split point that balances computational load across parties.
    /// Useful when parties have different hardware capabilities.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If one party has a powerful GPU and another has a weak CPU,
    /// this option ensures each party does a fair share of the work relative to its capacity.</para>
    /// </remarks>
    BalancedCompute
}
