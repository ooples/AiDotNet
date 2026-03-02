namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated meta-learning strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each strategy determines how the server uses client adaptation results
/// to update the global initialization for better per-client fine-tuning.</para>
/// </remarks>
public enum FederatedMetaLearningStrategy
{
    /// <summary>No meta-learning — standard aggregation of client updates.</summary>
    None,
    /// <summary>Reptile — first-order meta-update based on post-adaptation parameters.</summary>
    Reptile,
    /// <summary>PerFedAvg — per-client FedAvg treated as a Reptile-style first-order update.</summary>
    PerFedAvg,
    /// <summary>FedMAML — first-order MAML approximation; full second-order requires explicit gradient support.</summary>
    FedMAML
}

/// <summary>
/// Configuration options for federated meta-learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Meta-learning in federated settings aims to learn a "good starting point" (initial model)
/// that can adapt quickly to each client's local data with a small amount of fine-tuning.
///
/// In this library, federated meta-learning is implemented as an alternative server update rule that uses
/// client adaptation results (post-local training) to update the global initialization.
/// </remarks>
public sealed class FederatedMetaLearningOptions
{
    /// <summary>
    /// Gets or sets whether federated meta-learning is enabled.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the federated meta-learning strategy.
    /// </summary>
    public FederatedMetaLearningStrategy Strategy { get; set; } = FederatedMetaLearningStrategy.None;

    /// <summary>
    /// Gets or sets the server meta learning rate applied to the average adaptation delta.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls how strongly the server moves the global initialization toward the
    /// client-adapted models each round. A value of 1.0 means "move fully to the average adapted model"
    /// (similar to FedAvg when inner epochs match).
    /// </remarks>
    public double MetaLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of local adaptation epochs used for the inner loop.
    /// </summary>
    /// <remarks>
    /// If not set (or &lt;= 0), the trainer falls back to the federated <c>LocalEpochs</c> value.
    /// </remarks>
    public int InnerEpochs { get; set; } = 0;
}

