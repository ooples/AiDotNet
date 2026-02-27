namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the personalization strategy for federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each strategy determines how clients adapt the global model to their local data.</para>
/// </remarks>
public enum FederatedPersonalizationStrategy
{
    /// <summary>No personalization — all parameters are aggregated globally.</summary>
    None,
    /// <summary>FedPer — personalize the last (classification head) layers, share the body.</summary>
    FedPer,
    /// <summary>FedRep — learn shared representations with personalized heads; alternating optimization.</summary>
    FedRep,
    /// <summary>Ditto — train a regularized personalized model alongside the global model.</summary>
    Ditto,
    /// <summary>pFedMe — Moreau-envelope-based personalization with proximal local solver.</summary>
    PFedMe,
    /// <summary>Clustered — cluster clients by gradient similarity, aggregate within clusters.</summary>
    Clustered,
    /// <summary>FedBABU — freeze head during FL, fine-tune body, then locally fine-tune head.</summary>
    FedBABU,
    /// <summary>FedRoD — dual classifiers: one aggregated generic + one local personalized.</summary>
    FedRoD,
    /// <summary>FedCP — conditional computation policy routing inputs to model subsets.</summary>
    FedCP,
    /// <summary>kNN-Per — kNN cache over global features for zero-cost personalization at inference.</summary>
    KNNPer,
    /// <summary>FedSelect — learned sparse binary masks determining personalized vs shared params.</summary>
    FedSelect,
    /// <summary>pFedGate — gated layer-wise mixture of local and global parameters.</summary>
    PFedGate,
    /// <summary>FedAGHN — adaptive gradient-based heterogeneous networks.</summary>
    FedAGHN,
    /// <summary>FedPAC — personalization via aggregation and calibration with prototype alignment.</summary>
    FedPAC
}

/// <summary>
/// Configuration options for personalized federated learning (PFL).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Personalization means each client can end up with a model that works better for its own data,
/// while still learning shared knowledge from other clients.
///
/// This options class controls which personalization algorithm is used (FedPer, FedRep, Ditto, pFedMe, clustered, etc.)
/// and the key hyperparameters for those algorithms.
/// </remarks>
public sealed class FederatedPersonalizationOptions
{
    /// <summary>
    /// Gets or sets whether personalization is enabled.
    /// </summary>
    /// <remarks>
    /// If true and <see cref="Strategy"/> is not <see cref="FederatedPersonalizationStrategy.None"/>,
    /// the trainer applies a personalization algorithm.
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the personalization strategy.
    /// </summary>
    public FederatedPersonalizationStrategy Strategy { get; set; } = FederatedPersonalizationStrategy.None;

    /// <summary>
    /// Gets or sets the fraction of parameters treated as "personalized" (not aggregated globally).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Many PFL methods conceptually "split" a model into:
    /// - Shared parameters (learned collaboratively)
    /// - Personalized parameters (kept local per client or per cluster)
    ///
    /// When using vector-based models, we approximate this by taking the last N% of the parameter vector.
    /// </remarks>
    public double PersonalizedParameterFraction { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the number of extra local adaptation epochs applied after receiving the aggregated global model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is an optional "fine-tune" step that can improve local performance.
    /// Set to 0 to disable.
    /// </remarks>
    public int LocalAdaptationEpochs { get; set; } = 0;

    /// <summary>
    /// Gets or sets the Ditto regularization strength (lambda).
    /// </summary>
    /// <remarks>
    /// Higher values keep personalized models closer to the current global model.
    /// </remarks>
    public double DittoLambda { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the pFedMe proximal strength (mu).
    /// </summary>
    public double PFedMeMu { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of inner proximal steps for pFedMe (K).
    /// </summary>
    public int PFedMeInnerSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of clusters used for clustered personalization.
    /// </summary>
    public int ClusterCount { get; set; } = 3;
}

