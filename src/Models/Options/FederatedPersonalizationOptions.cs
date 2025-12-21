namespace AiDotNet.Models.Options;

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
    /// If true and <see cref="Strategy"/> is not "None", the trainer applies a personalization algorithm.
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the personalization strategy name.
    /// </summary>
    /// <remarks>
    /// Supported built-ins:
    /// - "None"
    /// - "FedPer"
    /// - "FedRep"
    /// - "Ditto"
    /// - "pFedMe"
    /// - "Clustered"
    /// </remarks>
    public string Strategy { get; set; } = "None";

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

