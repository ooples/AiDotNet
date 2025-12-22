namespace AiDotNet.Models;

/// <summary>
/// Contains metadata and metrics about federated learning training progress and results.
/// </summary>
/// <remarks>
/// This class tracks various metrics throughout the federated learning process to help
/// monitor training progress, diagnose issues, and evaluate model quality.
///
/// <b>For Beginners:</b> Metadata is like a training diary that records what happened
/// during federated learning - how long it took, how accurate the model became, which
/// clients participated, etc.
///
/// Think of this as a comprehensive training report containing:
/// - Performance metrics: Accuracy, loss, convergence
/// - Resource usage: Time, communication costs
/// - Participation: Which clients contributed
/// - Privacy tracking: Privacy budget consumption
///
/// For example, after training you might see:
/// - Total rounds: 50 (out of max 100)
/// - Final accuracy: 92.5%
/// - Training time: 2 hours
/// - Total clients participated: 100
/// - Privacy budget used: ε=5.0 (out of 10.0 total)
/// </remarks>
public class FederatedLearningMetadata
{
    /// <summary>
    /// The recommended key for storing federated learning metadata inside <see cref="ModelMetadata{T}.Properties"/>.
    /// </summary>
    public const string MetadataKey = "FederatedLearning";

    /// <summary>
    /// Gets or sets the number of federated learning rounds completed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A round is one complete cycle where clients train and the
    /// server aggregates updates. This counts how many such cycles were completed.
    /// </remarks>
    public int RoundsCompleted { get; set; }

    /// <summary>
    /// Gets or sets the final global model accuracy on validation data.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Accuracy measures how often the model makes correct predictions.
    /// For example, 0.95 means the model is correct 95% of the time.
    ///
    /// This is measured on validation data that wasn't used for training.
    /// </remarks>
    public double FinalAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the final global model loss value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Loss measures how far the model's predictions are from the
    /// true values. Lower loss indicates better performance.
    ///
    /// For example:
    /// - Initial loss: 2.5
    /// - Final loss: 0.3
    /// - The model has improved significantly
    /// </remarks>
    public double FinalLoss { get; set; }

    /// <summary>
    /// Gets or sets the total training time in seconds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The total wall-clock time from start to finish of federated
    /// learning, including all rounds, communication, and aggregation.
    /// </remarks>
    public double TotalTrainingTimeSeconds { get; set; }

    /// <summary>
    /// Gets or sets the average time per round in seconds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How long each round takes on average. Useful for estimating
    /// how long future training runs will take.
    ///
    /// For example:
    /// - 50 rounds completed in 5000 seconds
    /// - Average: 100 seconds per round
    /// - Next training with 100 rounds will take ~10,000 seconds
    /// </remarks>
    public double AverageRoundTimeSeconds { get; set; }

    /// <summary>
    /// Gets or sets the history of loss values across all rounds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A list showing how loss changed after each round.
    /// Useful for plotting learning curves and diagnosing training issues.
    ///
    /// For example: [2.5, 1.8, 1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.3]
    /// Shows steady improvement from 2.5 to 0.3
    /// </remarks>
    public List<double> LossHistory { get; set; } = new List<double>();

    /// <summary>
    /// Gets or sets the history of accuracy values across all rounds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A list showing how accuracy improved after each round.
    ///
    /// For example: [0.60, 0.70, 0.78, 0.84, 0.88, 0.91, 0.93, 0.94, 0.945, 0.95]
    /// Shows accuracy improving from 60% to 95%
    /// </remarks>
    public List<double> AccuracyHistory { get; set; } = new List<double>();

    /// <summary>
    /// Gets or sets the total number of clients that participated across all rounds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How many different clients contributed to the model.
    ///
    /// For example:
    /// - 100 clients available
    /// - 10 clients selected per round
    /// - Over 50 rounds, might have 80 unique participants
    /// </remarks>
    public int TotalClientsParticipated { get; set; }

    /// <summary>
    /// Gets or sets the average number of clients selected per round.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How many clients were active in each round on average.
    ///
    /// For example:
    /// - Round 1: 10 clients
    /// - Round 2: 8 clients (some unavailable)
    /// - Round 3: 10 clients
    /// - Average: 9.3 clients per round
    /// </remarks>
    public double AverageClientsPerRound { get; set; }

    /// <summary>
    /// Gets or sets the total communication cost in megabytes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The total amount of data transferred between clients and server
    /// throughout training. Important for understanding bandwidth requirements.
    ///
    /// For example:
    /// - Each model update: 10 MB
    /// - 10 clients per round
    /// - 50 rounds
    /// - Total: 10 MB × 10 × 50 × 2 (up and down) = 10,000 MB = 10 GB
    /// </remarks>
    public double TotalCommunicationMB { get; set; }

    /// <summary>
    /// Gets or sets the total privacy budget (epsilon) consumed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If differential privacy is used, this tracks how much privacy
    /// budget has been spent. Privacy budget is finite - once exhausted, no more privacy
    /// guarantees.
    ///
    /// For example:
    /// - ε per round: 0.1
    /// - 50 rounds completed
    /// - Total consumed: 5.0
    /// - If total budget is 10.0, have 5.0 remaining
    /// </remarks>
    public double TotalPrivacyBudgetConsumed { get; set; }

    /// <summary>
    /// Gets or sets the total privacy delta consumed (basic composition reporting).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Delta (δ) is the probability that the privacy guarantee fails.
    /// When using simple accounting, deltas add up across rounds.
    /// </remarks>
    public double TotalPrivacyDeltaConsumed { get; set; }

    /// <summary>
    /// Gets or sets which privacy accountant was used for reporting.
    /// </summary>
    public string PrivacyAccountantUsed { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the reported epsilon at the reported delta (when supported by the accountant).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Some accountants (like RDP) can report a tighter epsilon for a given delta.
    /// This value is intended for reporting; it does not change how noise was applied.
    /// </remarks>
    public double ReportedEpsilonAtDelta { get; set; }

    /// <summary>
    /// Gets or sets the delta used when reporting <see cref="ReportedEpsilonAtDelta"/>.
    /// </summary>
    public double ReportedDelta { get; set; }

    /// <summary>
    /// Gets or sets whether training converged before reaching maximum rounds.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Convergence means the model stopped improving significantly.
    /// If true, training ended early because the model reached a good solution.
    ///
    /// For example:
    /// - Max rounds: 100
    /// - Converged at round 50
    /// - Converged = true
    /// - Saved time by stopping early
    /// </remarks>
    public bool Converged { get; set; }

    /// <summary>
    /// Gets or sets the round at which convergence was detected.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Which round did the model stop improving significantly?
    ///
    /// Useful for:
    /// - Setting better MaxRounds for future training
    /// - Understanding training dynamics
    /// - Comparing different algorithms
    /// </remarks>
    public int ConvergenceRound { get; set; }

    /// <summary>
    /// Gets or sets the aggregation strategy used during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Records which aggregation algorithm was used (FedAvg, FedProx, etc.).
    /// Important for reproducibility and understanding results.
    /// </remarks>
    public string AggregationStrategyUsed { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the server-side federated optimizer used (FedOpt family).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Some federated learning variants apply an optimizer on the server after aggregation
    /// (for example, FedAvgM, FedAdam, FedYogi). If no server optimizer is used, this is "None".
    /// </remarks>
    public string ServerOptimizerUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the asynchronous federated learning mode used (if any).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If async FL is enabled, this records which mode was used (FedAsync or FedBuff).
    /// If training was synchronous, this is "None".
    /// </remarks>
    public string AsyncModeUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets whether compression was enabled for client updates.
    /// </summary>
    public bool CompressionEnabled { get; set; }

    /// <summary>
    /// Gets or sets the compression strategy used for client updates (if enabled).
    /// </summary>
    public string CompressionStrategyUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the heterogeneity correction method used (if any).
    /// </summary>
    public string HeterogeneityCorrectionUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets whether homomorphic encryption was enabled.
    /// </summary>
    public bool HomomorphicEncryptionEnabled { get; set; }

    /// <summary>
    /// Gets or sets the HE scheme used (CKKS or BFV).
    /// </summary>
    public string HomomorphicEncryptionSchemeUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the HE mode used (HEOnly or Hybrid).
    /// </summary>
    public string HomomorphicEncryptionModeUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the HE provider used.
    /// </summary>
    public string HomomorphicEncryptionProviderUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets whether personalization was enabled.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Personalization means some parameters can remain client-specific (or cluster-specific),
    /// improving accuracy on non-IID client data.
    /// </remarks>
    public bool PersonalizationEnabled { get; set; }

    /// <summary>
    /// Gets or sets the personalization strategy used (FedPer, FedRep, Ditto, pFedMe, Clustered).
    /// </summary>
    public string PersonalizationStrategyUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the fraction of parameters treated as personalized for head-split strategies.
    /// </summary>
    public double PersonalizedParameterFraction { get; set; }

    /// <summary>
    /// Gets or sets the number of post-aggregation local adaptation epochs (if configured).
    /// </summary>
    public int PersonalizationLocalAdaptationEpochs { get; set; }

    /// <summary>
    /// Gets or sets whether federated meta-learning was enabled.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Meta-learning in federated settings learns a global initialization that adapts quickly to each client.
    /// </remarks>
    public bool MetaLearningEnabled { get; set; }

    /// <summary>
    /// Gets or sets the federated meta-learning strategy used (Reptile/PerFedAvg/FedMAML).
    /// </summary>
    public string MetaLearningStrategyUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the meta learning rate used by the server update rule.
    /// </summary>
    public double MetaLearningRateUsed { get; set; }

    /// <summary>
    /// Gets or sets the inner (client) adaptation epochs used for meta-learning.
    /// </summary>
    public int MetaLearningInnerEpochsUsed { get; set; }

    /// <summary>
    /// Gets or sets whether differential privacy was enabled.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Records whether privacy mechanisms were active during training.
    /// Important for understanding any accuracy trade-offs.
    /// </remarks>
    public bool DifferentialPrivacyEnabled { get; set; }

    /// <summary>
    /// Gets or sets whether secure aggregation was enabled.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Records whether client updates were encrypted during aggregation.
    /// </remarks>
    public bool SecureAggregationEnabled { get; set; }

    /// <summary>
    /// Gets or sets which secure aggregation mode was used.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Different secure aggregation modes provide different drop-out tolerance.
    /// This value is "None" when secure aggregation is disabled.
    /// </remarks>
    public string SecureAggregationModeUsed { get; set; } = "None";

    /// <summary>
    /// Gets or sets the minimum uploader count used by secure aggregation (dropout-resilient modes).
    /// </summary>
    public int SecureAggregationMinimumUploaderCountUsed { get; set; }

    /// <summary>
    /// Gets or sets the reconstruction threshold used by secure aggregation (dropout-resilient modes).
    /// </summary>
    public int SecureAggregationReconstructionThresholdUsed { get; set; }

    /// <summary>
    /// Gets or sets additional notes or observations about the training run.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A freeform field for recording anything unusual or noteworthy
    /// that happened during training.
    ///
    /// For example:
    /// - "Client 5 dropped out after round 30"
    /// - "Convergence was slower than expected"
    /// - "High variance in client update quality"
    /// </remarks>
    public string Notes { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the per-round detailed metrics.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Detailed information about each individual round, including
    /// which clients participated, their individual losses, communication costs, etc.
    ///
    /// Useful for:
    /// - Detailed analysis of training dynamics
    /// - Identifying problematic clients
    /// - Understanding convergence patterns
    /// </remarks>
    public List<RoundMetadata> RoundMetrics { get; set; } = new List<RoundMetadata>();
}
