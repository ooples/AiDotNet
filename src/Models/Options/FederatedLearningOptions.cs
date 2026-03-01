namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for federated learning training.
/// </summary>
/// <remarks>
/// This class contains all the configurable parameters needed to set up and run a federated learning system.
///
/// <b>For Beginners:</b> Options are like the settings panel for federated learning.
/// Just as you configure settings for a video game (difficulty, graphics quality, etc.),
/// these options let you configure how federated learning should work.
///
/// Key configuration areas:
/// - Client Management: How many clients, how to select them
/// - Training: Learning rates, epochs, batch sizes
/// - Privacy: Differential privacy parameters
/// - Communication: How often to aggregate, compression settings
/// - Convergence: When to stop training
///
/// For example, a typical configuration might be:
/// - 100 total clients (e.g., hospitals)
/// - Select 10 clients per round (10% participation)
/// - Each client trains for 5 local epochs
/// - Use privacy budget ε=1.0, δ=1e-5
/// - Run for maximum 100 rounds or until convergence
/// </remarks>
public class FederatedLearningOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the total number of clients participating in federated learning.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the total pool of participants available for training.
    /// In each round, a subset of these clients may be selected.
    ///
    /// For example:
    /// - Mobile keyboard app: Millions of phones
    /// - Healthcare: 50 hospitals
    /// - Financial: 100 bank branches
    /// </remarks>
    public int NumberOfClients { get; set; } = 10;

    /// <summary>
    /// Gets or sets the fraction of clients to select for each training round (0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Not all clients participate in every round. This setting controls
    /// what percentage of clients are active in each round.
    ///
    /// Common values:
    /// - 1.0: All clients participate (small deployments)
    /// - 0.1: 10% participate (large deployments, reduces communication)
    /// - 0.01: 1% participate (massive deployments like mobile devices)
    ///
    /// For example, with 1000 clients and fraction 0.1:
    /// - Each round randomly selects 100 clients
    /// - Reduces server load and communication costs
    /// - Still converges if enough clients are selected
    /// </remarks>
    public double ClientSelectionFraction { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets client selection options (strategy and related parameters).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If this is null, the trainer uses uniform random sampling with
    /// <see cref="ClientSelectionFraction"/>.
    /// </remarks>
    public ClientSelectionOptions? ClientSelection { get; set; } = new();

    /// <summary>
    /// Gets or sets the number of local training epochs each client performs per round.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An epoch is one complete pass through the client's local dataset.
    /// More epochs mean more local training but also more computation time.
    ///
    /// Trade-offs:
    /// - More epochs (5-10): Better local adaptation, slower rounds
    /// - Fewer epochs (1-2): Faster rounds, more communication needed
    ///
    /// For example:
    /// - Client has 1000 samples
    /// - LocalEpochs = 5
    /// - Client processes all 1000 samples 5 times before sending update
    /// </remarks>
    public int LocalEpochs { get; set; } = 5;

    /// <summary>
    /// Gets or sets the maximum number of federated learning rounds to execute.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A round is one complete cycle where clients train and the server
    /// aggregates updates. This sets the maximum number of such cycles.
    ///
    /// Typical values:
    /// - Quick experiments: 10-50 rounds
    /// - Production training: 100-1000 rounds
    /// - Large scale: 1000+ rounds
    ///
    /// Training may stop early if convergence criteria are met.
    /// </remarks>
    public int MaxRounds { get; set; } = 100;

    /// <summary>
    /// Gets or sets the learning rate for local client training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Learning rate controls how big of a step the model takes when
    /// learning from data. Too large and learning is unstable; too small and learning is slow.
    ///
    /// Common values:
    /// - Deep learning: 0.001 - 0.01
    /// - Traditional ML: 0.01 - 0.1
    ///
    /// For example:
    /// - LearningRate = 0.01 means adjust weights by 1% of the gradient
    /// - Higher values = faster learning but less stability
    /// - Lower values = slower but more stable learning
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the batch size for local training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Instead of processing all data at once, we process it in smaller
    /// batches. Batch size is how many samples to process before updating the model.
    ///
    /// Trade-offs:
    /// - Larger batches (64-512): More stable gradients, requires more memory
    /// - Smaller batches (8-32): Less memory, more noise in updates
    ///
    /// For example:
    /// - Client has 1000 samples, BatchSize = 32
    /// - Data is split into approximately 32 batches of 31-32 samples each
    /// - Model is updated after processing each batch
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets whether to use differential privacy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Differential privacy adds mathematical noise to protect individual
    /// data points from being identified in the model updates.
    ///
    /// When enabled:
    /// - Privacy guarantees are provided
    /// - Some accuracy may be sacrificed
    /// - Individual contributions are hidden
    ///
    /// Use cases requiring privacy:
    /// - Healthcare data
    /// - Financial records
    /// - Personal communications
    /// </remarks>
    public bool UseDifferentialPrivacy { get; set; } = false;

    /// <summary>
    /// Gets or sets where differential privacy is applied (local, central, or both).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This controls *when* noise is added:
    /// - Local: noise is added on each device before it sends updates.
    /// - Central: noise is added after the server combines updates.
    /// - Both: adds noise in both places for extra protection.
    /// </remarks>
    public DifferentialPrivacyMode DifferentialPrivacyMode { get; set; } = DifferentialPrivacyMode.Central;

    /// <summary>
    /// Gets or sets the clipping norm used by differential privacy mechanisms.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Clipping limits how big an update can be before adding noise.
    /// This is required to make privacy guarantees meaningful.
    /// </remarks>
    public double DifferentialPrivacyClipNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets which privacy accountant to use for reporting privacy spend.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A privacy accountant tracks how much privacy budget has been spent over rounds.
    /// - Basic: adds up epsilon and delta across rounds.
    /// - Rdp: uses a Rényi DP accountant for tighter reporting (recommended default when DP is enabled).
    /// </remarks>
    public FederatedPrivacyAccountant PrivacyAccountant { get; set; } = FederatedPrivacyAccountant.Rdp;

    /// <summary>
    /// Gets or sets the epsilon (ε) parameter for differential privacy (privacy budget).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Epsilon controls the privacy-utility tradeoff. Lower values mean
    /// stronger privacy but potentially less accurate models.
    ///
    /// Common values:
    /// - ε = 0.1: Very strong privacy, significant accuracy loss
    /// - ε = 1.0: Strong privacy, moderate accuracy loss (recommended)
    /// - ε = 10.0: Weak privacy, minimal accuracy loss
    ///
    /// For example:
    /// - With ε = 1.0, an adversary cannot distinguish whether any specific individual's
    ///   data was used in training (within factor e^1 ≈ 2.7)
    /// </remarks>
    public double PrivacyEpsilon { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the delta (δ) parameter for differential privacy (failure probability).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Delta is the probability that the privacy guarantee fails.
    /// It should be very small, typically much less than 1/number_of_data_points.
    ///
    /// Common values:
    /// - δ = 1e-5 (0.00001): Standard choice
    /// - δ = 1e-6: Stronger guarantee
    ///
    /// For example:
    /// - δ = 1e-5 means there's a 0.001% chance privacy is compromised
    /// - Should be smaller than 1/total_number_of_samples across all clients
    /// </remarks>
    public double PrivacyDelta { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use secure aggregation.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Secure aggregation encrypts client updates so that the server
    /// can only see the aggregated result, not individual contributions.
    ///
    /// Benefits:
    /// - Server cannot see individual client updates
    /// - Protects against honest-but-curious server
    /// - Only the final aggregated model is visible
    ///
    /// For example:
    /// - Without: Server sees each hospital's model update
    /// - With: Server only sees combined update from all hospitals
    /// - No single hospital's contribution is visible
    /// </remarks>
    public bool UseSecureAggregation { get; set; } = false;

    /// <summary>
    /// Gets or sets secure aggregation configuration options.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If you enable secure aggregation, you can optionally use this object to choose
    /// a dropout-resilient protocol mode and tune its thresholds. If this is null, industry-standard defaults
    /// are used based on your selected mode.
    /// </remarks>
    public SecureAggregationOptions? SecureAggregation { get; set; } = new();

    /// <summary>
    /// Gets or sets the convergence threshold for early stopping.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Training stops early if improvement between rounds falls below
    /// this threshold, indicating the model has converged (stopped improving significantly).
    ///
    /// For example:
    /// - ConvergenceThreshold = 0.001
    /// - If loss improves by less than 0.001 for several consecutive rounds, stop training
    /// - Saves time and resources by avoiding unnecessary rounds
    /// </remarks>
    public double ConvergenceThreshold { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the minimum number of rounds to train before checking convergence.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Don't check for convergence too early. Wait at least this many
    /// rounds before considering early stopping.
    ///
    /// This prevents:
    /// - Stopping too early due to initial volatility
    /// - Missing later improvements
    ///
    /// Typical value: 10-20 rounds
    /// </remarks>
    public int MinRoundsBeforeConvergence { get; set; } = 10;

    /// <summary>
    /// Gets or sets the aggregation strategy to use.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how client updates are combined.
    ///
    /// Available strategies include FedAvg, FedProx, FedBN, and multiple robust strategies (Median, TrimmedMean, Krum, Bulyan, etc.).
    ///
    /// Different strategies work better for different scenarios.
    /// </remarks>
    public FederatedAggregationStrategy AggregationStrategy { get; set; } = FederatedAggregationStrategy.FedAvg;

    /// <summary>
    /// Gets or sets options for robust aggregation strategies (Median, TrimmedMean, Krum, MultiKrum, Bulyan).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Most users can ignore this unless they select a robust aggregation strategy.
    /// If not provided, industry-standard defaults are used.
    /// </remarks>
    public RobustAggregationOptions? RobustAggregation { get; set; } = new();

    /// <summary>
    /// Gets or sets server-side federated optimization options (FedOpt family).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> FedOpt applies an optimizer step on the server after aggregation.
    /// If this is null or set to <see cref="FederatedServerOptimizer.None"/>, the server uses the aggregated parameters directly (FedAvg-style).
    /// </remarks>
    public FederatedServerOptimizerOptions? ServerOptimizer { get; set; } = new();

    /// <summary>
    /// Gets or sets asynchronous federated learning options (FedAsync / FedBuff).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Async FL can reduce waiting on slow clients by applying updates as they arrive
    /// (FedAsync) or in small buffers (FedBuff). If not set or set to <see cref="FederatedAsyncMode.None"/>, training is synchronous.
    /// </remarks>
    public AsyncFederatedLearningOptions? AsyncFederatedLearning { get; set; } = new();

    /// <summary>
    /// Gets or sets federated heterogeneity correction options (SCAFFOLD / FedNova / FedDyn).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These methods help reduce client drift on non-IID data by transforming
    /// client updates before aggregation. If not set or set to <see cref="FederatedHeterogeneityCorrection.None"/>, no correction is applied.
    /// </remarks>
    public FederatedHeterogeneityCorrectionOptions? HeterogeneityCorrection { get; set; } = new();

    /// <summary>
    /// Gets or sets homomorphic encryption options for federated aggregation (CKKS/BFV).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If enabled, the server aggregates encrypted updates without seeing individual updates in plaintext.
    /// </remarks>
    public HomomorphicEncryptionOptions? HomomorphicEncryption { get; set; } = new();

    /// <summary>
    /// Gets or sets the proximal term coefficient for FedProx algorithm.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> FedProx adds a penalty to prevent client models from
    /// deviating too much from the global model. This parameter controls the penalty strength.
    ///
    /// Common values:
    /// - 0.0: No proximal term (equivalent to FedAvg)
    /// - 0.01 - 0.1: Mild constraint
    /// - 1.0+: Strong constraint
    ///
    /// Use FedProx when:
    /// - Clients have very different data distributions
    /// - Some clients are much slower than others
    /// - You want more stable convergence
    /// </remarks>
    public double ProximalMu { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to enable personalization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Personalization allows each client to maintain some client-specific
    /// model parameters while sharing common parameters with other clients.
    ///
    /// Benefits:
    /// - Better performance on local data
    /// - Handles non-IID data (data that varies across clients)
    /// - Combines benefits of global and local models
    ///
    /// For example:
    /// - Global layers: Learn general patterns from all clients
    /// - Personalized layers: Adapt to each client's specific data
    /// </remarks>
    public bool EnablePersonalization { get; set; } = false;

    /// <summary>
    /// Gets or sets the fraction of model layers to keep personalized (not aggregated).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When personalization is enabled, this determines what fraction
    /// of the model remains client-specific vs. shared globally.
    ///
    /// For example:
    /// - PersonalizationLayerFraction = 0.2
    /// - Last 20% of model layers stay personalized
    /// - First 80% are aggregated globally
    ///
    /// Typical use:
    /// - Output layers personalized, feature extractors shared
    /// </remarks>
    public double PersonalizationLayerFraction { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets personalization options (preferred configuration for FedPer/FedRep/Ditto/pFedMe/clustered).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If this is set, it overrides the legacy
    /// <see cref="EnablePersonalization"/> / <see cref="PersonalizationLayerFraction"/> settings.
    /// </remarks>
    public FederatedPersonalizationOptions? Personalization { get; set; } = null;

    /// <summary>
    /// Gets or sets federated meta-learning options (Per-FedAvg / FedMAML / Reptile-style).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Federated meta-learning learns a global initialization that adapts quickly to each client.
    /// If set and enabled, the trainer uses a meta-update rule instead of standard FedAvg-style aggregation.
    /// </remarks>
    public FederatedMetaLearningOptions? MetaLearning { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to use gradient compression to reduce communication costs.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Compression reduces the size of model updates sent between
    /// clients and server, saving bandwidth and time.
    ///
    /// Techniques:
    /// - Quantization: Use fewer bits per parameter
    /// - Sparsification: Send only top-k largest updates
    /// - Sketching: Use randomized compression
    ///
    /// Trade-off:
    /// - Reduces communication by 10-100x
    /// - May slightly slow convergence
    /// </remarks>
    public bool UseCompression { get; set; } = false;

    /// <summary>
    /// Gets or sets federated compression options.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the preferred way to configure compression. If null,
    /// the legacy <see cref="UseCompression"/> / <see cref="CompressionRatio"/> properties are used.
    /// </remarks>
    public FederatedCompressionOptions? Compression { get; set; } = null;

    /// <summary>
    /// Gets or sets the compression ratio (0.0 to 1.0) if compression is enabled.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Controls how much to compress. Lower values mean more compression
    /// but potentially more accuracy loss.
    ///
    /// For example:
    /// - 0.01: Keep top 1% of gradients (99% compression)
    /// - 0.1: Keep top 10% of gradients (90% compression)
    /// - 1.0: No compression
    /// </remarks>
    public double CompressionRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets Trusted Execution Environment options for hardware-backed secure aggregation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When set, the federated learning server performs aggregation inside a
    /// hardware-protected enclave (Intel SGX/TDX, AMD SEV-SNP, ARM CCA). Clients can verify the
    /// enclave via remote attestation before sending their updates. This is 10-100x faster than
    /// homomorphic encryption while providing hardware-level isolation.</para>
    ///
    /// <para>Set to null (default) to use standard in-memory aggregation without TEE.</para>
    /// </remarks>
    public TeeOptions? TrustedExecutionEnvironment { get; set; } = new();

    /// <summary>
    /// Gets or sets federated graph learning options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When set, the FL system handles graph-structured data where each client
    /// holds a subgraph of a larger graph. Includes cross-client edge discovery (PSI), pseudo-node
    /// strategies for missing neighbors, and graph-aware aggregation weighting.</para>
    ///
    /// <para>Set to null (default) for standard non-graph federated learning.</para>
    /// </remarks>
    public FederatedGraphOptions? GraphLearning { get; set; } = new();

    /// <summary>
    /// Gets or sets federated unlearning options (right to be forgotten).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When a client exercises their GDPR right to be forgotten, the system
    /// must remove their contribution from the trained model. These options configure how unlearning
    /// is performed, including method selection (exact retraining, gradient ascent, influence functions,
    /// or diffusive noise) and verification parameters.</para>
    ///
    /// <para>Set to null (default) if unlearning is not needed.</para>
    /// </remarks>
    public FederatedUnlearningOptions? Unlearning { get; set; } = new();

    /// <summary>
    /// Gets or sets fairness constraint options for equitable model performance across client groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When set, the FL system enforces fairness constraints to prevent
    /// the global model from favoring majority client groups at the expense of underrepresented ones.
    /// Supports demographic parity, equalized odds, equal opportunity, and minimax fairness.</para>
    ///
    /// <para>Set to null (default) for standard aggregation without fairness constraints.</para>
    /// </remarks>
    public FederatedFairnessOptions? Fairness { get; set; } = new();

    /// <summary>
    /// Gets or sets client contribution evaluation options for measuring each client's value.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When set, the system evaluates how much each client contributed
    /// to the global model using Shapley values, Data Shapley, or prototypical representations.
    /// This enables free-rider detection, fair compensation, and quality monitoring.</para>
    ///
    /// <para>Set to null (default) to skip contribution evaluation.</para>
    /// </remarks>
    public ContributionEvaluationOptions? ContributionEvaluation { get; set; } = new();

    /// <summary>
    /// Gets or sets federated concept drift detection and adaptation options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In production FL, client data distributions change over time
    /// (seasonal patterns, evolving fraud, shifting user behavior). Drift detection monitors
    /// each client's training behavior and adapts aggregation when changes are detected.</para>
    ///
    /// <para>Set to null (default) to disable drift detection.</para>
    /// </remarks>
    public FederatedDriftOptions? DriftDetection { get; set; } = new();

    /// <summary>
    /// Gets or sets a random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Random seed makes randomness reproducible. Using the same
    /// seed will produce the same random client selections, initializations, etc.
    ///
    /// Benefits:
    /// - Reproducible experiments
    /// - Easier debugging
    /// - Fair comparison between methods
    ///
    /// Set to null for truly random behavior.
    /// </remarks>
    public int? RandomSeed { get; set; } = null;

    /// <summary>
    /// Gets or sets the federated learning mode (horizontal or vertical).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose the paradigm that matches how data is distributed:</para>
    /// <list type="bullet">
    /// <item><description><b>Horizontal (default):</b> Each client has the same features for different samples
    /// (e.g., multiple hospitals with patient records sharing the same columns).</description></item>
    /// <item><description><b>Vertical:</b> Each client has different features for the same entities
    /// (e.g., a bank has income data, a hospital has medical data for the same people).
    /// When set to Vertical, you must also configure <see cref="VerticalLearning"/>.</description></item>
    /// </list>
    /// </remarks>
    public FederatedLearningMode Mode { get; set; } = FederatedLearningMode.Horizontal;

    /// <summary>
    /// Gets or sets vertical federated learning options (split learning, entity alignment, etc.).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only used when <see cref="Mode"/> is set to
    /// <see cref="FederatedLearningMode.Vertical"/>. Configures the split neural network,
    /// entity alignment via PSI, missing feature handling, and label privacy protection.</para>
    ///
    /// <para>Set to null (default) when using horizontal federated learning.</para>
    /// </remarks>
    public VerticalFederatedLearningOptions? VerticalLearning { get; set; } = new();

    /// <summary>
    /// Gets or sets Private Set Intersection options for entity alignment.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> PSI allows parties to discover which entities they share
    /// without revealing non-shared data. This is the mandatory first step for vertical FL
    /// and can also be used in graph FL for cross-client edge discovery.</para>
    ///
    /// <para>Supports multiple protocols: Diffie-Hellman, Oblivious Transfer, Bloom Filter,
    /// Circuit-Based, Fuzzy Matching, and Multi-Party PSI.</para>
    ///
    /// <para>Set to null (default) if PSI is not needed (e.g., pure horizontal FL with known clients).</para>
    /// </remarks>
    public PsiOptions? PrivateSetIntersection { get; set; } = new();

    /// <summary>
    /// Gets or sets Multi-Party Computation options for secure operations beyond summation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard secure aggregation only computes sums privately. MPC
    /// enables arbitrary secure computations including comparisons, clipping, top-k selection,
    /// and gradient transformations using secret sharing and garbled circuits.</para>
    ///
    /// <para>Use MPC when you need secure gradient clipping, median aggregation,
    /// or Krum/Bulyan distance computation on encrypted data.</para>
    ///
    /// <para>Set to null (default) for standard plaintext or basic secure aggregation.</para>
    /// </remarks>
    public MpcOptions? MultiPartyComputation { get; set; } = new();

    /// <summary>
    /// Gets or sets verification options for zero-knowledge proof-based update validation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Verification allows the server to cryptographically verify
    /// that client updates are well-formed (e.g., gradients are within declared bounds, local
    /// training was performed correctly) without clients revealing their private data.</para>
    ///
    /// <para>Complements Byzantine-robust aggregators (Krum, Bulyan) which detect statistical
    /// anomalies. ZK proofs provide cryptographic guarantees of computational correctness.</para>
    ///
    /// <para>Set to null (default) to skip verification.</para>
    /// </remarks>
    public VerificationOptions? Verification { get; set; } = new();

    /// <summary>
    /// Gets or sets advanced communication compression options (PowerSGD, sketching, error feedback).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Advanced compression techniques go beyond basic TopK/RandomK
    /// sparsification. PowerSGD uses low-rank approximation, gradient sketching uses Count Sketch,
    /// error feedback accumulates compression errors across rounds, and adaptive compression
    /// adjusts ratios per client based on bandwidth.</para>
    ///
    /// <para>These methods can achieve 100-1000x compression while maintaining convergence.
    /// Set to null (default) to use basic compression via <see cref="Compression"/>.</para>
    /// </remarks>
    public AdvancedCompressionOptions? AdvancedCompression { get; set; } = new();

    /// <summary>
    /// Gets or sets federated knowledge distillation options (FedMD, FedDF, FedGEN).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Knowledge distillation enables model-heterogeneous FL where different
    /// clients can use different model architectures. Instead of sharing model parameters directly,
    /// clients share soft predictions (logits) and learn from each other's knowledge.</para>
    ///
    /// <para>Set to null (default) for standard parameter aggregation.</para>
    /// </remarks>
    public FederatedDistillationOptions? Distillation { get; set; } = new();

    /// <summary>
    /// Gets or sets federated adapter options for parameter-efficient fine-tuning (LoRA, prompt tuning).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When fine-tuning large foundation models in FL, sending full model
    /// parameters is prohibitively expensive. Adapters only train and communicate a tiny fraction
    /// (&lt;1%) of parameters, making federated LLM fine-tuning practical.</para>
    ///
    /// <para>Set to null (default) for standard full-model aggregation.</para>
    /// </remarks>
    public FederatedAdapterOptions? Adapters { get; set; } = new();

    /// <summary>
    /// Gets or sets decentralized (serverless) federated learning options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard FL uses a central server. Decentralized FL removes
    /// the server — nodes communicate directly using gossip or ring topologies. This eliminates
    /// single points of failure and can be more robust for edge/IoT deployments.</para>
    ///
    /// <para>Set to null (default) for standard server-based FL.</para>
    /// </remarks>
    public DecentralizedFederatedOptions? Decentralized { get; set; } = new();

    /// <summary>
    /// Gets or sets federated continual learning options for preventing catastrophic forgetting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the FL system learns new tasks over time, it can forget
    /// previously learned knowledge. Federated continual learning strategies (EWC, orthogonal
    /// projection) identify and protect important model parameters across training rounds.</para>
    ///
    /// <para>Set to null (default) to disable continual learning protection.</para>
    /// </remarks>
    public FederatedContinualLearningOptions? ContinualLearning { get; set; } = new();

    /// <summary>
    /// Gets or sets backdoor attack detection and defense options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Backdoor attacks are stealthy — the model works correctly on clean
    /// data but misbehaves when a trigger pattern is present. These detectors (Neural Cleanse,
    /// Direction Alignment Inspector) analyze client updates for signs of backdoor injection
    /// and filter suspicious updates before aggregation.</para>
    ///
    /// <para>Set to null (default) to skip backdoor detection.</para>
    /// </remarks>
    public BackdoorDefenseOptions? BackdoorDefense { get; set; } = new();
}
