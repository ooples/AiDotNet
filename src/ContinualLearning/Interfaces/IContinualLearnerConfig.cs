namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Configuration interface for continual learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This interface defines the settings needed for continual learning,
/// such as learning rates, memory constraints, and regularization parameters.</para>
///
/// <para><b>Continual Learning</b> is the ability to learn new tasks sequentially without
/// forgetting previously learned knowledge. This is challenging because neural networks
/// tend to suffer from "catastrophic forgetting" - learning new information overwrites
/// old knowledge.</para>
///
/// <para><b>Common Strategies Include:</b>
/// <list type="bullet">
/// <item><description><b>EWC (Elastic Weight Consolidation):</b> Protects important weights from changing</description></item>
/// <item><description><b>LwF (Learning without Forgetting):</b> Uses knowledge distillation from teacher model</description></item>
/// <item><description><b>GEM (Gradient Episodic Memory):</b> Projects gradients to prevent forgetting</description></item>
/// <item><description><b>SI (Synaptic Intelligence):</b> Tracks online importance of weights</description></item>
/// <item><description><b>Experience Replay:</b> Stores and replays examples from previous tasks</description></item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Parisi et al. "Continual Lifelong Learning with Neural Networks: A Review" (2019)</para>
/// </remarks>
public interface IContinualLearnerConfig<T>
{
    #region Core Training Parameters

    /// <summary>
    /// Learning rate for training. Default: 0.001.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how much to update the model in each step.
    /// Lower values (e.g., 0.0001) mean slower but more stable learning.
    /// Higher values (e.g., 0.01) mean faster but potentially unstable learning.</para>
    /// </remarks>
    T LearningRate { get; }

    /// <summary>
    /// Number of training epochs per task. Default: 10.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> One epoch means the model sees all training data once.
    /// More epochs can improve learning but may also cause overfitting.</para>
    /// </remarks>
    int EpochsPerTask { get; }

    /// <summary>
    /// Batch size for training. Default: 32.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Number of samples processed together before updating weights.
    /// Larger batches are more stable but use more memory. Common values: 16, 32, 64, 128.</para>
    /// </remarks>
    int BatchSize { get; }

    #endregion

    #region Memory Parameters

    /// <summary>
    /// Maximum number of examples to store from previous tasks. Default: 1000.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many old examples to keep for experience replay.
    /// More examples reduce forgetting but use more memory.</para>
    /// </remarks>
    int MemorySize { get; }

    /// <summary>
    /// Number of samples per task to store in memory. Default: auto-calculated based on MemorySize.
    /// </summary>
    int SamplesPerTask { get; }

    /// <summary>
    /// Memory sampling strategy. Default: Reservoir.
    /// </summary>
    MemorySamplingStrategy MemoryStrategy { get; }

    /// <summary>
    /// Use prioritized experience replay based on sample importance. Default: false.
    /// </summary>
    bool UsePrioritizedReplay { get; }

    #endregion

    #region EWC-Specific Parameters

    /// <summary>
    /// EWC regularization strength (lambda). Default: 1000.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how strongly to protect important weights.
    /// Higher values (e.g., 5000) prevent more forgetting but may reduce plasticity.</para>
    /// <para><b>Reference:</b> Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (2017)</para>
    /// </remarks>
    T EwcLambda { get; }

    /// <summary>
    /// Number of samples for Fisher Information computation. Default: 200.
    /// </summary>
    int FisherSamples { get; }

    /// <summary>
    /// Use empirical Fisher (gradient squared) vs true Fisher. Default: true.
    /// </summary>
    bool UseEmpiricalFisher { get; }

    /// <summary>
    /// Normalize Fisher Information matrix. Default: true.
    /// </summary>
    bool NormalizeFisher { get; }

    #endregion

    #region Online-EWC Parameters

    /// <summary>
    /// Decay factor for online EWC (gamma). Default: 0.95.
    /// </summary>
    /// <remarks>
    /// <para>Controls how quickly old Fisher Information decays when learning new tasks.
    /// Higher values (closer to 1) maintain more memory of old tasks.</para>
    /// <para><b>Reference:</b> Schwarz et al. "Progress & Compress" (2018)</para>
    /// </remarks>
    T OnlineEwcGamma { get; }

    #endregion

    #region LwF-Specific Parameters

    /// <summary>
    /// Temperature for knowledge distillation softmax. Default: 2.0.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher temperature makes probability distributions softer,
    /// capturing more information about relative class similarities.</para>
    /// <para><b>Reference:</b> Li and Hoiem "Learning without Forgetting" (2017)</para>
    /// </remarks>
    T DistillationTemperature { get; }

    /// <summary>
    /// Weight for distillation loss vs task loss. Default: 1.0.
    /// </summary>
    T DistillationWeight { get; }

    /// <summary>
    /// Use soft targets from teacher model. Default: true.
    /// </summary>
    bool UseSoftTargets { get; }

    #endregion

    #region GEM-Specific Parameters

    /// <summary>
    /// Memory strength for GEM constraint. Default: 0.5.
    /// </summary>
    /// <remarks>
    /// <para>Controls how strictly to enforce the non-forgetting constraint.
    /// Higher values more strictly prevent any increase in loss on old tasks.</para>
    /// <para><b>Reference:</b> Lopez-Paz and Ranzato "Gradient Episodic Memory for Continual Learning" (2017)</para>
    /// </remarks>
    T GemMemoryStrength { get; }

    /// <summary>
    /// Margin for A-GEM (Averaged GEM). Default: 0.0.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Chaudhry et al. "Efficient Lifelong Learning with A-GEM" (2019)</para>
    /// </remarks>
    T AGemMargin { get; }

    /// <summary>
    /// Number of reference gradients for A-GEM. Default: 256.
    /// </summary>
    int AGemReferenceGradients { get; }

    #endregion

    #region SI-Specific Parameters

    /// <summary>
    /// SI regularization coefficient (c). Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Zenke et al. "Continual Learning Through Synaptic Intelligence" (2017)</para>
    /// </remarks>
    T SiC { get; }

    /// <summary>
    /// SI dampening factor (xi). Default: 0.1.
    /// </summary>
    T SiXi { get; }

    #endregion

    #region MAS-Specific Parameters

    /// <summary>
    /// MAS regularization coefficient (lambda). Default: 1.0.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Aljundi et al. "Memory Aware Synapses" (2018)</para>
    /// </remarks>
    T MasLambda { get; }

    #endregion

    #region PackNet-Specific Parameters

    /// <summary>
    /// Pruning ratio for PackNet. Default: 0.75.
    /// </summary>
    /// <remarks>
    /// <para>Fraction of weights to prune after each task, freeing capacity for new tasks.</para>
    /// <para><b>Reference:</b> Mallya and Lazebnik "PackNet" (2018)</para>
    /// </remarks>
    T PackNetPruneRatio { get; }

    /// <summary>
    /// Retrain epochs after pruning. Default: 5.
    /// </summary>
    int PackNetRetrainEpochs { get; }

    #endregion

    #region Progressive Neural Networks Parameters

    /// <summary>
    /// Use lateral connections in progressive networks. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Rusu et al. "Progressive Neural Networks" (2016)</para>
    /// </remarks>
    bool PnnUseLateralConnections { get; }

    /// <summary>
    /// Lateral connection scaling factor. Default: 1.0.
    /// </summary>
    T PnnLateralScaling { get; }

    #endregion

    #region iCaRL-Specific Parameters

    /// <summary>
    /// Number of exemplars per class for iCaRL. Default: 20.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Rebuffi et al. "iCaRL: Incremental Classifier and Representation Learning" (2017)</para>
    /// </remarks>
    int ICarlExemplarsPerClass { get; }

    /// <summary>
    /// Use herding for exemplar selection. Default: true.
    /// </summary>
    bool ICarlUseHerding { get; }

    #endregion

    #region BiC-Specific Parameters

    /// <summary>
    /// Validation set fraction for BiC bias correction. Default: 0.1.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Wu et al. "Large Scale Incremental Learning" (2019)</para>
    /// </remarks>
    T BiCValidationFraction { get; }

    #endregion

    #region HAT-Specific Parameters

    /// <summary>
    /// Sparsity coefficient for HAT. Default: 0.01.
    /// </summary>
    /// <remarks>
    /// <para><b>Reference:</b> Serra et al. "Overcoming Catastrophic Forgetting with Hard Attention to the Task" (2018)</para>
    /// </remarks>
    T HatSparsity { get; }

    /// <summary>
    /// Smax value for gradient-based attention. Default: 400.
    /// </summary>
    T HatSmax { get; }

    #endregion

    #region Evaluation Parameters

    /// <summary>
    /// Compute backward transfer metric. Default: true.
    /// </summary>
    bool ComputeBackwardTransfer { get; }

    /// <summary>
    /// Compute forward transfer metric. Default: true.
    /// </summary>
    bool ComputeForwardTransfer { get; }

    /// <summary>
    /// Evaluation frequency (every N epochs). Default: 1.
    /// </summary>
    int EvaluationFrequency { get; }

    #endregion

    #region Advanced Parameters

    /// <summary>
    /// Random seed for reproducibility. Default: null (random).
    /// </summary>
    int? RandomSeed { get; }

    /// <summary>
    /// Maximum number of tasks to support. Default: 100.
    /// </summary>
    int MaxTasks { get; }

    /// <summary>
    /// Enable gradient clipping. Default: false.
    /// </summary>
    bool UseGradientClipping { get; }

    /// <summary>
    /// Gradient clipping max norm. Default: 1.0.
    /// </summary>
    T GradientClipNorm { get; }

    /// <summary>
    /// Enable weight decay regularization. Default: false.
    /// </summary>
    bool UseWeightDecay { get; }

    /// <summary>
    /// Weight decay coefficient. Default: 0.0001.
    /// </summary>
    T WeightDecay { get; }

    #endregion

    /// <summary>
    /// Validates the configuration.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    bool IsValid();
}
