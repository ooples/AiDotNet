namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the continual learning strategy to use for preventing catastrophic forgetting.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Continual learning strategies help neural networks learn new tasks
/// without forgetting previously learned ones. Different strategies use different approaches to
/// balance learning new knowledge while preserving old knowledge.</para>
/// </remarks>
public enum ContinualLearningStrategyType
{
    /// <summary>
    /// Elastic Weight Consolidation - penalizes changes to important weights using Fisher information.
    /// </summary>
    EWC,

    /// <summary>
    /// Online EWC - memory-efficient variant that maintains running Fisher estimate.
    /// </summary>
    OnlineEWC,

    /// <summary>
    /// Synaptic Intelligence - tracks weight importance online during training.
    /// </summary>
    SynapticIntelligence,

    /// <summary>
    /// Memory Aware Synapses - unsupervised importance estimation using output sensitivity.
    /// </summary>
    MAS,

    /// <summary>
    /// Learning without Forgetting - uses knowledge distillation to preserve old predictions.
    /// </summary>
    LearningWithoutForgetting,

    /// <summary>
    /// Gradient Episodic Memory - constrains gradients to not hurt stored examples.
    /// </summary>
    GEM,

    /// <summary>
    /// Averaged GEM - efficient variant using single averaged constraint.
    /// </summary>
    AveragedGEM,

    /// <summary>
    /// Experience Replay - stores and replays examples from previous tasks.
    /// </summary>
    ExperienceReplay,

    /// <summary>
    /// Generative Replay - uses generative model to create pseudo-examples for rehearsal.
    /// </summary>
    GenerativeReplay,

    /// <summary>
    /// PackNet - isolates parameters per task through pruning and freezing.
    /// </summary>
    PackNet,

    /// <summary>
    /// Progressive Neural Networks - adds new columns with lateral connections for each task.
    /// </summary>
    ProgressiveNeuralNetworks,

    /// <summary>
    /// Variational Continual Learning - Bayesian approach using posterior as prior.
    /// </summary>
    VCL
}

/// <summary>
/// Specifies the buffer management strategy for Experience Replay.
/// </summary>
public enum ReplayBufferStrategy
{
    /// <summary>
    /// Reservoir sampling - uniform random replacement ensuring equal probability.
    /// </summary>
    Reservoir,

    /// <summary>
    /// Ring buffer - FIFO queue where oldest samples are removed first.
    /// </summary>
    Ring,

    /// <summary>
    /// Class-balanced sampling - maintains equal samples per class.
    /// </summary>
    ClassBalanced
}

/// <summary>
/// Represents configuration options for continual learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Continual learning (also called lifelong learning) allows models
/// to learn from a sequence of tasks without forgetting what was learned before. This is important
/// because standard neural networks suffer from "catastrophic forgetting" - when trained on new
/// data, they tend to forget previously learned patterns.</para>
///
/// <para><b>Typical Usage:</b></para>
/// <code>
/// var options = new ContinualLearningOptions
/// {
///     Strategy = ContinualLearningStrategyType.EWC,
///     Lambda = 400.0
/// };
/// </code>
///
/// <para><b>How to Choose a Strategy:</b></para>
/// <list type="bullet">
/// <item><description><b>EWC/OnlineEWC:</b> Good baseline for regularization-based continual learning.</description></item>
/// <item><description><b>SynapticIntelligence/MAS:</b> Online alternatives that don't need data storage.</description></item>
/// <item><description><b>LearningWithoutForgetting:</b> Useful when previous task data isn't available.</description></item>
/// <item><description><b>GEM/A-GEM:</b> Strong constraints that guarantee no forgetting on stored samples.</description></item>
/// <item><description><b>ExperienceReplay:</b> Simple and effective; good when memory storage is acceptable.</description></item>
/// <item><description><b>GenerativeReplay:</b> Privacy-preserving alternative to experience replay.</description></item>
/// <item><description><b>PackNet:</b> Zero forgetting through parameter isolation; limited by network capacity.</description></item>
/// <item><description><b>ProgressiveNeuralNetworks:</b> Zero forgetting with knowledge transfer; grows with tasks.</description></item>
/// <item><description><b>VCL:</b> Principled Bayesian approach with uncertainty quantification.</description></item>
/// </list>
/// </remarks>
public class ContinualLearningOptions
{
    /// <summary>
    /// Gets or sets the continual learning strategy to use.
    /// </summary>
    /// <remarks>
    /// The strategy determines how the model balances learning new tasks with preserving old knowledge.
    /// Default is EWC (Elastic Weight Consolidation), which is a widely-used and effective baseline.
    /// </remarks>
    public ContinualLearningStrategyType Strategy { get; set; } = ContinualLearningStrategyType.EWC;

    /// <summary>
    /// Gets or sets the regularization strength (lambda) for weight consolidation strategies.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lambda controls the trade-off between learning new tasks and
    /// remembering old ones:</para>
    /// <list type="bullet">
    /// <item><description>Higher lambda: Strong protection of old knowledge, but may struggle to learn new tasks.</description></item>
    /// <item><description>Lower lambda: Easier to learn new tasks, but more forgetting of old tasks.</description></item>
    /// </list>
    /// <para>Typical values: 100-5000 for EWC/OnlineEWC, 0.1-10 for other strategies.</para>
    /// </remarks>
    public double Lambda { get; set; } = 400.0;

    /// <summary>
    /// Gets or sets the decay factor for Online EWC.
    /// </summary>
    /// <remarks>
    /// <para>Controls how quickly older task importance decays in Online EWC:</para>
    /// <list type="bullet">
    /// <item><description>Gamma = 1.0: Equal weight to all tasks.</description></item>
    /// <item><description>Gamma &lt; 1.0: More emphasis on recent tasks.</description></item>
    /// </list>
    /// </remarks>
    public double Gamma { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the damping constant for Synaptic Intelligence.
    /// </summary>
    /// <remarks>
    /// Small constant to prevent division by zero when weights don't change much.
    /// Typical values: 0.001 to 0.1.
    /// </remarks>
    public double Damping { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the temperature for knowledge distillation in LearningWithoutForgetting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls how "soft" the probability distribution becomes:</para>
    /// <list type="bullet">
    /// <item><description>T = 1: Normal softmax (sharp, peaked distribution).</description></item>
    /// <item><description>T = 2-5: Softer distribution that reveals class relationships.</description></item>
    /// <item><description>T &gt; 5: Very soft, almost uniform distribution.</description></item>
    /// </list>
    /// <para>Typical values for LwF: 2-4.</para>
    /// </remarks>
    public double Temperature { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the margin for gradient constraints in GEM.
    /// </summary>
    /// <remarks>
    /// Controls how much "safety buffer" to keep for gradient constraints.
    /// Higher margin means gradients are more constrained, reducing forgetting but potentially slowing learning.
    /// </remarks>
    public double Margin { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum number of samples to store per task in memory-based strategies.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many examples to remember from each task.
    /// More examples = better protection but higher memory cost.</para>
    /// <para>Used by: GEM, A-GEM, ExperienceReplay.</para>
    /// </remarks>
    public int MemorySize { get; set; } = 256;

    /// <summary>
    /// Gets or sets the batch size to sample from memory for reference gradients in A-GEM.
    /// </summary>
    /// <remarks>
    /// Controls how many stored samples are used to compute the reference gradient.
    /// Larger values give more accurate constraints but increase computation.
    /// </remarks>
    public int SampleSize { get; set; } = 64;

    /// <summary>
    /// Gets or sets the maximum buffer size for Experience Replay.
    /// </summary>
    /// <remarks>
    /// The total number of samples that can be stored across all tasks.
    /// Typical values: 500-5000 depending on available memory.
    /// </remarks>
    public int MaxBufferSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the ratio of replay samples to new samples in replay-based strategies.
    /// </summary>
    /// <remarks>
    /// <para>Controls the mix of old and new data during training:</para>
    /// <list type="bullet">
    /// <item><description>0.5 means 50% replay, 50% new data.</description></item>
    /// <item><description>Higher values emphasize remembering over learning.</description></item>
    /// </list>
    /// <para>Used by: ExperienceReplay, GenerativeReplay.</para>
    /// </remarks>
    public double ReplayRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the buffer management strategy for Experience Replay.
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description><b>Reservoir:</b> Random replacement, good for diverse sampling.</description></item>
    /// <item><description><b>Ring:</b> FIFO, emphasizes recent data.</description></item>
    /// <item><description><b>ClassBalanced:</b> Maintains equal representation per class.</description></item>
    /// </list>
    /// </remarks>
    public ReplayBufferStrategy BufferStrategy { get; set; } = ReplayBufferStrategy.Reservoir;

    /// <summary>
    /// Gets or sets the batch size for generating pseudo-examples in Generative Replay.
    /// </summary>
    /// <remarks>
    /// Number of synthetic samples to generate per batch for rehearsal.
    /// Typical values: 16-64.
    /// </remarks>
    public int ReplayBatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the pruning ratio for PackNet.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Determines how much of the remaining network capacity to use per task.</para>
    /// <list type="bullet">
    /// <item><description>0.5 means 50% of free weights are pruned after each task.</description></item>
    /// <item><description>With ratio 0.5, you can fit approximately log2(1/ratio) tasks.</description></item>
    /// </list>
    /// <para>Must be between 0 and 1 (exclusive).</para>
    /// </remarks>
    public double PruningRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use lateral connections in Progressive Neural Networks.
    /// </summary>
    /// <remarks>
    /// Lateral connections allow new task columns to receive input from previous columns,
    /// enabling knowledge transfer. Disable for simple multi-head training.
    /// </remarks>
    public bool UseLateralConnections { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial log-variance for weight distributions in VCL.
    /// </summary>
    /// <remarks>
    /// Controls initial uncertainty in weight distributions.
    /// Value of -3.0 means standard deviation of approximately 0.22.
    /// </remarks>
    public double InitialLogVariance { get; set; } = -3.0;

    /// <summary>
    /// Gets or sets the number of Monte Carlo samples for VCL.
    /// </summary>
    /// <remarks>
    /// More samples provide better approximation of model uncertainty but increase computation.
    /// Typical values: 5-20.
    /// </remarks>
    public int NumMcSamples { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for Monte Carlo Dropout in VCL.
    /// </summary>
    /// <remarks>
    /// Controls the variation between forward passes for uncertainty estimation.
    /// Typical values: 0.1-0.5.
    /// </remarks>
    public double DropoutRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// Setting a seed ensures reproducible behavior across runs.
    /// Leave as null for random behavior each time.
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize importance scores.
    /// </summary>
    /// <remarks>
    /// Normalization can help when combining importance from different tasks
    /// or when importance values have very different scales.
    /// </remarks>
    public bool NormalizeImportance { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of samples to use for Fisher Information estimation.
    /// </summary>
    /// <remarks>
    /// More samples give more accurate importance estimates but take longer to compute.
    /// Used by EWC and related methods. Typical values: 100-1000.
    /// </remarks>
    public int FisherSampleSize { get; set; } = 200;
}
