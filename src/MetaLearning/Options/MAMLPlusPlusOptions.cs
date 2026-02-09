using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for MAML++ (How to Train Your MAML) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MAML++ is a production-hardened version of MAML that addresses several training instabilities
/// through multi-step loss optimization, derivative-order annealing, per-step learning rates,
/// and batch normalization stability fixes.
/// </para>
/// <para><b>For Beginners:</b> MAML++ is "MAML done right" for production use.
///
/// Original MAML has several practical problems:
/// - Training can be unstable (loss explodes)
/// - Batch normalization doesn't work well
/// - A single learning rate isn't optimal for all inner steps
/// - Second-order gradients are expensive and often unhelpful early on
///
/// MAML++ fixes ALL of these with:
/// 1. **Multi-Step Loss**: Supervise every inner step, not just the last one
/// 2. **Per-Step Learning Rates**: Each adaptation step has its own learning rate
/// 3. **Derivative-Order Annealing**: Start with first-order, gradually add second-order
/// 4. **Stable Batch Norm**: Per-step batch norm statistics for stable adaptation
///
/// Think of it like upgrading from a prototype to production code - same idea, but
/// with all the engineering needed to work reliably at scale.
/// </para>
/// <para>
/// Reference: Antoniou, A., Edwards, H., &amp; Storkey, A. (2019).
/// How to Train Your MAML. ICLR 2019.
/// </para>
/// </remarks>
public class MAMLPlusPlusOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the neural network that MAML++ will meta-train.
    /// MAML++ learns initial parameters for this model that can be quickly adapted to new tasks.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for outer loop updates.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the base inner learning rate (used for initialization of per-step rates).
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// In MAML++, each adaptation step can have its own learning rate (see PerStepLearningRates).
    /// This base rate is used to initialize per-step rates if they're not explicitly set.
    /// </para>
    /// <para><b>For Beginners:</b> The starting learning rate for each adaptation step.
    /// MAML++ can learn different rates for each step, but they all start at this value.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the outer learning rate for meta-parameter updates.
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of inner-loop adaptation steps.
    /// </summary>
    /// <value>Default is 5.</value>
    /// <remarks>
    /// <para>
    /// MAML++ uses multi-step loss optimization, supervising every step. More steps give
    /// better adaptation but increase computational cost. With per-step learning rates,
    /// 5 steps is usually sufficient.
    /// </para>
    /// <para><b>For Beginners:</b> How many gradient updates to perform when adapting to a task.
    /// Unlike standard MAML where only the final step is supervised, MAML++ supervises every step.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks per meta-batch.
    /// </summary>
    /// <value>Default is 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the gradient clipping threshold.
    /// </summary>
    /// <value>Default is 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed.
    /// </summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of evaluation tasks.
    /// </summary>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets the evaluation frequency.
    /// </summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save checkpoints.
    /// </summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets the checkpoint frequency.
    /// </summary>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is false (MAML++ uses full second-order with annealing).</value>
    /// <remarks>
    /// <para>
    /// MAML++ introduces derivative-order annealing: start with first-order gradients and
    /// gradually introduce second-order gradients as training stabilizes. This property
    /// controls the final target - if true, always use first-order; if false, anneal to second-order.
    /// </para>
    /// <para><b>For Beginners:</b> MAML++ is smart about gradient computation:
    /// - Starts simple (first-order) when training begins
    /// - Gradually switches to more accurate (second-order) as things stabilize
    /// - Set true to always use simple gradients (faster but less accurate)
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = false;

    #endregion

    #region MAML++ Specific Properties

    /// <summary>
    /// Gets or sets whether to use multi-step loss optimization (MSL).
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// Multi-step loss computes a weighted loss at every inner-loop step, not just the final one.
    /// This provides richer gradient signal and stabilizes training by ensuring intermediate
    /// adapted models also perform well.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of only checking performance after ALL adaptation steps,
    /// MSL checks at EVERY step. This helps the model learn to make good progress at each step,
    /// not just at the end. Think of it like a teacher checking your work after each problem,
    /// not just at the end of the exam.
    /// </para>
    /// </remarks>
    public bool UseMultiStepLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the weights for multi-step loss at each adaptation step.
    /// </summary>
    /// <value>Default is null (equal weights). Array length should equal AdaptationSteps.</value>
    /// <remarks>
    /// <para>
    /// If null, equal weights are used for all steps. Custom weights allow emphasizing
    /// later steps (which should be more adapted) over earlier steps.
    /// </para>
    /// <para><b>For Beginners:</b> Controls how much to care about performance at each step.
    /// null means equal importance at all steps. You could also weight later steps more heavily
    /// (e.g., [0.1, 0.2, 0.3, 0.4] for 4 steps) to emphasize final performance.
    /// </para>
    /// </remarks>
    public double[]? MultiStepLossWeights { get; set; }

    /// <summary>
    /// Gets or sets whether to use per-step learning rates (LSLR).
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// Learned Step-size Learning Rates (LSLR) assigns a separate learning rate to each
    /// adaptation step. Early steps may need larger rates for coarse adjustment, while
    /// later steps may need smaller rates for fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of one learning rate for all steps, each step
    /// gets its own rate that's learned during meta-training. This is like adjusting the
    /// volume knob differently for each song - some need more, some need less.
    /// </para>
    /// </remarks>
    public bool UsePerStepLearningRates { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use derivative-order annealing.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// Starts with first-order gradients (FOMAML) and linearly anneals to second-order
    /// gradients (full MAML) over AnnealingIterations. This prevents instability from
    /// second-order gradients early in training when parameters are far from convergence.
    /// </para>
    /// <para><b>For Beginners:</b> Gradually increases gradient accuracy during training:
    /// - Early training: Use simple (first-order) gradients for stability
    /// - Late training: Switch to complex (second-order) gradients for accuracy
    /// This prevents training from blowing up in early stages.
    /// </para>
    /// </remarks>
    public bool UseDerivativeOrderAnnealing { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of iterations over which to anneal from first-order to second-order.
    /// </summary>
    /// <value>Default is 500.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many training iterations to gradually transition from
    /// simple to complex gradients. Set to about half of NumMetaIterations.
    /// </para>
    /// </remarks>
    public int AnnealingIterations { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use per-step batch normalization statistics.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// Standard batch norm uses running statistics that become stale during inner-loop adaptation.
    /// Per-step batch norm maintains separate running statistics for each adaptation step,
    /// preventing the statistics mismatch that destabilizes MAML training.
    /// </para>
    /// <para><b>For Beginners:</b> Batch normalization can break during MAML's inner loop because
    /// the statistics become outdated. This fix keeps separate statistics for each adaptation step,
    /// like having separate calibration for each stage of a process.
    /// </para>
    /// </remarks>
    public bool UsePerStepBatchNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the cosine annealing schedule minimum learning rate ratio.
    /// </summary>
    /// <value>Default is 0.01 (1% of initial learning rate at minimum).</value>
    /// <remarks>
    /// <para>
    /// MAML++ uses cosine annealing for the outer learning rate. This sets the minimum
    /// learning rate as a fraction of the initial OuterLearningRate.
    /// </para>
    /// <para><b>For Beginners:</b> The outer learning rate decreases following a cosine curve
    /// during training. This value sets how low it can go (as a fraction of the starting rate).
    /// 0.01 means it can decrease to 1% of its starting value.
    /// </para>
    /// </remarks>
    public double CosineAnnealingMinRatio { get; set; } = 0.01;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MAMLPlusPlusOptions class.
    /// </summary>
    /// <param name="metaModel">The model to be meta-trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public MAMLPlusPlusOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0 &&
               AnnealingIterations >= 0 &&
               CosineAnnealingMinRatio >= 0 && CosineAnnealingMinRatio <= 1;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new MAMLPlusPlusOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer,
            DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate,
            OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps,
            MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            UseFirstOrder = UseFirstOrder,
            UseMultiStepLoss = UseMultiStepLoss,
            MultiStepLossWeights = MultiStepLossWeights != null ? (double[])MultiStepLossWeights.Clone() : null,
            UsePerStepLearningRates = UsePerStepLearningRates,
            UseDerivativeOrderAnnealing = UseDerivativeOrderAnnealing,
            AnnealingIterations = AnnealingIterations,
            UsePerStepBatchNorm = UsePerStepBatchNorm,
            CosineAnnealingMinRatio = CosineAnnealingMinRatio
        };
    }

    #endregion
}
