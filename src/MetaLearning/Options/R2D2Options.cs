using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for R2-D2 (Meta-learning with Differentiable Closed-form Solvers) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// R2-D2 replaces the iterative inner-loop optimization of MAML with a closed-form differentiable
/// ridge regression solver. This makes the inner loop extremely fast (single forward pass) while
/// remaining fully differentiable for meta-gradient computation.
/// </para>
/// <para><b>For Beginners:</b> R2-D2 is like MAML but with a mathematical shortcut:
///
/// **The Problem with MAML's inner loop:**
/// - MAML takes K gradient steps to adapt (iterative, slow)
/// - Each step requires computing gradients (expensive)
/// - More steps = more memory for second-order gradients
///
/// **R2-D2's solution:**
/// - Instead of gradient steps, use ridge regression (closed-form, instant)
/// - Ridge regression has an exact mathematical solution: w = (X^T X + lambda I)^-1 X^T y
/// - This solution is differentiable, so meta-gradients still flow through it
/// - Result: Lightning-fast inner loop with one matrix solve instead of K gradient steps
///
/// **Analogy:**
/// - MAML: Walk toward the solution step by step (might need 5-10 steps)
/// - R2-D2: Jump directly to the solution using math (one step, exact answer)
///
/// The catch: R2-D2's "jump" only works for the last layer (linear classifier),
/// so the feature extractor still needs good meta-learned features.
/// </para>
/// <para>
/// Reference: Bertinetto, L., Henriques, J. F., Torr, P., &amp; Vedaldi, A. (2019).
/// Meta-learning with Differentiable Closed-form Solvers. ICLR 2019.
/// </para>
/// </remarks>
public class R2D2Options<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the feature extractor model (backbone).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model serves as the feature extractor (backbone). Its output features
    /// are fed into the differentiable ridge regression solver. The backbone is meta-learned
    /// in the outer loop, while the ridge regression classifier is computed in closed-form.
    /// </para>
    /// <para><b>For Beginners:</b> This is the neural network that converts raw inputs into
    /// meaningful features. R2-D2 then uses ridge regression on these features to classify.
    /// The network learns to produce features that are easy to classify with ridge regression.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function.
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the outer loop optimizer.
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the inner loop optimizer (not used for R2-D2, kept for interface compatibility).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader.
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the inner learning rate (not used directly; R2-D2 uses closed-form solver).
    /// </summary>
    /// <value>Default is 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> R2-D2 doesn't use gradient descent in the inner loop,
    /// so this parameter is not directly used. It's kept for interface compatibility.
    /// The ridge regression regularization parameter (Lambda) controls the inner "learning" instead.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the outer learning rate.
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps (set to 1 for R2-D2's closed-form solver).
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// <para>
    /// R2-D2 uses a closed-form solver that computes the optimal classifier in one step.
    /// This is conceptually 1 adaptation step. Setting higher values has no effect since
    /// the closed-form solution is already optimal.
    /// </para>
    /// <para><b>For Beginners:</b> R2-D2 only needs 1 step because it computes the
    /// exact solution mathematically. Unlike MAML which needs multiple steps to
    /// approximately reach the solution.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the meta-batch size.
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
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed.
    /// </summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets evaluation tasks count.
    /// </summary>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets evaluation frequency.
    /// </summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save checkpoints.
    /// </summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets checkpoint frequency.
    /// </summary>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is false (R2-D2 naturally supports differentiation through the solver).</value>
    /// <remarks>
    /// <para>
    /// R2-D2's closed-form solver is naturally differentiable, so second-order gradients
    /// flow through the ridge regression solution without issues. First-order approximation
    /// is still available for computational savings if needed.
    /// </para>
    /// <para><b>For Beginners:</b> Keep false. R2-D2's math naturally supports full gradients
    /// without the instability issues MAML has. This is one of its key advantages.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = false;

    #endregion

    #region R2-D2 Specific Properties

    /// <summary>
    /// Gets or sets the ridge regression regularization parameter (lambda).
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// Lambda controls the strength of L2 regularization in the ridge regression solver.
    /// The closed-form solution is: w = (X^T X + lambda I)^-1 X^T y
    /// Larger lambda = more regularization = simpler classifier = less overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how "smooth" the classifier is:
    /// - Small lambda (0.01): Very flexible classifier, might overfit on few examples
    /// - Medium lambda (1.0): Good balance (recommended)
    /// - Large lambda (100): Very smooth classifier, might underfit
    ///
    /// Think of it like smoothing a curve: more smoothing prevents fitting noise but
    /// might miss real patterns. 1.0 is usually a good starting point.
    /// </para>
    /// </remarks>
    public double Lambda { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether lambda should be meta-learned.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// When true, lambda is treated as a meta-parameter and updated in the outer loop.
    /// This allows the algorithm to automatically find the optimal regularization strength.
    /// </para>
    /// <para><b>For Beginners:</b> If true, the algorithm automatically figures out the best
    /// smoothing level. If false, it uses the fixed value you provide. Usually set to true
    /// to let the algorithm optimize this for you.
    /// </para>
    /// </remarks>
    public bool LearnLambda { get; set; } = true;

    /// <summary>
    /// Gets or sets the feature embedding dimension for ridge regression.
    /// </summary>
    /// <value>Default is 0 (uses the feature extractor's output dimension automatically).</value>
    /// <remarks>
    /// <para>
    /// If 0, the ridge regression operates on the raw feature extractor output.
    /// If positive, an additional linear projection maps features to this dimension before
    /// ridge regression. This can help when the feature dimension is too large.
    /// </para>
    /// <para><b>For Beginners:</b> The size of the feature representation used for classification.
    /// Set to 0 to automatically match the feature extractor's output. Set to a smaller number
    /// (e.g., 64) if the feature extractor outputs very high-dimensional features.
    /// </para>
    /// </remarks>
    public int EmbeddingDimension { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use Woodbury identity for efficient matrix inversion.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// The Woodbury identity allows computing (X^T X + lambda I)^-1 more efficiently
    /// when the number of samples is much smaller than the feature dimension (common
    /// in few-shot learning). This reduces complexity from O(d^3) to O(n^3) where
    /// n &lt;&lt; d for few-shot tasks.
    /// </para>
    /// <para><b>For Beginners:</b> A math trick that makes the computation faster when
    /// you have few examples but many features (which is exactly the few-shot setting).
    /// Keep this as true for best performance.
    /// </para>
    /// </remarks>
    public bool UseWoodburyIdentity { get; set; } = true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the R2D2Options class.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public R2D2Options(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0 &&
               Lambda > 0 &&
               EmbeddingDimension >= 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new R2D2Options<T, TInput, TOutput>(MetaModel)
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
            Lambda = Lambda,
            LearnLambda = LearnLambda,
            EmbeddingDimension = EmbeddingDimension,
            UseWoodburyIdentity = UseWoodburyIdentity
        };
    }

    #endregion
}
