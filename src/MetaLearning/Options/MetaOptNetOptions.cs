using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MetaOptNet replaces the gradient-based inner-loop of MAML with a differentiable
/// convex optimization solver. Instead of taking gradient steps, it solves a closed-form
/// optimization problem (like ridge regression or SVM) to get the classifier.
/// </para>
/// <para><b>For Beginners:</b> In MAML, the inner loop does gradient descent:
/// "take a step, take a step, take a step..." which can be slow and unstable.
/// MetaOptNet says: "Why iterate? Just solve the optimal answer directly!"
/// </para>
/// <para>
/// It uses mathematical formulas that give the best classifier in one shot.
/// This is:
/// - <b>Faster:</b> No iterative optimization
/// - <b>More stable:</b> Convex problems have unique solutions
/// - <b>Theoretically grounded:</b> Based on well-understood optimization theory
/// </para>
/// <para>
/// Reference: Lee, K., Maji, S., Ravichandran, A., &amp; Soatto, S. (2019).
/// Meta-Learning with Differentiable Convex Optimization. CVPR 2019.
/// </para>
/// </remarks>
public class MetaOptNetOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (feature encoder) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a feature encoder (typically a CNN for images).
    /// MetaOptNet learns the encoder while using a convex solver for classification.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses cross-entropy loss internally).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter (outer loop) updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates.
    /// Default: null (not used in MetaOptNet - inner loop is solved analytically).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (not used - solver is analytical).
    /// </summary>
    /// <value>Default is 0.01 (placeholder, not used).</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (encoder training).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    /// <value>Default is 1 (MetaOptNet uses closed-form solution).</value>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default is 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default is 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate the meta-learner.
    /// </summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save model checkpoints.
    /// </summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is false (MetaOptNet uses implicit differentiation).</value>
    public bool UseFirstOrder { get; set; } = false;

    #endregion

    #region MetaOptNet-Specific Properties

    /// <summary>
    /// Gets or sets the convex base learner.
    /// </summary>
    /// <value>Default is SVM, the paper's base learner (MetaOptNet-SVM).</value>
    /// <remarks>
    /// <para>
    /// The Crammer and Singer multi-class SVM in its dual form (eq. 10) is the paper's headline learner; ridge
    /// regression (eq. 11) is the cheaper alternative it compares against, and logistic regression is the third
    /// convex choice the framework allows. All three are solved numerically and differentiated through their KKT
    /// conditions.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// - <b>SVM:</b> the paper's choice, best accuracy, a small quadratic program per task
    /// - <b>RidgeRegression:</b> a closed form, fastest
    /// - <b>LogisticRegression:</b> probabilistic scores, solved by Newton's method
    /// </para>
    /// </remarks>
    public ConvexSolverType SolverType { get; set; } = ConvexSolverType.SVM;

    /// <summary>
    /// Gets or sets a regularization parameter that overrides the active solver's own setting.
    /// </summary>
    /// <value>Default is null: each solver uses <see cref="SvmCost"/>, <see cref="RidgeLambda"/> or
    /// <see cref="LogisticLambda"/>.</value>
    /// <remarks>
    /// <para>The solvers regularize on different scales - the reference implementation uses C = 0.1 for the SVM and
    /// lambda = 50 for ridge regression - so one shared value cannot serve both. Set this only to force one.</para>
    /// </remarks>
    public double? RegularizationStrength { get; set; }

    /// <summary>
    /// Gets or sets C, the SVM's regularization parameter (eq. 4).
    /// </summary>
    /// <value>Default is 0.1, the reference implementation's value.</value>
    public double SvmCost { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets lambda, the ridge regression regularization (eq. 11).
    /// </summary>
    /// <value>Default is 50.0, the reference implementation's value.</value>
    public double RidgeLambda { get; set; } = 50.0;

    /// <summary>
    /// Gets or sets the L2 regularization of the logistic regression base learner.
    /// </summary>
    /// <value>Default is 1.0. The paper names logistic regression as a choice but reports no value; this one is
    /// ours.</value>
    public double LogisticLambda { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the label smoothing applied to the query cross-entropy.
    /// </summary>
    /// <value>Default is 0.0 (off). The paper uses 0.1 for miniImageNet with ResNet-12.</value>
    /// <remarks>
    /// <para>Smoothing belongs to that benchmark's training recipe rather than to the method, so it is off by
    /// default. Above zero, the query loss is the cross-entropy against the smoothed label distribution.</para>
    /// </remarks>
    public double LabelSmoothing { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets the width of each example's embedding: the embedding network's per-example output width.
    /// </summary>
    /// <value>Default is 512.</value>
    public int EmbeddingDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether to scale each embedding to unit length before solving.
    /// </summary>
    /// <value>Default is false: the reference implementation's heads read the embeddings as they are.</value>
    public bool NormalizeEmbeddings { get; set; } = false;

    /// <summary>
    /// Gets or sets whether the logit scale of eq. 12 is learned.
    /// </summary>
    /// <value>Default is true, as in the reference implementation.</value>
    /// <remarks>
    /// <para>The scale multiplies the base learner's logits before the loss. False keeps it at
    /// <see cref="InitialTemperature"/>.</para>
    /// </remarks>
    public bool UseLearnedTemperature { get; set; } = true;

    /// <summary>
    /// Gets or sets the starting value of the logit scale of eq. 12.
    /// </summary>
    /// <value>Default is 1.0, the reference implementation's initialisation.</value>
    /// <remarks>
    /// <para>It multiplies the logits; it used to divide them and was clamped at 0.01, and its gradient was a
    /// forward difference.</para>
    /// </remarks>
    public double InitialTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the iteration budget of the iterative base learners (SVM and logistic regression).
    /// </summary>
    /// <value>Default is 15, the reference implementation's QP iteration count.</value>
    public int MaxSolverIterations { get; set; } = 15;

    /// <summary>
    /// Gets or sets the convergence tolerance for iterative solvers.
    /// </summary>
    /// <value>Default is 1e-6.</value>
    public double SolverTolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the L2 regularization strength for the encoder.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double EncoderL2Regularization { get; set; } = 0.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MetaOptNetOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The feature encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public MetaOptNetOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all MetaOptNet configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               (RegularizationStrength is null || RegularizationStrength > 0) &&
               SvmCost > 0 &&
               RidgeLambda > 0 &&
               LogisticLambda > 0 &&
               LabelSmoothing >= 0 && LabelSmoothing < 1 &&
               NumClasses > 0 &&
               EmbeddingDimension > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               MaxSolverIterations > 0 &&
               SolverTolerance > 0 &&
               InitialTemperature > 0;
    }

    /// <summary>
    /// Creates a deep copy of the MetaOptNet options.
    /// </summary>
    /// <returns>A new MetaOptNetOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new MetaOptNetOptions<T, TInput, TOutput>(MetaModel)
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
            SolverType = SolverType,
            RegularizationStrength = RegularizationStrength,
            SvmCost = SvmCost,
            RidgeLambda = RidgeLambda,
            LogisticLambda = LogisticLambda,
            LabelSmoothing = LabelSmoothing,
            NumClasses = NumClasses,
            EmbeddingDimension = EmbeddingDimension,
            NormalizeEmbeddings = NormalizeEmbeddings,
            UseLearnedTemperature = UseLearnedTemperature,
            InitialTemperature = InitialTemperature,
            MaxSolverIterations = MaxSolverIterations,
            SolverTolerance = SolverTolerance,
            EncoderL2Regularization = EncoderL2Regularization
        };
    }

    #endregion
}
