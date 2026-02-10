using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the WarpGrad (Warped Gradient Descent) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// WarpGrad learns a preconditioning matrix (warp-layers) that transforms gradients during
/// inner-loop adaptation. Unlike MAML which learns a good initialization, WarpGrad learns
/// a good gradient descent geometry that makes adaptation more efficient.
/// </para>
/// <para><b>For Beginners:</b> WarpGrad is like learning a better map for navigating a landscape.
///
/// Imagine you're trying to reach the bottom of a valley (optimal solution):
/// - Standard gradient descent: Walk straight downhill (may zigzag on narrow valleys)
/// - MAML: Start from a better position on the hillside
/// - WarpGrad: Learn to reshape the landscape so downhill is always the right direction
///
/// The "warp layers" transform the gradients (direction signals) so that even simple
/// gradient descent moves efficiently toward the solution. This is like giving the model
/// a compass that's been calibrated for different types of tasks.
///
/// Key advantages over MAML:
/// - No need to backpropagate through the inner loop (much cheaper)
/// - Warp layers provide task-independent gradient conditioning
/// - Can be combined with any inner-loop optimizer
/// </para>
/// <para>
/// Reference: Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., &amp; Hadsell, R. (2020).
/// Meta-Learning with Warped Gradient Descent. ICLR 2020.
/// </para>
/// </remarks>
public class WarpGradOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model's parameters serve as the task learner, with warp-layers interleaved
    /// between its layers. The model's gradients are transformed by the learned warp-layers
    /// during inner-loop adaptation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the neural network that actually makes predictions.
    /// WarpGrad will learn special "warp layers" that improve how this model adapts to new tasks.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses the model's default loss function).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for outer loop updates (warp-layer and shared parameter updates).
    /// Default: null (uses built-in SGD with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates (task parameter updates).
    /// Default: null (uses built-in SGD with InnerLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task-specific adaptation).
    /// </summary>
    /// <value>The inner learning rate. Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Controls how quickly task-specific parameters are updated during adaptation.
    /// With warp-layers preconditioning the gradients, this learning rate interacts
    /// with the warp geometry to determine effective step sizes.
    /// </para>
    /// <para><b>For Beginners:</b> How fast the model adapts to each new task.
    /// The warp layers help make this learning rate more effective by reshaping
    /// the optimization landscape. Typical range: 0.001 to 0.1.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-parameter and warp-layer updates).
    /// </summary>
    /// <value>The outer learning rate. Default is 0.001.</value>
    /// <remarks>
    /// <para>
    /// Controls how quickly the warp-layers and initialization parameters are updated
    /// during meta-training. This affects both the gradient preconditioning (warp-layers)
    /// and the initial task-learner parameters.
    /// </para>
    /// <para><b>For Beginners:</b> How fast the "gradient compass" (warp layers) and
    /// starting point improve across all tasks. Start with 0.001.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps for inner loop adaptation.
    /// </summary>
    /// <value>The number of adaptation steps. Default is 5.</value>
    /// <remarks>
    /// <para>
    /// WarpGrad does not require backpropagating through these steps (unlike MAML),
    /// so more steps are computationally feasible. The warp-layers are trained to make
    /// these steps effective regardless of the number of steps used.
    /// </para>
    /// <para><b>For Beginners:</b> How many gradient descent steps to take when adapting
    /// to a new task. Since WarpGrad doesn't need to track through these steps (unlike MAML),
    /// you can use more steps cheaply. 5-10 is a good default.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>The meta-batch size. Default is 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>The number of meta-iterations. Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>The gradient clip threshold, or null to disable. Default is 10.0.</value>
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
    /// Gets or sets whether to use first-order approximation for meta-gradients.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <para>
    /// WarpGrad is inherently a first-order method because warp-layers are NOT differentiated
    /// through the inner loop. The meta-gradient flows through the warp-layer outputs only,
    /// not through the inner-loop optimization trajectory. Setting this to true is standard.
    /// </para>
    /// <para><b>For Beginners:</b> Keep this as true. WarpGrad's key advantage is that it
    /// doesn't need expensive second-order gradients like MAML does.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region WarpGrad-Specific Properties

    /// <summary>
    /// Gets or sets the hidden dimension for warp-layer MLPs.
    /// </summary>
    /// <value>The warp-layer hidden dimension. Default is 64.</value>
    /// <remarks>
    /// <para>
    /// Each warp-layer is a small MLP that transforms gradients. This controls the width
    /// of those MLPs. Larger values give more expressive gradient transformations but
    /// increase the number of meta-parameters.
    /// </para>
    /// <para><b>For Beginners:</b> Controls how complex the gradient transformation can be.
    /// 64 is a good default. Use 32 for simpler tasks, 128 for very complex ones.
    /// </para>
    /// </remarks>
    public int WarpLayerHiddenDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of warp-layers to interleave with the task learner.
    /// </summary>
    /// <value>The number of warp layers. Default is 2.</value>
    /// <remarks>
    /// <para>
    /// Warp-layers are placed between consecutive layers of the task learner.
    /// More warp-layers provide finer-grained gradient control but add meta-parameters.
    /// Typically set to the number of hidden layers in the task learner.
    /// </para>
    /// <para><b>For Beginners:</b> How many gradient transformation points to insert.
    /// Usually 1-3 is sufficient. More warp layers = more precise gradient control
    /// but also more parameters to meta-learn.
    /// </para>
    /// </remarks>
    public int NumWarpLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether warp-layers use diagonal (element-wise) or full matrix transformations.
    /// </summary>
    /// <value>True for diagonal warp (faster, fewer params); false for full matrix. Default is false.</value>
    /// <remarks>
    /// <para>
    /// Diagonal warp-layers scale each gradient dimension independently (like per-parameter
    /// learning rates). Full matrix warp-layers can rotate and scale gradient space
    /// (more expressive but more parameters).
    /// </para>
    /// <para><b>For Beginners:</b>
    /// - Diagonal (true): Each gradient dimension is scaled independently. Fast, simple.
    /// - Full (false): Gradient dimensions can interact. More powerful but slower.
    /// Start with false (full) for best results, switch to true if training is too slow.
    /// </para>
    /// </remarks>
    public bool UseDiagonalWarp { get; set; } = false;

    /// <summary>
    /// Gets or sets the learning rate for warp-layer parameter updates.
    /// </summary>
    /// <value>The warp learning rate. Default is 0.001.</value>
    /// <remarks>
    /// <para>
    /// Warp-layer parameters can be updated at a different rate than the task-learner
    /// initialization. If null or zero, uses OuterLearningRate for warp-layer updates.
    /// </para>
    /// <para><b>For Beginners:</b> How fast the gradient transformation (compass) improves.
    /// Usually the same as OuterLearningRate. Set separately if warp layers need
    /// different update dynamics than the initialization parameters.
    /// </para>
    /// </remarks>
    public double WarpLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the initialization scale for warp-layer parameters.
    /// </summary>
    /// <value>The initialization scale. Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Warp-layers are initialized near identity (no transformation) and gradually
    /// learn useful gradient warpings. This scale controls how close to identity
    /// the initial warp is. Smaller values start closer to unwarped gradients.
    /// </para>
    /// <para><b>For Beginners:</b> How much the initial gradient transformation differs
    /// from "no transformation." Small values (0.01) mean warp layers start nearly
    /// transparent and gradually learn useful transformations. This is safest.
    /// </para>
    /// </remarks>
    public double WarpInitScale { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the L2 regularization weight for warp-layer parameters.
    /// </summary>
    /// <value>The warp regularization weight. Default is 0.0001.</value>
    /// <remarks>
    /// <para>
    /// Regularizing warp-layers prevents them from learning extreme gradient transformations
    /// that might destabilize training. This pulls warp parameters toward zero (identity warp).
    /// </para>
    /// <para><b>For Beginners:</b> Prevents the gradient transformation from becoming too extreme.
    /// A small value like 0.0001 gently keeps warp layers reasonable.
    /// </para>
    /// </remarks>
    public double WarpRegularization { get; set; } = 0.0001;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the WarpGradOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The model to be meta-trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// var options = new WarpGradOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder)
    /// {
    ///     NumWarpLayers = 3,
    ///     WarpLayerHiddenDim = 64,
    ///     AdaptationSteps = 5,
    ///     InnerLearningRate = 0.01
    /// };
    /// var warpGrad = new WarpGradAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    /// </code>
    /// </example>
    public WarpGradOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all WarpGrad configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0 &&
               EvaluationFrequency > 0 &&
               CheckpointFrequency > 0 &&
               NumWarpLayers > 0 &&
               WarpLayerHiddenDim > 0 &&
               WarpLearningRate > 0 &&
               WarpInitScale > 0 &&
               WarpRegularization >= 0;
    }

    /// <summary>
    /// Creates a copy of the WarpGrad options. Value-type properties are copied by value;
    /// reference-type properties (MetaModel, LossFunction, MetaOptimizer, InnerOptimizer,
    /// DataLoader) are copied by reference (shallow copy).
    /// </summary>
    /// <returns>A new WarpGradOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new WarpGradOptions<T, TInput, TOutput>(MetaModel)
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
            WarpLayerHiddenDim = WarpLayerHiddenDim,
            NumWarpLayers = NumWarpLayers,
            UseDiagonalWarp = UseDiagonalWarp,
            WarpLearningRate = WarpLearningRate,
            WarpInitScale = WarpInitScale,
            WarpRegularization = WarpRegularization
        };
    }

    #endregion
}
