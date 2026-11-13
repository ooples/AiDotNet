using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of WarpGrad (Meta-Learning with Warped Gradient Descent) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// WarpGrad learns a preconditioning transformation (warp-layers) that reshapes the gradient
/// descent landscape to make inner-loop adaptation more efficient. Unlike MAML which only
/// learns a good initialization, WarpGrad learns a good optimization geometry.
/// </para>
/// <para><b>For Beginners:</b> WarpGrad improves how gradient descent works, not just where it starts:
///
/// **How it works:**
/// 1. Standard gradient descent follows the steepest direction downhill
/// 2. But "steepest" depends on how you measure distance in parameter space
/// 3. WarpGrad learns warp-layers that transform gradients before they're applied
/// 4. These transformed gradients lead to more effective parameter updates
/// 5. The warp-layers are shared across all tasks (meta-learned)
///
/// **Analogy:**
/// Imagine navigating a city to reach your destination:
/// - Standard gradient descent: Always walk directly toward the destination (may hit walls)
/// - MAML: Start from a good location in the city
/// - WarpGrad: Learn the city's road network so you always take efficient routes
///
/// **Why it's better than MAML:**
/// - No second-order gradients needed (no backprop through inner loop)
/// - Warp-layers provide task-independent preconditioning
/// - Can use any number of inner-loop steps cheaply
/// - Naturally handles different parameter scales across layers
/// </para>
/// <para><b>Algorithm - WarpGrad:</b>
/// <code>
/// # Initialization
/// theta = model_parameters       # Task-learner initialization (meta-learned)
/// W = warp_layer_parameters      # Gradient preconditioning (meta-learned)
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         theta_i = copy(theta)   # Start from meta-learned initialization
///
///         # Inner loop: Adapt with warped gradients
///         for step in range(K):
///             loss = compute_loss(theta_i, support_x, support_y)
///             g = gradient(loss, theta_i)         # Raw gradient
///             g_warped = warp(g, W)               # Transform gradient through warp-layers
///             theta_i = theta_i - alpha * g_warped # Update with warped gradient
///
///         # Evaluate adapted model on query set
///         meta_loss_i = compute_loss(theta_i, query_x, query_y)
///
///     # Outer loop: Update initialization AND warp-layers
///     theta = theta - beta * mean(grad(meta_loss, theta))
///     W = W - beta_w * mean(grad(meta_loss, W))
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Gradient Geometry**: Warp-layers learn a Riemannian metric that makes gradient
///    descent more effective, similar to natural gradient methods but task-adapted.
///
/// 2. **No Inner-Loop Backprop**: Unlike MAML, gradients for warp-layers flow through
///    the warp transformation only, not through the entire inner-loop trajectory.
///    This makes WarpGrad O(K) per task instead of MAML's O(K^2).
///
/// 3. **Complementary to Initialization**: WarpGrad improves both WHERE you start (theta)
///    and HOW you move (W), providing two orthogonal axes of meta-learning.
///
/// 4. **Identity Initialization**: Warp-layers start near identity (no warping) and
///    gradually learn useful transformations during meta-training.
/// </para>
/// <para>
/// Reference: Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., &amp; Hadsell, R. (2020).
/// Meta-Learning with Warped Gradient Descent. ICLR 2020.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Warped Gradient Descent",
    "https://arxiv.org/abs/1909.00025",
    Year = 2020,
    Authors = "Flennerhag, S., Rusu, A. A., Pascanu, R., Visin, F., Yin, H., & Hadsell, R.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class WarpGradAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly WarpGradOptions<T, TInput, TOutput> _warpOptions;

    /// <summary>
    /// Warp-layer parameters that precondition gradients during inner-loop adaptation.
    /// Each entry is a vector of warp parameters for one warp-layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the meta-learned parameters that define the gradient preconditioning.
    /// They are updated only in the outer loop and shared across all tasks.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "compass calibration" parameters.
    /// They tell the model how to transform its raw gradients into more effective
    /// update directions. They're shared across all tasks and improve during meta-training.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _warpLayerParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.WarpGrad;

    /// <summary>
    /// Gets the warp-layer parameters for inspection or serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the learned gradient transformation parameters.
    /// After meta-training, you can inspect them to understand how the algorithm has learned
    /// to transform gradients for effective adaptation.
    /// </para>
    /// </remarks>
    public IReadOnlyList<Vector<T>> WarpLayerParameters => _warpLayerParams.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the WarpGrad algorithm.
    /// </summary>
    /// <param name="options">Configuration options for WarpGrad.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or its MetaModel is null.</exception>
    /// <exception cref="ArgumentException">Thrown when options validation fails.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes:
    /// 1. The base meta-learner with the provided model and loss function
    /// 2. Warp-layer parameters near identity (small random perturbation)
    /// 3. The random number generator for reproducible results
    /// </para>
    /// <para><b>For Beginners:</b> Creates a new WarpGrad meta-learner. You need to provide:
    /// - A neural network model (MetaModel in options) - this is what makes predictions
    /// - Configuration options - how many warp layers, learning rates, etc.
    /// The warp layers start as "do nothing" transformations and learn useful warpings
    /// during meta-training.
    /// </para>
    /// </remarks>
    public WarpGradAlgorithm(WarpGradOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            // The warp is a learned per-coordinate gradient scale, which an adaptive inner optimizer such as Adam
            // divides back out: the warp would stop shaping adaptation and its finite-difference gradient would go to
            // zero. So the inner steps stay plain SGD; a meta-optimizer still takes the outer step.
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _warpOptions = options;
        _warpLayerParams = new List<Vector<T>>();

        InitializeWarpLayers();
    }

    /// <summary>
    /// Initializes warp-layer parameters near identity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each warp-layer is initialized with small random values so that it starts
    /// as a near-identity transformation. This means initially, warped gradients
    /// are approximately equal to raw gradients, and the warp gradually learns
    /// useful transformations during meta-training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the gradient transformation layers
    /// to start as "pass-through" (do nothing). As meta-training progresses,
    /// they gradually learn to transform gradients in useful ways. Starting
    /// near identity ensures stable early training.
    /// </para>
    /// </remarks>
    private void InitializeWarpLayers()
    {
        var modelParams = ParamModel.GetParameters();
        int paramsPerLayer = modelParams.Length / Math.Max(_warpOptions.NumWarpLayers, 1);

        for (int i = 0; i < _warpOptions.NumWarpLayers; i++)
        {
            int layerDim = _warpOptions.UseDiagonalWarp
                ? paramsPerLayer
                : _warpOptions.WarpLayerHiddenDim;

            // Initialize near identity: U(1 - s, 1 + s). One NextDouble per warp parameter, in order; the
            // loop only stores the draws.
            var draws = new double[layerDim];
            for (int j = 0; j < layerDim; j++)
                draws[j] = RandomGenerator.NextDouble();

            _warpLayerParams.Add(AroundIdentity(draws));
        }
    }

    /// <summary>
    /// Maps U(0, 1) draws u to 1 + (u - 0.5) * 2 * WarpInitScale, in double, rounding each warp parameter to T once
    /// (rounding u to Half first lost bits of the draw before it was scaled).
    /// </summary>
    private Vector<T> AroundIdentity(double[] draws)
    {
        double s = _warpOptions.WarpInitScale;
        var u = new Tensor<double>(draws, new[] { draws.Length });
        return Engine.TensorAddScalar(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(u, -0.5), 2.0 * s), 1.0)
            .Cast<T>().ToVector();
    }

    /// <summary>
    /// Performs one meta-training step on a batch of tasks using warped gradient descent.
    /// </summary>
    /// <param name="taskBatch">A batch of meta-learning tasks, each with support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <remarks>
    /// <para>
    /// For each task in the batch:
    /// 1. Clone the model to get a fresh copy with the meta-learned initialization
    /// 2. Perform K inner-loop steps using warped gradients on the support set
    /// 3. Evaluate the adapted model on the query set to get the meta-loss
    ///
    /// After processing all tasks, update:
    /// - The task-learner initialization (theta) using averaged meta-gradients
    /// - The warp-layer parameters (W) using their own meta-gradients
    /// </para>
    /// <para><b>For Beginners:</b> This is where the "learning to learn" happens:
    ///
    /// For each practice task:
    /// 1. Start with the current best initialization
    /// 2. Adapt using warped (transformed) gradients on the task's examples
    /// 3. Test how well the adapted model performs on held-out examples
    /// 4. Measure how good the adaptation was
    ///
    /// Then improve both:
    /// - The starting point (initialization) - so future adaptations start better
    /// - The gradient transformation (warp) - so future adaptations move more efficiently
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var warpGradientsList = new List<List<Vector<T>>>();
        var losses = new List<T>();

        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop with warped gradients; the warp meta-gradient is accumulated along the trajectory.
            var trajectory = WarpedTrajectory(task, initParams);

            // Evaluate on query set (the model holds the adapted parameters)
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));

            // First-order meta-gradient for the initialization: the query gradient at the adapted parameters
            metaGradients.Add(ClipGradients(trajectory.QueryGradient));
            warpGradientsList.Add(trajectory.WarpGradients);
        }

        // Restore original params before applying outer update
        ParamModel.SetParameters(initParams);

        // Average and apply meta-gradients to initialization
        if (metaGradients.Count > 0)
        {
            var avgMetaGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyOuterGradients(initParams, avgMetaGrad, _warpOptions.OuterLearningRate);
            ParamModel.SetParameters(updatedParams);
        }

        // Average and apply warp-layer gradients
        UpdateWarpLayers(warpGradientsList);

        return ComputeMean(losses);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using warped gradient descent.
    /// </summary>
    /// <param name="task">The meta-learning task containing support and query sets.</param>
    /// <returns>An adapted model specialized for the given task.</returns>
    /// <remarks>
    /// <para>
    /// Adaptation uses the learned warp-layers to precondition gradients during
    /// inner-loop optimization. The process is:
    /// 1. Start from the meta-learned initialization
    /// 2. For each adaptation step, compute raw gradients on the support set
    /// 3. Transform gradients through the learned warp-layers
    /// 4. Apply the warped gradients to update task parameters
    /// 5. Return the adapted model
    /// </para>
    /// <para><b>For Beginners:</b> This is how you use WarpGrad on a new task:
    ///
    /// 1. Start with the model's meta-learned parameters (good starting point)
    /// 2. Look at the task's few examples (support set)
    /// 3. Compute gradients (which direction to adjust the model)
    /// 4. Transform those gradients using the learned warp (make them more effective)
    /// 5. Update the model using the warped gradients
    /// 6. Repeat for a few steps
    /// 7. Return the adapted model, ready to make predictions
    ///
    /// The warp layers make each gradient step count more, so even with few examples,
    /// the model can adapt effectively.
    /// </para>
    /// <para>
    /// Adaptation runs on a private clone of the meta-model. It used to step the shared meta-model itself and
    /// return a wrapper around it, so adapting to a task overwrote the meta-learned initialisation.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task is null)
            throw new ArgumentNullException(nameof(task));

        // Start from meta-learned initialization, on a copy the task owns
        var taskModel = CloneModel();
        var taskParameters = InterfaceGuard.Parameterizable(taskModel);
        var adaptedParams = taskParameters.GetParameters();

        // Inner loop: adapt with warped gradients
        for (int step = 0; step < _warpOptions.AdaptationSteps; step++)
        {
            var rawGradients = ComputeGradients(taskModel, task.SupportInput, task.SupportOutput);
            var warpedGradients = ApplyWarpLayers(rawGradients);
            warpedGradients = ClipGradients(warpedGradients);

            adaptedParams = ApplyInnerGradients(adaptedParams, warpedGradients, _warpOptions.InnerLearningRate, step == 0);
            taskParameters.SetParameters(adaptedParams);
        }

        // Return adapted model wrapped for inference
        return new WarpGradModel<T, TInput, TOutput>(taskModel, adaptedParams);
    }

    /// <summary>
    /// Applies the learned warp-layers to transform raw gradients.
    /// </summary>
    /// <param name="rawGradients">The raw gradients from the loss computation.</param>
    /// <returns>Warped (preconditioned) gradients.</returns>
    /// <remarks>
    /// <para>
    /// The warp transformation applies element-wise scaling to different segments of the
    /// gradient vector. For diagonal warp, each element is scaled independently.
    /// The warp parameters act as a learned preconditioner that reshapes the
    /// optimization landscape.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the gradient transformation happens.
    /// Think of it as applying a "filter" to the raw gradients:
    /// - Some gradient dimensions get amplified (the warp says "this direction is important")
    /// - Some get dampened (the warp says "this direction is noisy, go carefully")
    /// - The overall effect is more efficient parameter updates
    /// </para>
    /// </remarks>
    private Vector<T> ApplyWarpLayers(Vector<T> rawGradients)
        => Engine.Multiply(rawGradients, WarpScale(rawGradients.Length));

    /// <summary>
    /// The per-coordinate preconditioner P: warp layer w covers the coordinates
    /// [w * L / numLayers, (w + 1) * L / numLayers) (the last layer runs to the end), and coordinate i in it is scaled by
    /// its warp parameter (i - start) mod |warp_w|.
    /// </summary>
    private Vector<T> WarpScale(int length)
    {
        if (_warpLayerParams.Count == 0) return Engine.Fill(length, NumOps.One);

        var segments = new List<Vector<T>>(_warpLayerParams.Count);
        for (int w = 0; w < _warpLayerParams.Count; w++)
        {
            var (_, span) = WarpSpan(w, length);
            if (span > 0) segments.Add(Tile(_warpLayerParams[w], span));
        }

        return Vector<T>.Concatenate(segments);
    }

    /// <summary>The first coordinate and the number of coordinates warp layer <paramref name="w"/> scales.</summary>
    private (int Start, int Span) WarpSpan(int w, int length)
    {
        int paramsPerLayer = length / Math.Max(_warpOptions.NumWarpLayers, 1);
        int start = w * paramsPerLayer;
        int end = (w == _warpLayerParams.Count - 1) ? length : Math.Min(start + paramsPerLayer, length);
        return (start, Math.Max(0, end - start));
    }

    /// <summary>The first <paramref name="span"/> entries of <paramref name="warp"/> repeated end to end.</summary>
    private Vector<T> Tile(Vector<T> warp, int span)
    {
        // An empty warp layer cannot scale anything: those coordinates pass through unwarped.
        if (warp.Length == 0) return Engine.Fill(span, NumOps.One);
        int repeats = (span + warp.Length - 1) / warp.Length;
        var copies = new List<Vector<T>>(repeats);
        for (int r = 0; r < repeats; r++) copies.Add(warp);
        return Vector<T>.Concatenate(copies).GetSubVector(0, span);
    }

    /// <summary>
    /// The adjoint of <see cref="WarpScale"/>: sums a per-coordinate gradient over the coordinates each warp parameter
    /// scales, giving one gradient vector per warp layer.
    /// </summary>
    private List<Vector<T>> ReduceToWarpLayers(Vector<T> coordinateGradient)
    {
        var result = new List<Vector<T>>(_warpLayerParams.Count);
        for (int w = 0; w < _warpLayerParams.Count; w++)
        {
            int size = _warpLayerParams[w].Length;
            var (start, span) = WarpSpan(w, coordinateGradient.Length);
            if (size == 0 || span == 0)
            {
                result.Add(new Vector<T>(size));
                continue;
            }

            int repeats = (span + size - 1) / size;
            var padded = Vector<T>.Concatenate(new[]
            {
                coordinateGradient.GetSubVector(start, span),
                new Vector<T>((repeats * size) - span),
            });
            result.Add(Engine.ReduceSum(Tensor<T>.FromVector(padded, new[] { repeats, size }), new[] { 0 }, keepDims: false).ToVector());
        }

        return result;
    }

    /// <summary>
    /// Runs the warped inner loop from <paramref name="initParams"/> and returns the adapted parameters, the query
    /// gradient there, and the WarpGrad meta-gradient of the warp parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Flennerhag et al. (ICLR 2020), Eq. 7 / Algorithm 1: the warp meta-objective is
    /// <c>J(P) = sum_k L_query(theta_k - alpha P g_k)</c> over the points <c>theta_k</c> of the task's adaptation
    /// trajectory, each held fixed (stop-gradient), with <c>g_k</c> the support gradient there. So the warp is
    /// meta-learned WITHOUT backpropagating through the inner loop, and its gradient is exact per step:
    /// <c>dJ/dP = -alpha sum_k grad L_query(theta_{k+1}) * g_k</c>, summed over the coordinates each warp parameter
    /// scales. (This replaces a finite-difference loop that re-ran the full inner loop once per warp parameter.)
    /// Gradient clipping of the warped step, when enabled, is treated as the identity in this derivative.
    /// </para>
    /// </remarks>
    private (Vector<T> Adapted, Vector<T> QueryGradient, List<Vector<T>> WarpGradients) WarpedTrajectory(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var taskParams = initParams.Clone();
        var scale = WarpScale(initParams.Length);
        T negativeRate = NumOps.FromDouble(-_warpOptions.InnerLearningRate);
        var coordinateGradient = new Vector<T>(initParams.Length);
        ParamModel.SetParameters(taskParams);
        Vector<T>? queryGradient = null;

        for (int step = 0; step < _warpOptions.AdaptationSteps; step++)
        {
            var rawGradients = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var warpedGradients = ClipGradients(Engine.Multiply(rawGradients, scale));
            taskParams = ApplyInnerGradients(taskParams, warpedGradients, _warpOptions.InnerLearningRate, step == 0);
            ParamModel.SetParameters(taskParams);

            queryGradient = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            coordinateGradient = Engine.Add(
                coordinateGradient, Engine.Multiply(Engine.Multiply(queryGradient, rawGradients), negativeRate));
        }

        if (queryGradient is null)
            queryGradient = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);

        return (taskParams, queryGradient, ReduceToWarpLayers(coordinateGradient));
    }

    /// <summary>
    /// Computes the WarpGrad meta-gradient of each warp layer for one task (see <see cref="WarpedTrajectory"/>).
    /// </summary>
    /// <param name="task">The current task for gradient computation.</param>
    /// <param name="initParams">The initialization parameters before adaptation.</param>
    /// <returns>List of gradient vectors, one per warp-layer.</returns>
    internal List<Vector<T>> ComputeWarpGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> initParams)
    {
        var restore = ParamModel.GetParameters();
        try
        {
            return WarpedTrajectory(task, initParams).WarpGradients;
        }
        finally
        {
            ParamModel.SetParameters(restore);
        }
    }

    /// <summary>
    /// Updates warp-layer parameters using averaged gradients from all tasks in the batch.
    /// </summary>
    /// <param name="warpGradientsList">List of per-task warp gradients.</param>
    /// <remarks>
    /// <para>
    /// Warp-layer updates include:
    /// 1. Average gradients across all tasks in the batch
    /// 2. Apply L2 regularization to keep warp parameters near identity
    /// 3. Update warp parameters using the warp-specific learning rate
    /// </para>
    /// <para><b>For Beginners:</b> This is where the gradient transformation gets improved.
    /// We average the feedback from all tasks ("how should the warp change?") and then
    /// update the warp parameters. We also add a gentle pull toward the "no transformation"
    /// state (regularization) to prevent the warp from becoming too extreme.
    /// </para>
    /// </remarks>
    private void UpdateWarpLayers(List<List<Vector<T>>> warpGradientsList)
    {
        if (warpGradientsList.Count == 0) return;

        T regularization = NumOps.FromDouble(_warpOptions.WarpRegularization);
        T rate = NumOps.FromDouble(_warpOptions.WarpLearningRate);

        for (int w = 0; w < _warpLayerParams.Count; w++)
        {
            var current = _warpLayerParams[w];
            var avgGrad = AverageVectors(warpGradientsList.ConvertAll(taskGrads => taskGrads[w]));

            // Add L2 regularization gradient: reg_grad = lambda * (w - 1.0), pulling warp params toward 1.0 (identity scaling)
            if (_warpOptions.WarpRegularization > 0)
            {
                avgGrad = Engine.Add(
                    avgGrad, Engine.Multiply(Engine.Subtract(current, Engine.Fill(current.Length, NumOps.One)), regularization));
            }

            // Apply gradient clipping, then the update
            avgGrad = ClipGradients(avgGrad);
            _warpLayerParams[w] = Engine.Subtract(current, Engine.Multiply(avgGrad, rate));
        }
    }
}

/// <summary>
/// Adapted model wrapper for WarpGrad inference, providing predictions using task-adapted parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This wrapper holds the adapted parameters from WarpGrad's inner-loop optimization.
/// It delegates predictions to the underlying model with the adapted parameters set.
/// </para>
/// <para><b>For Beginners:</b> After WarpGrad adapts a model to a new task, this wrapper
/// packages the adapted model so you can use it for predictions. It remembers both the
/// original model architecture and the task-specific parameters that were learned.
/// </para>
/// </remarks>
internal class WarpGradModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _adaptedParams;

    /// <summary>
    /// Gets metadata about the adapted WarpGrad model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Stores information about this adapted model,
    /// such as its type and performance metrics.
    /// </para>
    /// </remarks>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Creates a new WarpGrad adapted model wrapper.
    /// </summary>
    /// <param name="model">The base model with its architecture.</param>
    /// <param name="adaptedParams">The task-adapted parameters to use for inference.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a "snapshot" of the model after it's been
    /// adapted to a specific task. The adapted parameters encode what the model learned
    /// from the task's few examples through warped gradient descent.
    /// </para>
    /// </remarks>
    public WarpGradModel(IFullModel<T, TInput, TOutput> model, Vector<T> adaptedParams)
    {
        _model = model;
        _adaptedParams = adaptedParams;
    }

    /// <summary>
    /// Makes predictions using the task-adapted parameters.
    /// </summary>
    /// <param name="input">The input data to make predictions for.</param>
    /// <returns>The model's predictions using task-adapted parameters.</returns>
    /// <remarks>
    /// <para>
    /// Sets the adapted parameters on the model before prediction, ensuring that
    /// the model uses the task-specific parameters learned through warped gradient descent.
    /// </para>
    /// <para><b>For Beginners:</b> This loads the task-specific parameters into the model
    /// and then makes a prediction. The adapted parameters make the model specialized
    /// for the task it was adapted to.
    /// </para>
    /// </remarks>
    public TOutput Predict(TInput input)
    {
        InterfaceGuard.Parameterizable(_model).SetParameters(_adaptedParams);
        return _model.Predict(input);
    }

    /// <summary>
    /// Training is not supported on an already-adapted WarpGrad model.
    /// </summary>
    /// <param name="inputs">Input data (not used).</param>
    /// <param name="targets">Target data (not used).</param>
    /// <remarks>
    /// <para>
    /// WarpGrad adapted models are inference-only. To further train, use the WarpGrad
    /// algorithm's MetaTrain method to update the meta-parameters, then re-adapt.
    /// </para>
    /// <para><b>For Beginners:</b> Once a model has been adapted to a task, you can't
    /// train it further through this interface. If you need to improve the model,
    /// go back to the WarpGrad algorithm and run more meta-training iterations.
    /// </para>
    /// </remarks>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <summary>
    /// Gets metadata about the adapted WarpGrad model.
    /// </summary>
    /// <returns>Model metadata including architecture type and parameter count.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides information about the adapted model,
    /// including that it was adapted using WarpGrad and how many parameters it has.
    /// Useful for logging, debugging, and model management.
    /// </para>
    /// </remarks>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
