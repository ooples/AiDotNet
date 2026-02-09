using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;

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
public class WarpGradAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
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
            options.InnerOptimizer)
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
        var modelParams = MetaModel.GetParameters();
        int paramsPerLayer = modelParams.Length / Math.Max(_warpOptions.NumWarpLayers, 1);

        for (int i = 0; i < _warpOptions.NumWarpLayers; i++)
        {
            int layerDim = _warpOptions.UseDiagonalWarp
                ? paramsPerLayer
                : _warpOptions.WarpLayerHiddenDim;

            var warpParams = new Vector<T>(layerDim);

            // Initialize near identity: small random perturbation around 1.0 (identity scaling)
            for (int j = 0; j < layerDim; j++)
            {
                double initValue = 1.0 + (RandomGenerator.NextDouble() - 0.5) * 2.0 * _warpOptions.WarpInitScale;
                warpParams[j] = NumOps.FromDouble(initValue);
            }

            _warpLayerParams.Add(warpParams);
        }
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

        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Clone model and adapt with warped gradients
            var taskParams = new Vector<T>(initParams.Length);
            for (int i = 0; i < initParams.Length; i++)
            {
                taskParams[i] = initParams[i];
            }

            MetaModel.SetParameters(taskParams);

            // Inner loop: adapt with warped gradients
            for (int step = 0; step < _warpOptions.AdaptationSteps; step++)
            {
                var rawGradients = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var warpedGradients = ApplyWarpLayers(rawGradients);
                warpedGradients = ClipGradients(warpedGradients);

                taskParams = ApplyGradients(taskParams, warpedGradients, _warpOptions.InnerLearningRate);
                MetaModel.SetParameters(taskParams);
            }

            // Evaluate on query set
            var queryLoss = ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradients for initialization parameters
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGradients.Add(ClipGradients(metaGrad));

            // Compute warp-layer gradients using finite differences on warp parameters
            var warpGrads = ComputeWarpGradients(task, initParams);
            warpGradientsList.Add(warpGrads);
        }

        // Restore original params before applying outer update
        MetaModel.SetParameters(initParams);

        // Average and apply meta-gradients to initialization
        if (metaGradients.Count > 0)
        {
            var avgMetaGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgMetaGrad, _warpOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
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
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Start from meta-learned initialization
        var adaptedParams = new Vector<T>(MetaModel.GetParameters().Length);
        var initParams = MetaModel.GetParameters();
        for (int i = 0; i < initParams.Length; i++)
        {
            adaptedParams[i] = initParams[i];
        }

        MetaModel.SetParameters(adaptedParams);

        // Inner loop: adapt with warped gradients
        for (int step = 0; step < _warpOptions.AdaptationSteps; step++)
        {
            var rawGradients = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var warpedGradients = ApplyWarpLayers(rawGradients);
            warpedGradients = ClipGradients(warpedGradients);

            adaptedParams = ApplyGradients(adaptedParams, warpedGradients, _warpOptions.InnerLearningRate);
            MetaModel.SetParameters(adaptedParams);
        }

        // Return adapted model wrapped for inference
        return new WarpGradModel<T, TInput, TOutput>(MetaModel, adaptedParams);
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
    {
        var warped = new Vector<T>(rawGradients.Length);
        for (int i = 0; i < rawGradients.Length; i++)
        {
            warped[i] = rawGradients[i];
        }

        int paramsPerLayer = rawGradients.Length / Math.Max(_warpOptions.NumWarpLayers, 1);

        for (int w = 0; w < _warpLayerParams.Count; w++)
        {
            var warpParams = _warpLayerParams[w];
            int startIdx = w * paramsPerLayer;
            int endIdx = Math.Min(startIdx + paramsPerLayer, rawGradients.Length);

            for (int i = startIdx; i < endIdx; i++)
            {
                int warpIdx = (i - startIdx) % warpParams.Length;
                // Element-wise scaling: warped_g = g * warp_param
                warped[i] = NumOps.Multiply(warped[i], warpParams[warpIdx]);
            }
        }

        return warped;
    }

    /// <summary>
    /// Computes gradients for warp-layer parameters using finite differences.
    /// </summary>
    /// <param name="task">The current task for gradient computation.</param>
    /// <param name="initParams">The initialization parameters before adaptation.</param>
    /// <returns>List of gradient vectors, one per warp-layer.</returns>
    /// <remarks>
    /// <para>
    /// Warp-layer gradients measure how changing the gradient preconditioning affects
    /// the final query set loss after inner-loop adaptation. We use finite differences
    /// because the warp parameters appear inside the inner loop optimization trajectory.
    /// </para>
    /// <para><b>For Beginners:</b> This figures out how to improve the gradient transformation.
    /// For each warp parameter, we ask: "If I change this slightly, does the adapted model
    /// perform better or worse on the query set?" This tells us which direction to adjust
    /// the warp parameters.
    /// </para>
    /// </remarks>
    private List<Vector<T>> ComputeWarpGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> initParams)
    {
        var warpGrads = new List<Vector<T>>();
        double epsilon = 1e-5;

        for (int w = 0; w < _warpLayerParams.Count; w++)
        {
            var warpParams = _warpLayerParams[w];
            var grad = new Vector<T>(warpParams.Length);

            // Compute baseline loss with current warp
            double baselineLoss = NumOps.ToDouble(EvaluateWithWarp(task, initParams));

            // Perturb each warp parameter
            for (int j = 0; j < warpParams.Length; j++)
            {
                T original = warpParams[j];
                warpParams[j] = NumOps.Add(original, NumOps.FromDouble(epsilon));

                double perturbedLoss = NumOps.ToDouble(EvaluateWithWarp(task, initParams));
                grad[j] = NumOps.FromDouble((perturbedLoss - baselineLoss) / epsilon);

                warpParams[j] = original; // Restore
            }

            warpGrads.Add(grad);
        }

        return warpGrads;
    }

    /// <summary>
    /// Evaluates a task with the current warp configuration by running inner-loop adaptation.
    /// </summary>
    /// <param name="task">The task to evaluate.</param>
    /// <param name="initParams">Starting parameters.</param>
    /// <returns>Query set loss after adaptation.</returns>
    /// <remarks>
    /// <para>
    /// This is a helper that runs the full inner-loop adaptation with the current warp
    /// configuration and returns the query loss. Used by warp gradient computation
    /// to measure the effect of warp parameter changes.
    /// </para>
    /// <para><b>For Beginners:</b> This simulates the entire adaptation process with
    /// specific warp settings and returns how well the adapted model performs.
    /// It's used to figure out whether a change to the warp parameters helps or hurts.
    /// </para>
    /// </remarks>
    private T EvaluateWithWarp(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var taskParams = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
        {
            taskParams[i] = initParams[i];
        }

        MetaModel.SetParameters(taskParams);

        for (int step = 0; step < _warpOptions.AdaptationSteps; step++)
        {
            var rawGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var warpedGrad = ApplyWarpLayers(rawGrad);
            taskParams = ApplyGradients(taskParams, warpedGrad, _warpOptions.InnerLearningRate);
            MetaModel.SetParameters(taskParams);
        }

        return ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
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

        for (int w = 0; w < _warpLayerParams.Count; w++)
        {
            var avgGrad = new Vector<T>(_warpLayerParams[w].Length);

            // Average across tasks
            foreach (var taskGrads in warpGradientsList)
            {
                if (w < taskGrads.Count)
                {
                    for (int j = 0; j < avgGrad.Length; j++)
                    {
                        avgGrad[j] = NumOps.Add(avgGrad[j], taskGrads[w][j]);
                    }
                }
            }

            double scale = 1.0 / warpGradientsList.Count;
            for (int j = 0; j < avgGrad.Length; j++)
            {
                avgGrad[j] = NumOps.Multiply(avgGrad[j], NumOps.FromDouble(scale));
            }

            // Add L2 regularization gradient: reg_grad = lambda * (w - 1.0)
            // This pulls warp params toward 1.0 (identity scaling)
            if (_warpOptions.WarpRegularization > 0)
            {
                for (int j = 0; j < avgGrad.Length; j++)
                {
                    T regGrad = NumOps.Multiply(
                        NumOps.FromDouble(_warpOptions.WarpRegularization),
                        NumOps.Subtract(_warpLayerParams[w][j], NumOps.FromDouble(1.0)));
                    avgGrad[j] = NumOps.Add(avgGrad[j], regGrad);
                }
            }

            // Apply update
            double lr = _warpOptions.WarpLearningRate;
            for (int j = 0; j < _warpLayerParams[w].Length; j++)
            {
                _warpLayerParams[w][j] = NumOps.Subtract(
                    _warpLayerParams[w][j],
                    NumOps.Multiply(NumOps.FromDouble(lr), avgGrad[j]));
            }
        }
    }

    /// <summary>
    /// Computes the element-wise average of a list of vectors.
    /// </summary>
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
        _model.SetParameters(_adaptedParams);
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
    public void Train(TInput inputs, TOutput targets)
    {
        // Adapted models are frozen - use WarpGrad.MetaTrain for further training
    }

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
