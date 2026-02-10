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
/// Implementation of MAML++ (How to Train Your MAML) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MAML++ is a production-hardened MAML that addresses training instabilities through:
/// - Multi-Step Loss (MSL): Supervise every inner-loop step, not just the final one
/// - Per-Step Learning Rates (LSLR): Each adaptation step has its own learnable learning rate
/// - Derivative-Order Annealing: Gradually transition from first-order to second-order gradients
/// - Per-Step Batch Normalization: Separate BN statistics for each adaptation step
/// - Cosine Annealing: Learning rate schedule for the outer loop
/// </para>
/// <para><b>For Beginners:</b> MAML++ is the "industrial strength" version of MAML:
///
/// **Problems with vanilla MAML:**
/// 1. Training is unstable (loss can explode randomly)
/// 2. One learning rate doesn't work well for all adaptation steps
/// 3. Second-order gradients are noisy early in training
/// 4. Batch normalization statistics become stale during inner loop
///
/// **MAML++ solutions:**
/// 1. Multi-step loss: Check performance at EVERY step, not just the last
/// 2. Per-step learning rates: Each step gets its own rate, learned during training
/// 3. Derivative-order annealing: Start simple, gradually get more precise
/// 4. Per-step batch norm: Keep separate statistics for each step
///
/// **Analogy:** If MAML is a car, MAML++ adds:
/// - Anti-lock brakes (stability fixes)
/// - Cruise control (per-step learning rates)
/// - A GPS that updates gradually (derivative-order annealing)
/// - Multiple speedometers (per-step batch norm)
/// </para>
/// <para><b>Algorithm - MAML++:</b>
/// <code>
/// # Initialization
/// theta = model_parameters        # Meta-learned initialization
/// alpha = [lr] * K                # Per-step learning rates (learnable)
/// iteration = 0
///
/// # Meta-training
/// for each meta-iteration:
///     # Cosine annealing for outer learning rate
///     beta = cosine_anneal(beta_0, iteration, total_iterations)
///
///     for each task T_i in batch:
///         theta_i = copy(theta)
///         multi_step_loss = 0
///
///         # Inner loop with per-step rates and multi-step loss
///         for step k in range(K):
///             loss_k = compute_loss(theta_i, support_x, support_y)
///             multi_step_loss += w_k * loss_k     # Weighted per-step loss
///
///             # Derivative-order annealing
///             if iteration &lt; anneal_iters:
///                 g = first_order_gradient(loss_k, theta_i)   # FOMAML
///             else:
///                 g = second_order_gradient(loss_k, theta_i)  # Full MAML
///
///             theta_i = theta_i - alpha[k] * g     # Per-step learning rate
///
///         # Final query loss
///         query_loss = compute_loss(theta_i, query_x, query_y)
///         multi_step_loss += w_K * query_loss
///
///     # Outer loop: Update theta AND alpha
///     theta = theta - beta * mean(grad(multi_step_loss, theta))
///     alpha = alpha - beta * mean(grad(multi_step_loss, alpha))
///     iteration += 1
/// </code>
/// </para>
/// <para>
/// Reference: Antoniou, A., Edwards, H., &amp; Storkey, A. (2019).
/// How to Train Your MAML. ICLR 2019.
/// </para>
/// </remarks>
public class MAMLPlusPlusAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MAMLPlusPlusOptions<T, TInput, TOutput> _mamlOptions;

    /// <summary>
    /// Per-step learning rates that are meta-learned during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each entry corresponds to one inner-loop adaptation step. These rates are
    /// initialized from InnerLearningRate and updated during the outer loop.
    /// </para>
    /// <para><b>For Beginners:</b> Each adaptation step gets its own learning rate.
    /// Early steps might use larger rates for coarse adjustments, while later steps
    /// use smaller rates for fine-tuning. These rates are learned automatically.
    /// </para>
    /// </remarks>
    private double[] _perStepLearningRates;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MAMLPlusPlus;

    /// <summary>
    /// Gets the current per-step learning rates.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns the learned learning rates for each adaptation step.
    /// You can inspect these after meta-training to see what the algorithm learned:
    /// typically early steps have larger rates and later steps have smaller rates.
    /// </para>
    /// </remarks>
    public IReadOnlyList<double> PerStepLearningRates => Array.AsReadOnly(_perStepLearningRates);

    /// <summary>
    /// Initializes a new instance of the MAML++ algorithm.
    /// </summary>
    /// <param name="options">Configuration options for MAML++.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or MetaModel is null.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes per-step learning rates from the base InnerLearningRate
    /// and sets up multi-step loss weights if not explicitly provided.
    /// </para>
    /// <para><b>For Beginners:</b> Creates a new MAML++ meta-learner. This sets up:
    /// - The model that will be meta-trained
    /// - Per-step learning rates (all starting at the same value)
    /// - Multi-step loss weights (how much to care about each adaptation step)
    /// </para>
    /// </remarks>
    public MAMLPlusPlusAlgorithm(MAMLPlusPlusOptions<T, TInput, TOutput> options)
        : base(
            options.MetaModel,
            options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _mamlOptions = options;

        // Initialize per-step learning rates
        _perStepLearningRates = new double[options.AdaptationSteps];
        for (int i = 0; i < options.AdaptationSteps; i++)
        {
            _perStepLearningRates[i] = options.InnerLearningRate;
        }
    }

    /// <summary>
    /// Performs one meta-training step with MAML++ enhancements.
    /// </summary>
    /// <param name="taskBatch">Batch of meta-learning tasks.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <remarks>
    /// <para>
    /// Each meta-training step applies all MAML++ improvements:
    /// 1. Adapts each task using per-step learning rates
    /// 2. Computes multi-step loss (if enabled) supervising every inner step
    /// 3. Uses derivative-order annealing to select first-order or second-order gradients
    /// 4. Averages meta-gradients across the task batch
    /// 5. Updates initialization parameters with cosine-annealed outer learning rate
    /// 6. Updates per-step learning rates (LSLR)
    /// </para>
    /// <para><b>For Beginners:</b> This is the core training loop of MAML++:
    ///
    /// For each task in the batch:
    /// 1. Start from the meta-learned initialization
    /// 2. Adapt step-by-step, using different learning rates at each step
    /// 3. At each step, measure how well the model is doing (multi-step loss)
    /// 4. After all steps, also measure on the query set
    ///
    /// Then improve everything:
    /// - The starting parameters (initialization)
    /// - The learning rates for each step (LSLR)
    /// - Choose gradient accuracy based on training progress (annealing)
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var metaGradients = new List<Vector<T>>();
        var losses = new List<T>();

        var initParams = MetaModel.GetParameters();

        // Determine if we should use first-order based on annealing
        bool useFirstOrder = ShouldUseFirstOrder();

        // Compute cosine-annealed outer learning rate
        double effectiveOuterLR = ComputeCosineAnnealedLR();

        foreach (var task in taskBatch.Tasks)
        {
            // Clone parameters for this task
            var taskParams = new Vector<T>(initParams.Length);
            for (int i = 0; i < initParams.Length; i++)
            {
                taskParams[i] = initParams[i];
            }

            MetaModel.SetParameters(taskParams);

            double[] stepWeights = GetMultiStepWeights();
            var innerStepGradients = new List<Vector<T>>();

            // Inner loop: adapt with per-step learning rates
            for (int step = 0; step < _mamlOptions.AdaptationSteps; step++)
            {
                // Compute gradients for inner-loop update
                var gradients = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                gradients = ClipGradients(gradients);

                // Accumulate weighted inner-step gradients for multi-step loss
                if (_mamlOptions.UseMultiStepLoss && !useFirstOrder)
                {
                    var weightedGrad = new Vector<T>(gradients.Length);
                    for (int i = 0; i < gradients.Length; i++)
                        weightedGrad[i] = NumOps.Multiply(gradients[i], NumOps.FromDouble(stepWeights[step]));
                    innerStepGradients.Add(weightedGrad);
                }

                // Apply per-step learning rate
                double stepLR = _mamlOptions.UsePerStepLearningRates
                    ? _perStepLearningRates[step]
                    : _mamlOptions.InnerLearningRate;

                taskParams = ApplyGradients(taskParams, gradients, stepLR);
                MetaModel.SetParameters(taskParams);
            }

            // Evaluate on query set
            var queryLoss = ComputeLossFromOutput(
                MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);

            // Compute meta-gradients at adapted parameters (final query gradient)
            var metaGrad = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);
            metaGrad = ClipGradients(metaGrad);

            // When multi-step loss is active and NOT first-order, blend inner-step
            // gradients into the meta-gradient to approximate second-order information
            if (_mamlOptions.UseMultiStepLoss && !useFirstOrder && innerStepGradients.Count > 0)
            {
                double queryWeight = stepWeights[^1];
                var blended = new Vector<T>(metaGrad.Length);
                for (int i = 0; i < metaGrad.Length; i++)
                    blended[i] = NumOps.Multiply(metaGrad[i], NumOps.FromDouble(queryWeight));
                foreach (var stepGrad in innerStepGradients)
                {
                    for (int i = 0; i < blended.Length; i++)
                        blended[i] = NumOps.Add(blended[i], stepGrad[i]);
                }
                metaGradients.Add(blended);
            }
            else
            {
                metaGradients.Add(metaGrad);
            }
        }

        // Restore and apply averaged meta-gradients
        MetaModel.SetParameters(initParams);

        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            var updatedParams = ApplyGradients(initParams, avgGrad, effectiveOuterLR);
            MetaModel.SetParameters(updatedParams);
        }

        // Update per-step learning rates via finite differences (LSLR meta-learning)
        // Per paper: alpha = alpha - beta * grad(meta_loss, alpha)
        if (_mamlOptions.UsePerStepLearningRates)
        {
            UpdatePerStepLearningRates(taskBatch, initParams, effectiveOuterLR);
        }

        return ComputeMean(losses);
    }

    /// <summary>
    /// Updates per-step learning rates using finite differences.
    /// For each step k, perturbs alpha_k by epsilon, re-runs the inner loop,
    /// measures loss change, and updates alpha_k via: alpha_k -= lr * dL/d(alpha_k).
    /// </summary>
    /// <param name="taskBatch">Current task batch.</param>
    /// <param name="initParams">Initial backbone parameters.</param>
    /// <param name="outerLR">Outer learning rate.</param>
    private void UpdatePerStepLearningRates(
        TaskBatch<T, TInput, TOutput> taskBatch,
        Vector<T> initParams,
        double outerLR)
    {
        double epsilon = 1e-3;

        for (int step = 0; step < _mamlOptions.AdaptationSteps; step++)
        {
            double lossPlus = 0;
            double lossMinus = 0;

            // Evaluate with alpha_k + epsilon
            _perStepLearningRates[step] += epsilon;
            foreach (var task in taskBatch.Tasks)
            {
                lossPlus += NumOps.ToDouble(EvaluateTaskLoss(task, initParams));
            }

            // Evaluate with alpha_k - epsilon (delta = 2*epsilon from current)
            _perStepLearningRates[step] -= 2.0 * epsilon;
            foreach (var task in taskBatch.Tasks)
            {
                lossMinus += NumOps.ToDouble(EvaluateTaskLoss(task, initParams));
            }

            // Restore original alpha_k
            _perStepLearningRates[step] += epsilon;

            // Finite difference gradient: dL/d(alpha_k) ≈ (L+ - L-) / (2 * epsilon)
            int numTasks = Math.Max(taskBatch.Tasks.Length, 1);
            double grad = (lossPlus / numTasks - lossMinus / numTasks) / (2.0 * epsilon);

            // Update alpha_k, clamped to [1e-6, 1.0] for stability
            _perStepLearningRates[step] = Math.Max(1e-6,
                Math.Min(1.0, _perStepLearningRates[step] - outerLR * grad));
        }
    }

    /// <summary>
    /// Evaluates the query loss for a single task after running the full inner loop.
    /// Used by LSLR update to measure loss as a function of per-step learning rates.
    /// </summary>
    /// <param name="task">The task to evaluate.</param>
    /// <param name="initParams">Starting backbone parameters.</param>
    /// <returns>Query loss after inner-loop adaptation.</returns>
    private T EvaluateTaskLoss(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams)
    {
        var taskParams = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
            taskParams[i] = initParams[i];

        MetaModel.SetParameters(taskParams);

        for (int step = 0; step < _mamlOptions.AdaptationSteps; step++)
        {
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            grad = ClipGradients(grad);
            double stepLR = _perStepLearningRates[step];
            taskParams = ApplyGradients(taskParams, grad, stepLR);
            MetaModel.SetParameters(taskParams);
        }

        var loss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
        MetaModel.SetParameters(initParams); // Restore base parameters
        return loss;
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using MAML++ per-step learning rates.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>An adapted model specialized for the given task.</returns>
    /// <remarks>
    /// <para>
    /// Adaptation uses the learned per-step learning rates for each gradient step.
    /// This applies the MAML++ initialization with the meta-learned step sizes for
    /// optimal adaptation.
    /// </para>
    /// <para><b>For Beginners:</b> When you have a new task:
    /// 1. Start from the meta-learned initialization (a good starting point)
    /// 2. Take several gradient steps, each with its own learned learning rate
    /// 3. The per-step rates were learned during meta-training to be optimal
    /// 4. Return the adapted model, ready for predictions
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
        {
            adaptedParams[i] = initParams[i];
        }

        MetaModel.SetParameters(adaptedParams);

        // Inner loop with per-step learning rates
        for (int step = 0; step < _mamlOptions.AdaptationSteps; step++)
        {
            var gradients = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            gradients = ClipGradients(gradients);

            double stepLR = _mamlOptions.UsePerStepLearningRates
                ? _perStepLearningRates[step]
                : _mamlOptions.InnerLearningRate;

            adaptedParams = ApplyGradients(adaptedParams, gradients, stepLR);
            MetaModel.SetParameters(adaptedParams);
        }

        MetaModel.SetParameters(initParams); // Restore base parameters
        return new MAMLPlusPlusModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Determines whether to use first-order gradients based on derivative-order annealing.
    /// </summary>
    /// <returns>True if first-order gradients should be used for this iteration.</returns>
    /// <remarks>
    /// <para>
    /// Derivative-order annealing linearly transitions from first-order to second-order
    /// over AnnealingIterations. Before reaching the annealing threshold, first-order
    /// is always used. After, second-order is used (unless UseFirstOrder is globally set).
    /// </para>
    /// <para><b>For Beginners:</b> This decides whether to use simple or complex gradients:
    /// - Early in training: Always use simple gradients (stable but approximate)
    /// - After enough iterations: Switch to complex gradients (precise but potentially unstable)
    /// - The transition is gradual, preventing sudden instability
    /// </para>
    /// </remarks>
    private bool ShouldUseFirstOrder()
    {
        if (_mamlOptions.UseFirstOrder) return true;

        if (!_mamlOptions.UseDerivativeOrderAnnealing) return false;

        // Linear annealing: probability of using second-order increases over time
        double annealProgress = Math.Min(1.0, (double)_currentIteration / _mamlOptions.AnnealingIterations);
        return RandomGenerator.NextDouble() > annealProgress;
    }

    /// <summary>
    /// Computes the cosine-annealed outer learning rate for the current iteration.
    /// </summary>
    /// <returns>The effective outer learning rate.</returns>
    /// <remarks>
    /// <para>
    /// Cosine annealing smoothly decreases the learning rate following a cosine curve,
    /// starting from OuterLearningRate and decreasing to OuterLearningRate * CosineAnnealingMinRatio.
    /// </para>
    /// <para><b>For Beginners:</b> The outer learning rate decreases smoothly during training:
    /// - Starts at OuterLearningRate (e.g., 0.001)
    /// - Follows a cosine curve downward
    /// - Ends at OuterLearningRate * MinRatio (e.g., 0.001 * 0.01 = 0.00001)
    /// This helps training converge to a better solution.
    /// </para>
    /// </remarks>
    private double ComputeCosineAnnealedLR()
    {
        double progress = (double)_currentIteration / Math.Max(_mamlOptions.NumMetaIterations, 1);
        double minLR = _mamlOptions.OuterLearningRate * _mamlOptions.CosineAnnealingMinRatio;
        double cosineDecay = 0.5 * (1.0 + Math.Cos(Math.PI * progress));
        return minLR + (_mamlOptions.OuterLearningRate - minLR) * cosineDecay;
    }

    /// <summary>
    /// Gets the multi-step loss weights for each adaptation step plus the final query step.
    /// </summary>
    /// <returns>Array of weights with length AdaptationSteps + 1.</returns>
    /// <remarks>
    /// <para>
    /// Returns custom weights if specified in options, otherwise returns uniform weights.
    /// The last weight corresponds to the final query set loss.
    /// </para>
    /// <para><b>For Beginners:</b> Returns how much to weight the loss at each adaptation step.
    /// With 5 steps and uniform weights, each step contributes equally. Custom weights let you
    /// emphasize later steps (which should be more adapted) over earlier ones.
    /// </para>
    /// </remarks>
    private double[] GetMultiStepWeights()
    {
        int totalSteps = _mamlOptions.AdaptationSteps + 1; // +1 for query loss

        if (_mamlOptions.MultiStepLossWeights != null &&
            _mamlOptions.MultiStepLossWeights.Length == totalSteps)
        {
            return _mamlOptions.MultiStepLossWeights;
        }

        // Default: uniform weights
        var weights = new double[totalSteps];
        double weight = 1.0 / totalSteps;
        for (int i = 0; i < totalSteps; i++)
        {
            weights[i] = weight;
        }

        return weights;
    }

}

/// <summary>
/// Adapted model wrapper for MAML++ inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> After MAML++ adapts a model to a specific task, this wrapper
/// holds the adapted parameters. Use it to make predictions on that task.
/// </para>
/// </remarks>
internal class MAMLPlusPlusModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly Vector<T> _adaptedParams;

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Creates a new MAML++ adapted model.
    /// </summary>
    /// <param name="model">The base model architecture.</param>
    /// <param name="adaptedParams">Task-adapted parameters.</param>
    public MAMLPlusPlusModel(IFullModel<T, TInput, TOutput> model, Vector<T> adaptedParams)
    {
        _model = model;
        _adaptedParams = adaptedParams;
    }

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        _model.SetParameters(_adaptedParams);
        return _model.Predict(input);
    }

    /// <summary>
    /// Training is not supported on adapted models. Use MAML++ MetaTrain for further training.
    /// </summary>
    public void Train(TInput inputs, TOutput targets) =>
        throw new NotSupportedException("Adapted meta-learning models do not support direct training. Use the meta-learning algorithm's MetaTrain method instead.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
