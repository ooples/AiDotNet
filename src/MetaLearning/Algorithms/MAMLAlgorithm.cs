using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the MAML (Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// MAML (Model-Agnostic Meta-Learning) is a meta-learning algorithm that trains models
/// to be easily fine-tunable. It learns initial parameters such that a small number of
/// gradient steps on a new task will lead to good performance.
/// </para>
/// <para>
/// Key features:
/// - Model-agnostic: works with any model trainable with gradient descent
/// - Learns good initialization rather than learning a fixed feature extractor
/// - Enables few-shot learning with just 1-5 examples per class
/// </para>
/// <para>
/// <b>For Beginners:</b> MAML is like teaching someone how to learn quickly.
///
/// Normal machine learning: Train a model for one specific task
/// MAML: Train a model to be easily trainable for many different tasks
///
/// It's like learning how to learn - by practicing on many tasks, the model
/// learns what kind of parameters make it easy to adapt to new tasks quickly.
/// </para>
/// <para>
/// Reference: Finn, C., Abbeel, P., &amp; Levine, S. (2017).
/// Model-agnostic meta-learning for fast adaptation of deep networks.
/// </para>
/// </remarks>
public class MAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MAMLOptions<T, TInput, TOutput> _mamlOptions;

    /// <summary>
    /// Initializes a new instance of the MAMLAlgorithm class.
    /// </summary>
    /// <param name="options">MAML configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create MAML with minimal configuration (uses all defaults)
    /// var options = new MAMLOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var maml = new MAMLAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create MAML with custom configuration
    /// var options = new MAMLOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     LossFunction = new CrossEntropyLoss&lt;double&gt;(),
    ///     InnerLearningRate = 0.01,
    ///     OuterLearningRate = 0.001,
    ///     AdaptationSteps = 5,
    ///     UseFirstOrderApproximation = true
    /// };
    /// var maml = new MAMLAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public MAMLAlgorithm(MAMLOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _mamlOptions = options;
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MAML"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as MAML (Model-Agnostic Meta-Learning),
    /// which is useful for serialization, logging, and algorithm-specific handling.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MAML;

    /// <summary>
    /// Performs one meta-training step using MAML's bi-level optimization.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// MAML meta-training consists of two nested optimization loops:
    /// </para>
    /// <para>
    /// <b>Inner Loop (Task Adaptation):</b>
    /// For each task in the batch:
    /// 1. Clone the meta-model with current meta-parameters
    /// 2. Perform K gradient descent steps on the task's support set
    /// 3. Evaluate the adapted model on the task's query set
    /// </para>
    /// <para>
    /// <b>Outer Loop (Meta-Update):</b>
    /// 1. Compute meta-gradients based on query set performance
    /// 2. Average meta-gradients across all tasks in the batch
    /// 3. Apply gradient clipping if configured
    /// 4. Update meta-parameters using the averaged meta-gradients
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each call to this method makes the model slightly better
    /// at learning new tasks quickly. The returned loss value should decrease over time.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate meta-gradients across all tasks
        Vector<T>? accumulatedMetaGradients = null;
        T totalMetaLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Clone the meta model for this task
            var taskModel = CloneModel();
            var initialParams = taskModel.GetParameters();

            // Inner loop: Adapt to the task using support set
            var adaptedParams = InnerLoopAdaptation(taskModel, task);
            taskModel.SetParameters(adaptedParams);

            // Compute meta-loss on query set
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T metaLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute meta-gradients (gradients with respect to initial parameters)
            var taskMetaGradients = ComputeMetaGradients(initialParams, task);

            // Accumulate meta-gradients
            if (accumulatedMetaGradients == null)
            {
                accumulatedMetaGradients = taskMetaGradients;
            }
            else
            {
                for (int i = 0; i < accumulatedMetaGradients.Length; i++)
                {
                    accumulatedMetaGradients[i] = NumOps.Add(accumulatedMetaGradients[i], taskMetaGradients[i]);
                }
            }
        }

        if (accumulatedMetaGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average the meta-gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < accumulatedMetaGradients.Length; i++)
        {
            accumulatedMetaGradients[i] = NumOps.Divide(accumulatedMetaGradients[i], batchSizeT);
        }

        // Apply gradient clipping if configured
        if (_mamlOptions.GradientClipThreshold.HasValue && _mamlOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedMetaGradients = ClipGradients(accumulatedMetaGradients, _mamlOptions.GradientClipThreshold.Value);
        }

        // Outer loop: Update meta-parameters
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = ApplyGradients(currentMetaParams, accumulatedMetaGradients, _mamlOptions.OuterLearningRate);
        MetaModel.SetParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(totalMetaLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using MAML's inner loop optimization.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been fine-tuned to the given task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// This is where MAML's "learning to learn" capability shines. The meta-learned
    /// initialization allows the model to quickly adapt to new tasks with just a few
    /// gradient steps on a small support set.
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// 1. Clone the meta-model to preserve the learned initialization
    /// 2. Perform K gradient descent steps on the task's support set
    /// 3. Return the adapted model, ready for inference on new examples
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After meta-training, call this method when you have a new
    /// task with a few labeled examples. The returned model will be specialized for
    /// that task and ready to make predictions.
    /// </para>
    /// <para>
    /// The number of gradient steps is controlled by <see cref="MAMLOptions{T, TInput, TOutput}.AdaptationSteps"/>.
    /// Typically 1-10 steps are sufficient thanks to the good initialization learned during meta-training.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone the meta model
        var adaptedModel = CloneModel();

        // Perform inner loop adaptation
        var adaptedParameters = InnerLoopAdaptation(adaptedModel, task);
        adaptedModel.SetParameters(adaptedParameters);

        return adaptedModel;
    }

    /// <summary>
    /// Performs the inner loop adaptation to a specific task.
    /// </summary>
    /// <param name="model">The model to adapt.</param>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>The adapted parameters.</returns>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Perform K gradient steps on the support set
        for (int step = 0; step < _mamlOptions.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Apply gradients with inner learning rate
            parameters = ApplyGradients(parameters, gradients, _mamlOptions.InnerLearningRate);
            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Computes meta-gradients for the outer loop update.
    /// </summary>
    /// <param name="initialParams">The initial parameters before adaptation.</param>
    /// <param name="task">The task to compute meta-gradients for.</param>
    /// <returns>The meta-gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method supports both full second-order MAML and first-order MAML (FOMAML):
    /// </para>
    /// <para>
    /// <b>Full Second-Order MAML:</b>
    /// When <see cref="MAMLOptions{T, TInput, TOutput}.UseFirstOrderApproximation"/> is false and the model
    /// implements <see cref="ISecondOrderGradientComputable{T, TInput, TOutput}"/>, true second-order
    /// gradients are computed by backpropagating through the inner loop adaptation process.
    /// This computes dL/dtheta = dL/dtheta' * dtheta'/dtheta using Hessian-vector products.
    /// </para>
    /// <para>
    /// <b>FOMAML Approximation:</b>
    /// When UseFirstOrderApproximation is true OR the model doesn't support second-order gradients,
    /// the gradient is computed w.r.t. adapted parameters, treating them as constants.
    /// This ignores the dependency dtheta'/dtheta but is much more efficient and
    /// performs nearly as well in practice (Finn et al., 2017).
    /// </para>
    /// <para>
    /// <b>Computational Cost:</b>
    /// - Full MAML: O(K * P^2) where K is adaptation steps and P is parameter count
    /// - FOMAML: O(K * P) - linear in parameters, much more efficient
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Full MAML asks: "How should I change my starting point to improve learning?"
    /// FOMAML asks: "How should I change my ending point to improve performance?"
    /// FOMAML is simpler but works almost as well, so it's the default recommendation
    /// for large models.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeMetaGradients(Vector<T> initialParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Clone meta model and set initial parameters
        var model = CloneModel();
        model.SetParameters(initialParams);

        // Check if we should use full second-order MAML
        bool useSecondOrder = !_mamlOptions.UseFirstOrderApproximation &&
                              model is ISecondOrderGradientComputable<T, TInput, TOutput>;

        if (useSecondOrder)
        {
            // Full MAML: Compute second-order gradients through adaptation process
            // Build adaptation steps list for second-order gradient computation
            var adaptationSteps = BuildAdaptationStepsList(task);

            // Use the base class method which leverages ISecondOrderGradientComputable
            return ComputeSecondOrderGradients(
                model,
                adaptationSteps,
                task.QueryInput,
                task.QueryOutput,
                NumOps.FromDouble(_mamlOptions.InnerLearningRate));
        }
        else
        {
            // FOMAML: First-order approximation
            // Adapt to the task
            var adaptedParams = InnerLoopAdaptation(model, task);
            model.SetParameters(adaptedParams);

            // Compute gradient w.r.t. adapted parameters on query set
            return ComputeGradients(model, task.QueryInput, task.QueryOutput);
        }
    }

    /// <summary>
    /// Builds the list of adaptation steps for second-order gradient computation.
    /// </summary>
    /// <param name="task">The task containing support set data.</param>
    /// <returns>List of (input, target) tuples representing each adaptation step.</returns>
    /// <remarks>
    /// <para>
    /// For second-order MAML, we need to record each step of the inner loop adaptation
    /// so we can backpropagate through it. This method creates the step list that will
    /// be passed to <see cref="ISecondOrderGradientComputable{T, TInput, TOutput}.ComputeSecondOrderGradients"/>.
    /// </para>
    /// <para>
    /// The number of steps equals <see cref="MAMLOptions{T, TInput, TOutput}.AdaptationSteps"/>.
    /// Each step uses the same support set data (full-batch inner loop).
    /// </para>
    /// </remarks>
    private List<(TInput input, TOutput target)> BuildAdaptationStepsList(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var steps = new List<(TInput input, TOutput target)>();

        // Each adaptation step uses the support set
        for (int step = 0; step < _mamlOptions.AdaptationSteps; step++)
        {
            steps.Add((task.SupportInput, task.SupportOutput));
        }

        return steps;
    }
}
