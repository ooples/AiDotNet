using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of CAVIA (Fast Context Adaptation via Meta-Learning) for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// CAVIA separates model parameters into shared body parameters and task-specific context
/// parameters. Only the small context vector is adapted during the inner loop, making CAVIA
/// significantly faster than full MAML while achieving comparable performance.
/// </para>
/// <para><b>For Beginners:</b> CAVIA is a smarter version of MAML that adapts faster:
///
/// **How it works:**
/// 1. The model has two kinds of parameters:
///    - Body parameters (shared across all tasks) - these are the model's "core skills"
///    - Context parameters (adapted per task) - these are task-specific adjustments
/// 2. For a new task, only context parameters are updated (much faster!)
/// 3. The body parameters improve over time across all tasks
/// 4. Context is a small vector concatenated with/added to the input
///
/// **Simple example:**
/// - Body params: How to recognize shapes, edges, textures (shared knowledge)
/// - Context params: "Right now I'm looking at animals" vs "Right now I'm looking at vehicles"
/// - For each new task, only adjust what kind of thing you're looking at
/// - Your fundamental perception skills stay the same
///
/// **Why it's better than MAML:**
/// - MAML adapts ALL parameters (expensive, O(P) where P = total params)
/// - CAVIA adapts only context (cheap, O(C) where C = context_dim, C &lt;&lt; P)
/// - Less prone to meta-overfitting (fewer adapted parameters)
/// - Mathematically cleaner separation of shared vs. task-specific knowledge
/// </para>
/// <para><b>Algorithm - CAVIA:</b>
/// <code>
/// # Initialization
/// phi = model_parameters          # Shared body parameters
/// psi_dim = context_dimension     # Size of context vector (e.g., 100)
///
/// # Meta-training
/// for each meta-iteration:
///     for each task T_i in batch:
///         # Initialize fresh context for this task
///         psi_i = zeros(psi_dim)
///
///         # Inner loop: Adapt context only
///         for step in range(K):
///             x_augmented = inject_context(support_x, psi_i)
///             loss = model(x_augmented; phi) vs support_y
///             psi_i = psi_i - alpha * grad(loss, psi_i)  # Only update context!
///
///         # Evaluate on query set with adapted context
///         q_augmented = inject_context(query_x, psi_i)
///         meta_loss_i = model(q_augmented; phi) vs query_y
///
///     # Outer loop: Update body parameters
///     phi = phi - beta * mean(grad(meta_loss, phi))  # Update body only!
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Separation of Concerns**: Body parameters capture shared structure across tasks,
///    while context parameters capture task-specific information.
///
/// 2. **Efficient Adaptation**: Only adapting the small context vector means the inner
///    loop is O(context_dim) instead of O(total_params), typically 100x-1000x cheaper.
///
/// 3. **Reduced Meta-Overfitting**: Fewer adapted parameters means less risk of overfitting
///    the meta-learner to the training tasks.
///
/// 4. **Interpretable Context**: The context vector provides a low-dimensional representation
///    of what makes each task unique.
/// </para>
/// <para>
/// Reference: Zintgraf, L. M., Shiarli, K., Kurin, V., Hofmann, K., &amp; Whiteson, S. (2019).
/// Fast Context Adaptation via Meta-Learning. ICML 2019.
/// </para>
/// </remarks>
public class CAVIAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly CAVIAOptions<T, TInput, TOutput> _caviaOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _caviaOptions;

    /// <summary>
    /// Initializes a new instance of the CAVIAAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for CAVIA.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a CAVIA model ready for few-shot learning.
    ///
    /// <b>What CAVIA needs:</b>
    /// - <b>MetaModel:</b> Neural network whose input dimension accounts for the context
    ///   (input_dim + context_dim when using Concatenation mode)
    /// - <b>ContextDimension:</b> Size of the task-specific context vector (default 100)
    /// - <b>AdaptationSteps:</b> Number of context updates per task (default 5)
    ///
    /// <b>Important:</b> When using Concatenation mode, your model's input layer
    /// must accept (original_input_dim + context_dim) features.
    /// </para>
    /// </remarks>
    public CAVIAAlgorithm(CAVIAOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _caviaOptions = options;

        // Validate configuration
        if (!_caviaOptions.IsValid())
        {
            throw new ArgumentException("CAVIA configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.CAVIA"/>.</value>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.CAVIA;

    /// <summary>
    /// Performs one meta-training step using CAVIA's context adaptation strategy.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// CAVIA meta-training differs from MAML in that the inner loop only adapts context
    /// parameters while the outer loop updates the model's body parameters:
    /// </para>
    /// <para>
    /// <b>For each task in the batch:</b>
    /// 1. Initialize context parameters to the configured initial value (typically zeros)
    /// 2. Inner loop: Adapt context by gradient descent on support set (body frozen)
    /// 3. Augment query inputs with the adapted context
    /// 4. Compute meta-loss on the augmented query set
    /// 5. Compute gradients w.r.t. body parameters
    /// </para>
    /// <para>
    /// <b>Meta-update:</b>
    /// Average body gradients across all tasks and update shared model parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each call improves both the shared model (body) and
    /// the model's ability to quickly adapt its context for new tasks.
    /// The returned loss should decrease over training iterations.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;
        Vector<T>? accumulatedBodyGradients = null;

        foreach (var task in taskBatch.Tasks)
        {
            // Step 1: Initialize fresh context for this task
            var context = CreateInitialContext();

            // Step 2: Inner loop - adapt context parameters on support set
            context = AdaptContext(context, task.SupportInput, task.SupportOutput);

            // Step 3: Compute meta-loss on query set with adapted context
            var augmentedQueryInput = AugmentInput(task.QueryInput, context);
            var queryPredictions = MetaModel.Predict(augmentedQueryInput);
            T metaLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, metaLoss);

            // Step 4: Compute gradients w.r.t. body parameters (model parameters)
            var bodyGradients = ComputeGradients(MetaModel, augmentedQueryInput, task.QueryOutput);

            // Accumulate body gradients
            if (accumulatedBodyGradients == null)
            {
                accumulatedBodyGradients = bodyGradients;
            }
            else
            {
                for (int i = 0; i < accumulatedBodyGradients.Length; i++)
                {
                    accumulatedBodyGradients[i] = NumOps.Add(accumulatedBodyGradients[i], bodyGradients[i]);
                }
            }
        }

        if (accumulatedBodyGradients != null)
        {
            // Average body gradients
            T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
            for (int i = 0; i < accumulatedBodyGradients.Length; i++)
            {
                accumulatedBodyGradients[i] = NumOps.Divide(accumulatedBodyGradients[i], batchSizeT);
            }

            // Apply gradient clipping if configured
            if (_caviaOptions.GradientClipThreshold.HasValue && _caviaOptions.GradientClipThreshold.Value > 0)
            {
                accumulatedBodyGradients = ClipGradients(accumulatedBodyGradients, _caviaOptions.GradientClipThreshold.Value);
            }

            // Update body (model) parameters
            var currentParams = MetaModel.GetParameters();
            var updatedParams = ApplyGradients(currentParams, accumulatedBodyGradients, _caviaOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        // Return average meta-loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Adapts to a new task by learning task-specific context parameters.
    /// </summary>
    /// <param name="task">The new task containing support set examples.</param>
    /// <returns>A <see cref="CAVIAModel{T, TInput, TOutput}"/> that uses the adapted context for predictions.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// This is where CAVIA's efficiency shines - adaptation involves only updating the small
    /// context vector, not the entire model. The process is:
    /// </para>
    /// <para>
    /// 1. Initialize context to zeros (or configured initial value)
    /// 2. Perform K gradient steps on the support set, updating only context
    /// 3. Return a model that uses the adapted context + frozen body for inference
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When you have a new task with a few labeled examples:
    /// 1. Call this method with the support set
    /// 2. Get back a model customized for this specific task
    /// 3. Use the returned model to classify new examples
    ///
    /// The adaptation is very fast because only the context vector changes!
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Initialize fresh context
        var context = CreateInitialContext();

        // Adapt context on support set
        context = AdaptContext(context, task.SupportInput, task.SupportOutput);

        // Return a model that uses adapted context for inference
        return new CAVIAModel<T, TInput, TOutput>(
            MetaModel,
            context,
            _caviaOptions,
            NumOps);
    }

    /// <summary>
    /// Creates the initial context vector with the configured initial value.
    /// </summary>
    /// <returns>A context vector of dimension ContextDimension * NumContextVectors,
    /// initialized to the configured initial value (default: zeros).</returns>
    /// <remarks>
    /// <para>
    /// The initial context is the starting point for task-specific adaptation. In the original
    /// CAVIA paper, context is initialized to zeros so the model starts from a neutral state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of the context as a blank notepad before starting a new task.
    /// Each task fills in the notepad differently during adaptation. Starting from zeros means
    /// no assumptions about the task. If you set a non-zero initial value, the model starts
    /// with some prior belief about the task (useful when tasks share a common baseline).
    /// </para>
    /// </remarks>
    private Vector<T> CreateInitialContext()
    {
        int totalContextSize = _caviaOptions.ContextDimension * _caviaOptions.NumContextVectors;
        var context = new Vector<T>(totalContextSize);

        if (Math.Abs(_caviaOptions.ContextInitValue) > 1e-15)
        {
            T initValue = NumOps.FromDouble(_caviaOptions.ContextInitValue);
            for (int i = 0; i < totalContextSize; i++)
            {
                context[i] = initValue;
            }
        }

        return context;
    }

    /// <summary>
    /// Adapts the context vector using gradient descent on the support set.
    /// </summary>
    /// <param name="context">The current context vector.</param>
    /// <param name="supportInput">Support set inputs.</param>
    /// <param name="supportOutput">Support set outputs (labels).</param>
    /// <returns>The adapted context vector after K gradient steps.</returns>
    /// <remarks>
    /// <para>
    /// This is CAVIA's inner loop - the core of task-specific adaptation. Unlike MAML which
    /// updates all model parameters, CAVIA only updates the small context vector:
    /// </para>
    /// <para>
    /// <b>For each adaptation step:</b>
    /// 1. Augment support inputs with current context (inject task information)
    /// 2. Forward pass through the frozen body model (body params don't change)
    /// 3. Compute loss between predictions and support labels
    /// 4. Compute gradient of loss w.r.t. context parameters only (not body parameters)
    /// 5. Update context using gradient descent: psi = psi - alpha * grad
    /// 6. Optionally apply L2 regularization to prevent context from growing too large
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like fine-tuning a dial instead of rebuilding the whole machine.
    /// The model's core abilities (body) stay frozen - only the task-specific "dial" (context) turns.
    ///
    /// **Example with 5 adaptation steps:**
    /// - Step 1: Context = [0, 0, 0] -> Loss = 2.3 (random guessing)
    /// - Step 2: Context = [0.1, -0.2, 0.05] -> Loss = 1.8 (starting to learn task)
    /// - Step 3: Context = [0.15, -0.3, 0.1] -> Loss = 1.2 (getting better)
    /// - Step 4: Context = [0.2, -0.35, 0.12] -> Loss = 0.8 (good adaptation)
    /// - Step 5: Context = [0.22, -0.38, 0.13] -> Loss = 0.6 (well adapted!)
    ///
    /// Because the context vector is small (e.g., 100 dimensions vs. millions of body params),
    /// this adaptation is extremely fast.
    /// </para>
    /// </remarks>
    private Vector<T> AdaptContext(Vector<T> context, TInput supportInput, TOutput supportOutput)
    {
        for (int step = 0; step < _caviaOptions.AdaptationSteps; step++)
        {
            // Compute context gradients using finite differences on the context vector
            var contextGradients = ComputeContextGradients(context, supportInput, supportOutput);

            // Apply L2 regularization on context if configured
            if (_caviaOptions.UseContextRegularization)
            {
                T regStrength = NumOps.FromDouble(_caviaOptions.ContextRegularizationStrength);
                for (int i = 0; i < context.Length; i++)
                {
                    // L2 regularization gradient: reg_strength * context
                    T regGrad = NumOps.Multiply(regStrength, context[i]);
                    contextGradients[i] = NumOps.Add(contextGradients[i], regGrad);
                }
            }

            // Update context: psi = psi - alpha * grad
            T lr = NumOps.FromDouble(_caviaOptions.InnerLearningRate);
            for (int i = 0; i < context.Length; i++)
            {
                context[i] = NumOps.Subtract(context[i], NumOps.Multiply(lr, contextGradients[i]));
            }
        }

        return context;
    }

    /// <summary>
    /// Computes gradients of the loss with respect to context parameters using finite differences.
    /// </summary>
    /// <param name="context">The current context vector.</param>
    /// <param name="input">Input data to evaluate on.</param>
    /// <param name="expectedOutput">Expected output (labels).</param>
    /// <returns>Gradient vector with respect to each context parameter.</returns>
    /// <remarks>
    /// <para>
    /// Since context parameters are separate from the model's internal parameters,
    /// we compute their gradients using finite differences:
    /// <c>d(loss)/d(psi_i) = (L(psi + eps_i) - L(psi)) / eps</c>
    /// where eps_i is a small perturbation in the i-th context dimension.
    /// </para>
    /// <para>
    /// This is efficient because the context vector is small (typically 32-256 dimensions),
    /// so we only need (context_dim + 1) forward passes per gradient computation.
    /// For a context of size 100, that's 101 forward passes - much cheaper than backpropagating
    /// through the entire model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> To figure out how to adjust each context dimension, we use a simple trick:
    ///
    /// 1. Measure the loss with the current context (baseline)
    /// 2. For each context dimension:
    ///    a. Slightly increase that dimension (by a tiny amount epsilon)
    ///    b. Measure the loss again
    ///    c. If loss went up, this dimension should decrease (positive gradient)
    ///    d. If loss went down, this dimension should increase (negative gradient)
    /// 3. The gradient tells us which direction to move each dimension to reduce loss
    ///
    /// This is called "finite differences" and works like experimentally nudging each dial
    /// to see which way makes things better.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeContextGradients(Vector<T> context, TInput input, TOutput expectedOutput)
    {
        var gradients = new Vector<T>(context.Length);
        T epsilon = NumOps.FromDouble(1e-5);

        // Compute baseline loss with current context
        var baseAugmented = AugmentInput(input, context);
        var basePredictions = MetaModel.Predict(baseAugmented);
        T baseLoss = ComputeLossFromOutput(basePredictions, expectedOutput);

        // Compute gradient for each context dimension using forward finite differences
        for (int i = 0; i < context.Length; i++)
        {
            // Perturb context dimension i
            T originalValue = context[i];
            context[i] = NumOps.Add(originalValue, epsilon);

            // Compute perturbed loss
            var perturbedAugmented = AugmentInput(input, context);
            var perturbedPredictions = MetaModel.Predict(perturbedAugmented);
            T perturbedLoss = ComputeLossFromOutput(perturbedPredictions, expectedOutput);

            // Gradient = (perturbed_loss - base_loss) / epsilon
            gradients[i] = NumOps.Divide(
                NumOps.Subtract(perturbedLoss, baseLoss),
                epsilon);

            // Restore original value
            context[i] = originalValue;
        }

        return gradients;
    }

    /// <summary>
    /// Augments the input by injecting the context vector according to the configured injection mode.
    /// </summary>
    /// <param name="input">The original input.</param>
    /// <param name="context">The context vector to inject.</param>
    /// <returns>The augmented input with context information.</returns>
    /// <remarks>
    /// <para>
    /// This is the mechanism that makes CAVIA work: the context vector is injected into the input
    /// so the body model receives both the original features and the task-specific context.
    /// The injection mode determines how context is combined with input:
    /// </para>
    /// <para>
    /// <b>Injection Modes:</b>
    /// <list type="bullet">
    /// <item><b>Concatenation (default):</b> Appends context to each input sample's features.
    /// Input [batch, D] becomes [batch, D + C]. Model's first layer must accept D + C inputs.</item>
    /// <item><b>Addition:</b> Adds context element-wise to input features.
    /// Requires context_dim = input_dim. Acts as a learned bias per task.</item>
    /// <item><b>Multiplication:</b> Multiplies context element-wise with input features (FiLM-style).
    /// Requires context_dim = input_dim. Acts as a learned scaling per task.</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The context vector needs to reach the model somehow. There are three ways:
    ///
    /// 1. **Concatenation** (recommended): Glue the context to the end of each input.
    ///    Like adding extra columns to a spreadsheet. Most flexible and commonly used.
    ///    Example: input=[1.0, 2.0, 3.0] + context=[0.5, -0.3] = [1.0, 2.0, 3.0, 0.5, -0.3]
    ///
    /// 2. **Addition**: Add context values to matching input values.
    ///    Like adjusting each feature by a task-specific offset.
    ///    Example: input=[1.0, 2.0, 3.0] + context=[0.5, -0.3, 0.1] = [1.5, 1.7, 3.1]
    ///
    /// 3. **Multiplication**: Scale each feature by a task-specific factor (FiLM-style gating).
    ///    Like adjusting the importance of each feature per task.
    ///    Example: input=[1.0, 2.0, 3.0] * context=[1.5, 0.5, 1.0] = [1.5, 1.0, 3.0]
    /// </para>
    /// </remarks>
    private TInput AugmentInput(TInput input, Vector<T> context) =>
        CAVIAContextHelper<T>.AugmentInput<TInput>(input, context, _caviaOptions.ContextInjectionMode, NumOps);
}

/// <summary>
/// Shared static helper for CAVIA context injection operations.
/// Eliminates duplication between CAVIAAlgorithm and CAVIAModel.
/// </summary>
internal static class CAVIAContextHelper<T>
{
    /// <summary>Augments input by injecting context according to the specified injection mode.</summary>
    public static TInput AugmentInput<TInput>(TInput input, Vector<T> context,
        CAVIAContextInjectionMode mode, INumericOperations<T> numOps)
    {
        return mode switch
        {
            CAVIAContextInjectionMode.Concatenation => ConcatenateContext<TInput>(input, context),
            CAVIAContextInjectionMode.Addition => AddContext<TInput>(input, context, numOps),
            CAVIAContextInjectionMode.Multiplication => MultiplyContext<TInput>(input, context, numOps),
            _ => ConcatenateContext<TInput>(input, context)
        };
    }

    public static TInput ConcatenateContext<TInput>(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
            return (TInput)(object)ConcatenateTensorWithContext(tensor, context);
        if (input is Matrix<T> matrix)
            return (TInput)(object)ConcatenateMatrixWithContext(matrix, context);
        if (input is Vector<T> vector)
            return (TInput)(object)ConcatenateVectorWithContext(vector, context);
        throw new NotSupportedException(
            $"Input type {typeof(TInput).Name} is not supported for context concatenation. " +
            $"Supported types: Tensor<T>, Matrix<T>, Vector<T>.");
    }

    public static Tensor<T> ConcatenateTensorWithContext(Tensor<T> tensor, Vector<T> context)
    {
        if (tensor.Shape.Length == 1)
        {
            int newDim = tensor.Shape[0] + context.Length;
            var result = new Tensor<T>(new int[] { newDim });
            for (int i = 0; i < tensor.Shape[0]; i++)
                result[new int[] { i }] = tensor[new int[] { i }];
            for (int i = 0; i < context.Length; i++)
                result[new int[] { tensor.Shape[0] + i }] = context[i];
            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            int batchSize = tensor.Shape[0];
            int originalFeatures = tensor.Shape[1];
            int newFeatures = originalFeatures + context.Length;
            var result = new Tensor<T>(new int[] { batchSize, newFeatures });
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < originalFeatures; f++)
                    result[new int[] { b, f }] = tensor[new int[] { b, f }];
                for (int c = 0; c < context.Length; c++)
                    result[new int[] { b, originalFeatures + c }] = context[c];
            }
            return result;
        }
        throw new NotSupportedException(
            $"Tensor with {tensor.Shape.Length} dimensions is not supported for context concatenation.");
    }

    public static Matrix<T> ConcatenateMatrixWithContext(Matrix<T> matrix, Vector<T> context)
    {
        int newCols = matrix.Columns + context.Length;
        var result = new Matrix<T>(matrix.Rows, newCols);
        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int col = 0; col < matrix.Columns; col++)
                result[row, col] = matrix[row, col];
            for (int c = 0; c < context.Length; c++)
                result[row, matrix.Columns + c] = context[c];
        }
        return result;
    }

    public static Vector<T> ConcatenateVectorWithContext(Vector<T> vector, Vector<T> context)
    {
        var result = new Vector<T>(vector.Length + context.Length);
        for (int i = 0; i < vector.Length; i++)
            result[i] = vector[i];
        for (int i = 0; i < context.Length; i++)
            result[vector.Length + i] = context[i];
        return result;
    }

    public static TInput AddContext<TInput>(TInput input, Vector<T> context, INumericOperations<T> numOps)
    {
        if (input is Tensor<T> tensor)
            return (TInput)(object)AddContextToTensor(tensor, context, numOps);
        if (input is Matrix<T> matrix)
            return (TInput)(object)AddContextToMatrix(matrix, context, numOps);
        if (input is Vector<T> vector)
        {
            if (vector.Length != context.Length)
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match input dimension ({vector.Length}) for Addition injection mode.");
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
                result[i] = numOps.Add(vector[i], context[i]);
            return (TInput)(object)result;
        }
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context addition.");
    }

    public static Tensor<T> AddContextToTensor(Tensor<T> tensor, Vector<T> context, INumericOperations<T> numOps)
    {
        if (tensor.Shape.Length == 1)
        {
            if (tensor.Shape[0] != context.Length)
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match tensor dimension ({tensor.Shape[0]}) for Addition injection mode.");
            var result = new Tensor<T>(tensor.Shape);
            for (int i = 0; i < tensor.Shape[0]; i++)
                result[new int[] { i }] = numOps.Add(tensor[new int[] { i }], context[i]);
            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            if (tensor.Shape[1] != context.Length)
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match feature dimension ({tensor.Shape[1]}) for Addition injection mode.");
            var result = new Tensor<T>(tensor.Shape);
            for (int b = 0; b < tensor.Shape[0]; b++)
                for (int f = 0; f < tensor.Shape[1]; f++)
                    result[new int[] { b, f }] = numOps.Add(tensor[new int[] { b, f }], context[f]);
            return result;
        }
        throw new NotSupportedException($"Tensor with {tensor.Shape.Length} dimensions is not supported for context addition.");
    }

    public static Matrix<T> AddContextToMatrix(Matrix<T> matrix, Vector<T> context, INumericOperations<T> numOps)
    {
        if (matrix.Columns != context.Length)
            throw new ArgumentException(
                $"Context dimension ({context.Length}) must match feature dimension ({matrix.Columns}) for Addition injection mode.");
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int row = 0; row < matrix.Rows; row++)
            for (int col = 0; col < matrix.Columns; col++)
                result[row, col] = numOps.Add(matrix[row, col], context[col]);
        return result;
    }

    public static TInput MultiplyContext<TInput>(TInput input, Vector<T> context, INumericOperations<T> numOps)
    {
        if (input is Tensor<T> tensor)
            return (TInput)(object)MultiplyContextWithTensor(tensor, context, numOps);
        if (input is Matrix<T> matrix)
            return (TInput)(object)MultiplyContextWithMatrix(matrix, context, numOps);
        if (input is Vector<T> vector)
        {
            if (vector.Length != context.Length)
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match input dimension ({vector.Length}) for Multiplication injection mode.");
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
                result[i] = numOps.Multiply(vector[i], context[i]);
            return (TInput)(object)result;
        }
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context multiplication.");
    }

    public static Tensor<T> MultiplyContextWithTensor(Tensor<T> tensor, Vector<T> context, INumericOperations<T> numOps)
    {
        if (tensor.Shape.Length == 1)
        {
            if (tensor.Shape[0] != context.Length)
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match tensor dimension ({tensor.Shape[0]}) for Multiplication injection mode.");
            var result = new Tensor<T>(tensor.Shape);
            for (int i = 0; i < tensor.Shape[0]; i++)
                result[new int[] { i }] = numOps.Multiply(tensor[new int[] { i }], context[i]);
            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            if (tensor.Shape[1] != context.Length)
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match feature dimension ({tensor.Shape[1]}) for Multiplication injection mode.");
            var result = new Tensor<T>(tensor.Shape);
            for (int b = 0; b < tensor.Shape[0]; b++)
                for (int f = 0; f < tensor.Shape[1]; f++)
                    result[new int[] { b, f }] = numOps.Multiply(tensor[new int[] { b, f }], context[f]);
            return result;
        }
        throw new NotSupportedException($"Tensor with {tensor.Shape.Length} dimensions is not supported for context multiplication.");
    }

    public static Matrix<T> MultiplyContextWithMatrix(Matrix<T> matrix, Vector<T> context, INumericOperations<T> numOps)
    {
        if (matrix.Columns != context.Length)
            throw new ArgumentException(
                $"Context dimension ({context.Length}) must match feature dimension ({matrix.Columns}) for Multiplication injection mode.");
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int row = 0; row < matrix.Rows; row++)
            for (int col = 0; col < matrix.Columns; col++)
                result[row, col] = numOps.Multiply(matrix[row, col], context[col]);
        return result;
    }
}


/// <summary>
/// CAVIA inference model that uses adapted context parameters for predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model wraps the meta-learned body parameters with task-specific context parameters.
/// It is returned by <see cref="CAVIAAlgorithm{T, TInput, TOutput}.Adapt"/> and provides
/// fast inference by augmenting inputs with the adapted context before passing to the model.
/// </para>
/// <para><b>For Beginners:</b> After adapting CAVIA to a new task, you get this model.
/// It automatically adds the learned task context to your inputs, so you can use it
/// just like any other model: call Predict() with new examples.
/// </para>
/// </remarks>
public class CAVIAModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _bodyModel;
    private readonly Vector<T> _adaptedContext;
    private readonly CAVIAOptions<T, TInput, TOutput> _options;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the CAVIAModel with adapted context.
    /// </summary>
    /// <param name="bodyModel">The meta-learned body model (frozen during inference).</param>
    /// <param name="adaptedContext">The task-adapted context vector.</param>
    /// <param name="options">CAVIA configuration options.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    /// <exception cref="ArgumentNullException">Thrown when any required parameter is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This model is created by <see cref="CAVIAAlgorithm{T, TInput, TOutput}.Adapt"/>
    /// after learning a task-specific context from the support set. You don't create this directly -
    /// instead, call Adapt() with your few-shot examples and use the returned CAVIAModel for predictions.
    ///
    /// The model combines:
    /// - The shared body model (learned across all tasks during meta-training)
    /// - The adapted context (learned specifically for this task during adaptation)
    ///
    /// When you call Predict(), the context is automatically injected into the input before
    /// passing it through the body model, so you use it like any normal model.
    /// </para>
    /// </remarks>
    public CAVIAModel(
        IFullModel<T, TInput, TOutput> bodyModel,
        Vector<T> adaptedContext,
        CAVIAOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
    {
        Guard.NotNull(bodyModel);
        _bodyModel = bodyModel;
        Guard.NotNull(adaptedContext);
        _adaptedContext = adaptedContext;
        Guard.NotNull(options);
        _options = options;
        Guard.NotNull(numOps);
        _numOps = numOps;
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted context vector for inspection or further processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The context vector provides a low-dimensional summary of the task.
    /// It can be used for:
    /// - Task clustering (similar tasks have similar contexts)
    /// - Visualization (project context to 2D for analysis)
    /// - Transfer (reuse context from a similar seen task)
    /// </para>
    /// </remarks>
    public Vector<T> AdaptedContext => _adaptedContext;

    /// <summary>
    /// Makes predictions by augmenting input with the adapted context and running through the body model.
    /// </summary>
    /// <param name="input">The input to classify or regress.</param>
    /// <returns>Model predictions (class probabilities, regression values, etc.).</returns>
    /// <remarks>
    /// <para>
    /// The prediction process is:
    /// 1. Inject the adapted context into the input (using the configured injection mode)
    /// 2. Pass the augmented input through the frozen body model
    /// 3. Return the model's output
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This works just like any other model's Predict() method. The only
    /// difference is that behind the scenes, your input is augmented with the task-specific context
    /// before being processed. You don't need to worry about the context injection - it happens
    /// automatically based on the configuration.
    ///
    /// Example usage:
    /// <code>
    /// // Adapt to a new task
    /// var adaptedModel = caviaAlgorithm.Adapt(fewShotTask);
    ///
    /// // Use like any model - context injection is automatic
    /// var predictions = adaptedModel.Predict(newInput);
    /// </code>
    /// </para>
    /// </remarks>
    public TOutput Predict(TInput input)
    {
        // Augment input with adapted context
        var augmentedInput = AugmentInput(input, _adaptedContext);

        // Forward through the body model
        return _bodyModel.Predict(augmentedInput);
    }

    /// <summary>
    /// Trains the model (not applicable for CAVIA inference models).
    /// </summary>
    /// <param name="inputs">Training inputs (unused).</param>
    /// <param name="targets">Training targets (unused).</param>
    /// <exception cref="NotSupportedException">Always thrown - CAVIA inference models are frozen.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CAVIA inference models are "frozen" after adaptation - they can't be
    /// trained further. If you need to adapt to a different task, call
    /// <see cref="CAVIAAlgorithm{T, TInput, TOutput}.Adapt"/> again with the new task's support set.
    /// </para>
    /// </remarks>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException(
            "CAVIA inference models don't support training. " +
            "Use CAVIAAlgorithm.Adapt() to create a new adapted model.");
    }

    /// <summary>
    /// Updates model parameters (not applicable for CAVIA inference models).
    /// </summary>
    /// <param name="parameters">Parameters to set (unused).</param>
    /// <exception cref="NotSupportedException">Always thrown - CAVIA inference models are frozen.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The adapted model's parameters (body + context) are fixed after
    /// adaptation. To get a model with different parameters, create a new one via Adapt().
    /// </para>
    /// </remarks>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("CAVIA inference models don't have directly trainable parameters.");
    }

    /// <summary>
    /// Gets model parameters (not applicable for CAVIA inference models).
    /// </summary>
    /// <returns>This method always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown - use <see cref="AdaptedContext"/> to inspect the context.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The adapted context can be inspected via the <see cref="AdaptedContext"/>
    /// property. The body model's parameters are managed by the meta-learning algorithm.
    /// </para>
    /// </remarks>
    public Vector<T> GetParameters()
    {
        throw new NotSupportedException("CAVIA inference models don't expose parameters directly.");
    }

    /// <summary>
    /// Gets metadata about this CAVIA inference model.
    /// </summary>
    /// <returns>Model metadata describing this adapted CAVIA model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns descriptive information about the model, including
    /// the fact that it's a CAVIA-adapted model with a specific context vector.
    /// </para>
    /// </remarks>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>
    /// Augments input by injecting context according to the configured injection mode.
    /// </summary>
    /// <param name="input">The original input to augment.</param>
    /// <param name="context">The adapted context vector to inject.</param>
    /// <returns>The augmented input with context information included.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the same context injection used during adaptation, now applied
    /// at inference time. The context is injected into every input before the model sees it,
    /// so the model always processes inputs with the task-specific context information.
    /// </para>
    /// </remarks>
    private TInput AugmentInput(TInput input, Vector<T> context) =>
        CAVIAContextHelper<T>.AugmentInput<TInput>(input, context, _options.ContextInjectionMode, _numOps);
}
