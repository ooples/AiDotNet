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
    /// For each adaptation step:
    /// 1. Augment support inputs with current context
    /// 2. Forward pass through the frozen model
    /// 3. Compute loss
    /// 4. Compute gradient of loss w.r.t. context parameters (not body parameters)
    /// 5. Update context using gradient descent
    /// 6. Optionally apply L2 regularization to prevent context from growing too large
    /// </para>
    /// </remarks>
    private Vector<T> AdaptContext(Vector<T> context, TInput supportInput, TOutput supportOutput)
    {
        for (int step = 0; step < _caviaOptions.AdaptationSteps; step++)
        {
            // Augment input with current context
            var augmentedInput = AugmentInput(supportInput, context);

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
    /// we compute their gradients using finite differences: d(loss)/d(psi_i) = (L(psi+eps_i) - L(psi)) / eps.
    /// This is efficient because the context vector is small (typically 32-256 dimensions).
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
    /// The injection mode determines how context is combined with input:
    /// - <b>Concatenation:</b> Appends context to each input sample's features
    /// - <b>Addition:</b> Adds context element-wise to input features
    /// - <b>Multiplication:</b> Multiplies context element-wise with input features
    /// </para>
    /// <para>
    /// For Concatenation mode, the model's input layer must accept (input_dim + context_dim) features.
    /// For Addition/Multiplication, context_dim must equal input_dim.
    /// </para>
    /// </remarks>
    private TInput AugmentInput(TInput input, Vector<T> context)
    {
        return _caviaOptions.ContextInjectionMode switch
        {
            CAVIAContextInjectionMode.Concatenation => ConcatenateContext(input, context),
            CAVIAContextInjectionMode.Addition => AddContext(input, context),
            CAVIAContextInjectionMode.Multiplication => MultiplyContext(input, context),
            _ => ConcatenateContext(input, context)
        };
    }

    /// <summary>
    /// Concatenates the context vector with each sample in the input.
    /// </summary>
    private TInput ConcatenateContext(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
        {
            return (TInput)(object)ConcatenateTensorWithContext(tensor, context);
        }

        if (input is Matrix<T> matrix)
        {
            return (TInput)(object)ConcatenateMatrixWithContext(matrix, context);
        }

        if (input is Vector<T> vector)
        {
            return (TInput)(object)ConcatenateVectorWithContext(vector, context);
        }

        throw new NotSupportedException(
            $"Input type {typeof(TInput).Name} is not supported for context concatenation. " +
            $"Supported types: Tensor<T>, Matrix<T>, Vector<T>.");
    }

    /// <summary>
    /// Concatenates context with a tensor input.
    /// For 1D tensor [features]: creates [features + context_dim].
    /// For 2D tensor [batch, features]: creates [batch, features + context_dim].
    /// </summary>
    private Tensor<T> ConcatenateTensorWithContext(Tensor<T> tensor, Vector<T> context)
    {
        if (tensor.Shape.Length == 1)
        {
            // 1D: [features] -> [features + context_dim]
            int newDim = tensor.Shape[0] + context.Length;
            var result = new Tensor<T>(new int[] { newDim });

            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                result[new int[] { i }] = tensor[new int[] { i }];
            }
            for (int i = 0; i < context.Length; i++)
            {
                result[new int[] { tensor.Shape[0] + i }] = context[i];
            }

            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            // 2D: [batch, features] -> [batch, features + context_dim]
            int batchSize = tensor.Shape[0];
            int originalFeatures = tensor.Shape[1];
            int newFeatures = originalFeatures + context.Length;
            var result = new Tensor<T>(new int[] { batchSize, newFeatures });

            for (int b = 0; b < batchSize; b++)
            {
                // Copy original features
                for (int f = 0; f < originalFeatures; f++)
                {
                    result[new int[] { b, f }] = tensor[new int[] { b, f }];
                }
                // Append context (same context for all samples in the batch)
                for (int c = 0; c < context.Length; c++)
                {
                    result[new int[] { b, originalFeatures + c }] = context[c];
                }
            }

            return result;
        }

        throw new NotSupportedException(
            $"Tensor with {tensor.Shape.Length} dimensions is not supported for context concatenation. " +
            $"Supported: 1D [features] or 2D [batch, features].");
    }

    /// <summary>
    /// Concatenates context with a matrix input. Each row gets the context appended.
    /// [batch, features] -> [batch, features + context_dim].
    /// </summary>
    private Matrix<T> ConcatenateMatrixWithContext(Matrix<T> matrix, Vector<T> context)
    {
        int newCols = matrix.Columns + context.Length;
        var result = new Matrix<T>(matrix.Rows, newCols);

        for (int row = 0; row < matrix.Rows; row++)
        {
            // Copy original features
            for (int col = 0; col < matrix.Columns; col++)
            {
                result[row, col] = matrix[row, col];
            }
            // Append context
            for (int c = 0; c < context.Length; c++)
            {
                result[row, matrix.Columns + c] = context[c];
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates context with a vector input.
    /// [features] -> [features + context_dim].
    /// </summary>
    private Vector<T> ConcatenateVectorWithContext(Vector<T> vector, Vector<T> context)
    {
        var result = new Vector<T>(vector.Length + context.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = vector[i];
        }
        for (int i = 0; i < context.Length; i++)
        {
            result[vector.Length + i] = context[i];
        }

        return result;
    }

    /// <summary>
    /// Adds the context vector element-wise to each sample in the input.
    /// Requires context dimension to match input feature dimension.
    /// </summary>
    private TInput AddContext(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
        {
            return (TInput)(object)AddContextToTensor(tensor, context);
        }

        if (input is Matrix<T> matrix)
        {
            return (TInput)(object)AddContextToMatrix(matrix, context);
        }

        if (input is Vector<T> vector)
        {
            if (vector.Length != context.Length)
            {
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match input dimension ({vector.Length}) " +
                    $"for Addition injection mode.");
            }
            return (TInput)(object)Engine.Add(vector, context);
        }

        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context addition.");
    }

    /// <summary>
    /// Adds context element-wise to a tensor.
    /// </summary>
    private Tensor<T> AddContextToTensor(Tensor<T> tensor, Vector<T> context)
    {
        if (tensor.Shape.Length == 1)
        {
            if (tensor.Shape[0] != context.Length)
            {
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match tensor dimension ({tensor.Shape[0]}) " +
                    $"for Addition injection mode.");
            }

            var result = new Tensor<T>(tensor.Shape);
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                result[new int[] { i }] = NumOps.Add(tensor[new int[] { i }], context[i]);
            }
            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            if (tensor.Shape[1] != context.Length)
            {
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match feature dimension ({tensor.Shape[1]}) " +
                    $"for Addition injection mode.");
            }

            var result = new Tensor<T>(tensor.Shape);
            for (int b = 0; b < tensor.Shape[0]; b++)
            {
                for (int f = 0; f < tensor.Shape[1]; f++)
                {
                    result[new int[] { b, f }] = NumOps.Add(tensor[new int[] { b, f }], context[f]);
                }
            }
            return result;
        }

        throw new NotSupportedException(
            $"Tensor with {tensor.Shape.Length} dimensions is not supported for context addition.");
    }

    /// <summary>
    /// Adds context element-wise to each row of a matrix.
    /// </summary>
    private Matrix<T> AddContextToMatrix(Matrix<T> matrix, Vector<T> context)
    {
        if (matrix.Columns != context.Length)
        {
            throw new ArgumentException(
                $"Context dimension ({context.Length}) must match feature dimension ({matrix.Columns}) " +
                $"for Addition injection mode.");
        }

        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int col = 0; col < matrix.Columns; col++)
            {
                result[row, col] = NumOps.Add(matrix[row, col], context[col]);
            }
        }
        return result;
    }

    /// <summary>
    /// Multiplies the context vector element-wise with each sample in the input (FiLM-style gating).
    /// Requires context dimension to match input feature dimension.
    /// </summary>
    private TInput MultiplyContext(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
        {
            return (TInput)(object)MultiplyContextWithTensor(tensor, context);
        }

        if (input is Matrix<T> matrix)
        {
            return (TInput)(object)MultiplyContextWithMatrix(matrix, context);
        }

        if (input is Vector<T> vector)
        {
            if (vector.Length != context.Length)
            {
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match input dimension ({vector.Length}) " +
                    $"for Multiplication injection mode.");
            }
            return (TInput)(object)Engine.Multiply(vector, context);
        }

        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context multiplication.");
    }

    /// <summary>
    /// Multiplies context element-wise with a tensor.
    /// </summary>
    private Tensor<T> MultiplyContextWithTensor(Tensor<T> tensor, Vector<T> context)
    {
        if (tensor.Shape.Length == 1)
        {
            if (tensor.Shape[0] != context.Length)
            {
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match tensor dimension ({tensor.Shape[0]}) " +
                    $"for Multiplication injection mode.");
            }

            var result = new Tensor<T>(tensor.Shape);
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                result[new int[] { i }] = NumOps.Multiply(tensor[new int[] { i }], context[i]);
            }
            return result;
        }
        else if (tensor.Shape.Length == 2)
        {
            if (tensor.Shape[1] != context.Length)
            {
                throw new ArgumentException(
                    $"Context dimension ({context.Length}) must match feature dimension ({tensor.Shape[1]}) " +
                    $"for Multiplication injection mode.");
            }

            var result = new Tensor<T>(tensor.Shape);
            for (int b = 0; b < tensor.Shape[0]; b++)
            {
                for (int f = 0; f < tensor.Shape[1]; f++)
                {
                    result[new int[] { b, f }] = NumOps.Multiply(tensor[new int[] { b, f }], context[f]);
                }
            }
            return result;
        }

        throw new NotSupportedException(
            $"Tensor with {tensor.Shape.Length} dimensions is not supported for context multiplication.");
    }

    /// <summary>
    /// Multiplies context element-wise with each row of a matrix.
    /// </summary>
    private Matrix<T> MultiplyContextWithMatrix(Matrix<T> matrix, Vector<T> context)
    {
        if (matrix.Columns != context.Length)
        {
            throw new ArgumentException(
                $"Context dimension ({context.Length}) must match feature dimension ({matrix.Columns}) " +
                $"for Multiplication injection mode.");
        }

        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int col = 0; col < matrix.Columns; col++)
            {
                result[row, col] = NumOps.Multiply(matrix[row, col], context[col]);
            }
        }
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
    /// Initializes a new instance of the CAVIAModel.
    /// </summary>
    /// <param name="bodyModel">The meta-learned body model (frozen during inference).</param>
    /// <param name="adaptedContext">The task-adapted context vector.</param>
    /// <param name="options">CAVIA configuration options.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    public CAVIAModel(
        IFullModel<T, TInput, TOutput> bodyModel,
        Vector<T> adaptedContext,
        CAVIAOptions<T, TInput, TOutput> options,
        INumericOperations<T> numOps)
    {
        _bodyModel = bodyModel ?? throw new ArgumentNullException(nameof(bodyModel));
        _adaptedContext = adaptedContext ?? throw new ArgumentNullException(nameof(adaptedContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _numOps = numOps ?? throw new ArgumentNullException(nameof(numOps));
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
    /// <param name="input">The input to classify.</param>
    /// <returns>Model predictions (class probabilities, regression values, etc.).</returns>
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
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException(
            "CAVIA inference models don't support training. " +
            "Use CAVIAAlgorithm.Adapt() to create a new adapted model.");
    }

    /// <summary>
    /// Updates model parameters (not applicable for CAVIA inference models).
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("CAVIA inference models don't have directly trainable parameters.");
    }

    /// <summary>
    /// Gets model parameters (not applicable for CAVIA inference models).
    /// </summary>
    public Vector<T> GetParameters()
    {
        throw new NotSupportedException("CAVIA inference models don't expose parameters directly.");
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>Model metadata.</returns>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>
    /// Augments input by injecting context according to the configured injection mode.
    /// </summary>
    private TInput AugmentInput(TInput input, Vector<T> context)
    {
        return _options.ContextInjectionMode switch
        {
            CAVIAContextInjectionMode.Concatenation => ConcatenateContext(input, context),
            CAVIAContextInjectionMode.Addition => AddContext(input, context),
            CAVIAContextInjectionMode.Multiplication => MultiplyContext(input, context),
            _ => ConcatenateContext(input, context)
        };
    }

    private TInput ConcatenateContext(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
        {
            return (TInput)(object)ConcatenateTensorWithContext(tensor, context);
        }
        if (input is Matrix<T> matrix)
        {
            return (TInput)(object)ConcatenateMatrixWithContext(matrix, context);
        }
        if (input is Vector<T> vector)
        {
            return (TInput)(object)ConcatenateVectorWithContext(vector, context);
        }
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context concatenation.");
    }

    private Tensor<T> ConcatenateTensorWithContext(Tensor<T> tensor, Vector<T> context)
    {
        if (tensor.Shape.Length == 1)
        {
            int newDim = tensor.Shape[0] + context.Length;
            var result = new Tensor<T>(new int[] { newDim });
            for (int i = 0; i < tensor.Shape[0]; i++)
            {
                result[new int[] { i }] = tensor[new int[] { i }];
            }
            for (int i = 0; i < context.Length; i++)
            {
                result[new int[] { tensor.Shape[0] + i }] = context[i];
            }
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
                {
                    result[new int[] { b, f }] = tensor[new int[] { b, f }];
                }
                for (int c = 0; c < context.Length; c++)
                {
                    result[new int[] { b, originalFeatures + c }] = context[c];
                }
            }
            return result;
        }
        throw new NotSupportedException($"Tensor with {tensor.Shape.Length} dimensions is not supported.");
    }

    private Matrix<T> ConcatenateMatrixWithContext(Matrix<T> matrix, Vector<T> context)
    {
        int newCols = matrix.Columns + context.Length;
        var result = new Matrix<T>(matrix.Rows, newCols);
        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int col = 0; col < matrix.Columns; col++)
            {
                result[row, col] = matrix[row, col];
            }
            for (int c = 0; c < context.Length; c++)
            {
                result[row, matrix.Columns + c] = context[c];
            }
        }
        return result;
    }

    private Vector<T> ConcatenateVectorWithContext(Vector<T> vector, Vector<T> context)
    {
        var result = new Vector<T>(vector.Length + context.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = vector[i];
        }
        for (int i = 0; i < context.Length; i++)
        {
            result[vector.Length + i] = context[i];
        }
        return result;
    }

    private TInput AddContext(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                var result = new Tensor<T>(tensor.Shape);
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    result[new int[] { i }] = _numOps.Add(tensor[new int[] { i }], context[i]);
                }
                return (TInput)(object)result;
            }
            else if (tensor.Shape.Length == 2)
            {
                var result = new Tensor<T>(tensor.Shape);
                for (int b = 0; b < tensor.Shape[0]; b++)
                {
                    for (int f = 0; f < tensor.Shape[1]; f++)
                    {
                        result[new int[] { b, f }] = _numOps.Add(tensor[new int[] { b, f }], context[f]);
                    }
                }
                return (TInput)(object)result;
            }
        }
        if (input is Matrix<T> matrix)
        {
            var result = new Matrix<T>(matrix.Rows, matrix.Columns);
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int col = 0; col < matrix.Columns; col++)
                {
                    result[row, col] = _numOps.Add(matrix[row, col], context[col]);
                }
            }
            return (TInput)(object)result;
        }
        if (input is Vector<T> vector)
        {
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = _numOps.Add(vector[i], context[i]);
            }
            return (TInput)(object)result;
        }
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context addition.");
    }

    private TInput MultiplyContext(TInput input, Vector<T> context)
    {
        if (input is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                var result = new Tensor<T>(tensor.Shape);
                for (int i = 0; i < tensor.Shape[0]; i++)
                {
                    result[new int[] { i }] = _numOps.Multiply(tensor[new int[] { i }], context[i]);
                }
                return (TInput)(object)result;
            }
            else if (tensor.Shape.Length == 2)
            {
                var result = new Tensor<T>(tensor.Shape);
                for (int b = 0; b < tensor.Shape[0]; b++)
                {
                    for (int f = 0; f < tensor.Shape[1]; f++)
                    {
                        result[new int[] { b, f }] = _numOps.Multiply(tensor[new int[] { b, f }], context[f]);
                    }
                }
                return (TInput)(object)result;
            }
        }
        if (input is Matrix<T> matrix)
        {
            var result = new Matrix<T>(matrix.Rows, matrix.Columns);
            for (int row = 0; row < matrix.Rows; row++)
            {
                for (int col = 0; col < matrix.Columns; col++)
                {
                    result[row, col] = _numOps.Multiply(matrix[row, col], context[col]);
                }
            }
            return (TInput)(object)result;
        }
        if (input is Vector<T> vector)
        {
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = _numOps.Multiply(vector[i], context[i]);
            }
            return (TInput)(object)result;
        }
        throw new NotSupportedException($"Input type {typeof(TInput).Name} is not supported for context multiplication.");
    }
}
