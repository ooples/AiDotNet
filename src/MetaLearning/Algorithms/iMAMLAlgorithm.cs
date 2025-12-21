using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the iMAML (Implicit Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// iMAML is a memory-efficient variant of MAML that uses implicit differentiation to
/// compute meta-gradients. Instead of backpropagating through all adaptation steps,
/// it uses the implicit function theorem to directly compute gradients at the adapted
/// parameters, significantly reducing memory requirements.
/// </para>
/// <para>
/// Key advantages over MAML:
/// - Constant memory cost regardless of number of adaptation steps
/// - Can use many more adaptation steps without memory issues
/// - Often achieves better performance than first-order MAML (FOMAML)
/// </para>
/// <para>
/// <b>For Beginners:</b> iMAML solves one of MAML's biggest problems - memory usage.
/// </para>
/// <para>
/// The problem with MAML:
/// - To learn from adaptation, MAML needs to remember every step
/// - More adaptation steps = much more memory needed
/// - This limits how much adaptation you can do
/// </para>
/// <para>
/// How iMAML solves it:
/// - Uses a mathematical shortcut (implicit differentiation)
/// - Only needs to remember the start and end points
/// - Can do many more adaptation steps with the same memory
/// </para>
/// <para>
/// The implicit function theorem allows computing gradients through the adaptation
/// process by solving: (I + lambda * H)^(-1) * g, where H is the Hessian of the
/// inner loss and g is the gradient of the query loss. This is solved efficiently
/// using Conjugate Gradient iteration.
/// </para>
/// <para>
/// Reference: Rajeswaran, A., Finn, C., Kakade, S. M., &amp; Levine, S. (2019).
/// Meta-learning with implicit gradients.
/// </para>
/// </remarks>
public class iMAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly iMAMLOptions<T, TInput, TOutput> _imamlOptions;

    /// <summary>
    /// Initializes a new instance of the iMAMLAlgorithm class.
    /// </summary>
    /// <param name="options">iMAML configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create iMAML with minimal configuration (uses all defaults)
    /// var options = new iMAMLOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var imaml = new iMAMLAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create iMAML with custom configuration for more adaptation steps
    /// var options = new iMAMLOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     AdaptationSteps = 20,  // iMAML can handle many steps!
    ///     LambdaRegularization = 2.0,
    ///     ConjugateGradientIterations = 15
    /// };
    /// var imaml = new iMAMLAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public iMAMLAlgorithm(iMAMLOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _imamlOptions = options;
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.iMAML"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as iMAML (Implicit MAML),
    /// distinguishing it from standard MAML and other meta-learning algorithms.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.iMAML;

    /// <summary>
    /// Performs one meta-training step using iMAML's implicit gradient computation.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch.</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// iMAML meta-training differs from MAML in how meta-gradients are computed:
    /// </para>
    /// <para>
    /// <b>Inner Loop (Same as MAML):</b>
    /// For each task in the batch:
    /// 1. Clone the meta-model with current meta-parameters
    /// 2. Perform K gradient descent steps on the task's support set
    /// 3. Evaluate the adapted model on the task's query set
    /// </para>
    /// <para>
    /// <b>Implicit Gradient Computation (Different from MAML):</b>
    /// Instead of backpropagating through K steps:
    /// 1. Compute gradient of query loss w.r.t. adapted parameters
    /// 2. Solve (I + lambda * H)^(-1) * g using Conjugate Gradient
    /// 3. This gives the implicit meta-gradient with constant memory cost
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The key difference is step 2 - instead of remembering
    /// all K adaptation steps (expensive!), iMAML uses a mathematical trick to
    /// get the same answer without storing the intermediate steps.
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

            // Compute implicit meta-gradients (the key difference from MAML)
            var taskMetaGradients = ComputeImplicitMetaGradients(taskModel, initialParams, adaptedParams, task);

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
        if (_imamlOptions.GradientClipThreshold.HasValue && _imamlOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedMetaGradients = ClipGradients(accumulatedMetaGradients, _imamlOptions.GradientClipThreshold.Value);
        }

        // Outer loop: Update meta-parameters
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = ApplyGradients(currentMetaParams, accumulatedMetaGradients, _imamlOptions.OuterLearningRate);
        MetaModel.SetParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(totalMetaLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using iMAML's inner loop optimization.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been fine-tuned to the given task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// At adaptation time, iMAML works exactly like MAML - the implicit gradient
    /// computation is only used during meta-training. This means adaptation is
    /// fast and straightforward: just run K gradient steps on the support set.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When you have a new task and want to adapt to it,
    /// iMAML works just like MAML. The memory savings from implicit gradients
    /// only matter during the meta-training phase, not during adaptation.
    /// </para>
    /// <para>
    /// Because iMAML was meta-trained with many adaptation steps (enabled by
    /// constant memory cost), the learned initialization is often better than
    /// MAML's, leading to better adaptation performance.
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

        // Perform inner loop adaptation (same as MAML)
        var adaptedParameters = InnerLoopAdaptation(adaptedModel, task);
        adaptedModel.SetParameters(adaptedParameters);

        return adaptedModel;
    }

    /// <summary>
    /// Performs the inner loop adaptation to a specific task.
    /// </summary>
    /// <param name="model">The model to adapt.</param>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>The adapted parameters after K gradient steps.</returns>
    /// <remarks>
    /// <para>
    /// The inner loop is identical to MAML: perform K gradient descent steps
    /// on the support set. iMAML's innovation is in how meta-gradients are
    /// computed, not in the adaptation process itself.
    /// </para>
    /// </remarks>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Perform K gradient steps on the support set
        for (int step = 0; step < _imamlOptions.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Apply gradients with inner learning rate
            parameters = ApplyGradients(parameters, gradients, _imamlOptions.InnerLearningRate);
            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Computes implicit meta-gradients using the implicit function theorem.
    /// </summary>
    /// <param name="model">The adapted model.</param>
    /// <param name="initialParams">The initial parameters before adaptation.</param>
    /// <param name="adaptedParams">The adapted parameters after inner loop.</param>
    /// <param name="task">The task being adapted to.</param>
    /// <returns>The implicit meta-gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This is the core innovation of iMAML. Instead of backpropagating through
    /// all K adaptation steps (which requires O(K) memory), we use the implicit
    /// function theorem to compute gradients directly.
    /// </para>
    /// <para>
    /// The implicit equation is: (I + lambda * H) * v = g_query
    /// where:
    /// - H is the Hessian of the support loss at adapted parameters
    /// - g_query is the gradient of query loss at adapted parameters
    /// - lambda is the regularization parameter
    /// - v is the implicit meta-gradient we want to compute
    /// </para>
    /// <para>
    /// We solve this using Conjugate Gradient iteration, which only requires
    /// Hessian-vector products (not the full Hessian matrix).
    /// </para>
    /// </remarks>
    private Vector<T> ComputeImplicitMetaGradients(
        IFullModel<T, TInput, TOutput> model,
        Vector<T> initialParams,
        Vector<T> adaptedParams,
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Set model to adapted parameters
        model.SetParameters(adaptedParams);

        // Compute gradient of query loss with respect to adapted parameters
        var queryGradients = ComputeGradients(model, task.QueryInput, task.QueryOutput);

        // If using first-order approximation, just return query gradients
        if (_imamlOptions.UseFirstOrder)
        {
            return queryGradients;
        }

        // Solve the implicit equation using Conjugate Gradient
        // (I + lambda * H) * v = g_query
        var implicitGradients = SolveImplicitEquation(model, task, queryGradients);

        return implicitGradients;
    }

    /// <summary>
    /// Solves the implicit equation (I + lambda * H) * v = b using Conjugate Gradient.
    /// </summary>
    /// <param name="model">The model at adapted parameters.</param>
    /// <param name="task">The task for computing Hessian-vector products.</param>
    /// <param name="b">The right-hand side vector (query gradients).</param>
    /// <returns>The solution vector v (implicit meta-gradients).</returns>
    private Vector<T> SolveImplicitEquation(
        IFullModel<T, TInput, TOutput> model,
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> b)
    {
        int n = b.Length;
        var x = new Vector<T>(n); // Initial guess: zero vector
        var r = new Vector<T>(n);
        var p = new Vector<T>(n);

        // r = b - Ax (with x = 0, r = b)
        for (int i = 0; i < n; i++)
        {
            r[i] = b[i];
            p[i] = r[i];
        }

        T rsOld = DotProduct(r, r);
        T tolerance = NumOps.FromDouble(_imamlOptions.ConjugateGradientTolerance);

        for (int iter = 0; iter < _imamlOptions.ConjugateGradientIterations; iter++)
        {
            // Check convergence
            if (NumOps.ToDouble(rsOld) < _imamlOptions.ConjugateGradientTolerance)
            {
                break;
            }

            // Compute A * p = (I + lambda * H) * p
            var Ap = ComputeImplicitMatrixVectorProduct(model, task, p);

            T pAp = DotProduct(p, Ap);

            // Avoid division by zero
            if (NumOps.ToDouble(pAp) < 1e-12)
            {
                break;
            }

            T alpha = NumOps.Divide(rsOld, pAp);

            // x = x + alpha * p
            for (int i = 0; i < n; i++)
            {
                x[i] = NumOps.Add(x[i], NumOps.Multiply(alpha, p[i]));
            }

            // r = r - alpha * Ap
            for (int i = 0; i < n; i++)
            {
                r[i] = NumOps.Subtract(r[i], NumOps.Multiply(alpha, Ap[i]));
            }

            T rsNew = DotProduct(r, r);

            // Avoid division by zero
            if (NumOps.ToDouble(rsOld) < 1e-12)
            {
                break;
            }

            T beta = NumOps.Divide(rsNew, rsOld);

            // p = r + beta * p
            for (int i = 0; i < n; i++)
            {
                p[i] = NumOps.Add(r[i], NumOps.Multiply(beta, p[i]));
            }

            rsOld = rsNew;
        }

        return x;
    }

    /// <summary>
    /// Computes the implicit matrix-vector product (I + lambda * H) * v.
    /// </summary>
    /// <param name="model">The model at adapted parameters.</param>
    /// <param name="task">The task for computing Hessian-vector products.</param>
    /// <param name="v">The vector to multiply.</param>
    /// <returns>The result of (I + lambda * H) * v.</returns>
    /// <remarks>
    /// <para>
    /// The Hessian-vector product H * v is computed efficiently using finite differences:
    /// H * v approximately equals (grad(theta + eps*v) - grad(theta - eps*v)) / (2*eps)
    /// This avoids forming the full Hessian matrix.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeImplicitMatrixVectorProduct(
        IFullModel<T, TInput, TOutput> model,
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> v)
    {
        T lambda = NumOps.FromDouble(_imamlOptions.LambdaRegularization);
        T eps = NumOps.FromDouble(1e-5);

        // Get current parameters
        var theta = model.GetParameters();

        // Compute gradient at theta + eps*v
        var thetaPlus = new Vector<T>(theta.Length);
        for (int i = 0; i < theta.Length; i++)
        {
            thetaPlus[i] = NumOps.Add(theta[i], NumOps.Multiply(eps, v[i]));
        }
        model.SetParameters(thetaPlus);
        var gradPlus = ComputeGradients(model, task.SupportInput, task.SupportOutput);

        // Compute gradient at theta - eps*v
        var thetaMinus = new Vector<T>(theta.Length);
        for (int i = 0; i < theta.Length; i++)
        {
            thetaMinus[i] = NumOps.Subtract(theta[i], NumOps.Multiply(eps, v[i]));
        }
        model.SetParameters(thetaMinus);
        var gradMinus = ComputeGradients(model, task.SupportInput, task.SupportOutput);

        // Restore original parameters
        model.SetParameters(theta);

        // H * v = (gradPlus - gradMinus) / (2 * eps)
        var Hv = new Vector<T>(v.Length);
        T twoEps = NumOps.Multiply(NumOps.FromDouble(2.0), eps);
        for (int i = 0; i < v.Length; i++)
        {
            Hv[i] = NumOps.Divide(NumOps.Subtract(gradPlus[i], gradMinus[i]), twoEps);
        }

        // Result = I*v + lambda*H*v = v + lambda*Hv
        var result = new Vector<T>(v.Length);
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = NumOps.Add(v[i], NumOps.Multiply(lambda, Hv[i]));
        }

        return result;
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The dot product sum(a[i] * b[i]).</returns>
    private T DotProduct(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
    }
}
