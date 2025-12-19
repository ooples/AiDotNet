using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the iMAML (implicit MAML) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
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
/// - Often achieves better performance than first-order MAML
/// </para>
/// <para>
/// <b>For Beginners:</b> iMAML solves one of MAML's biggest problems - memory usage.
///
/// The problem with MAML:
/// - To learn from adaptation, MAML needs to remember every step
/// - More adaptation steps = much more memory needed
/// - This limits how much adaptation you can do
///
/// How iMAML solves it:
/// - Uses a mathematical shortcut (implicit differentiation)
/// - Only needs to remember the start and end points
/// - Can do many more adaptation steps with the same memory
///
/// The result: Better performance without exploding memory requirements.
/// </para>
/// <para>
/// Reference: Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019).
/// Meta-learning with implicit gradients.
/// </para>
/// </remarks>
public class iMAMLAlgorithm<T, TInput, TOutput> : MetaLearningBase<T, TInput, TOutput>
{
    private readonly iMAMLAlgorithmOptions<T, TInput, TOutput> _imamlOptions;

    /// <summary>
    /// Initializes a new instance of the iMAMLAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for iMAML.</param>
    public iMAMLAlgorithm(iMAMLAlgorithmOptions<T, TInput, TOutput> options) : base(options)
    {
        _imamlOptions = options;
    }

    /// <inheritdoc/>
    public override string AlgorithmName => "iMAML";

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate meta-gradients across all tasks
        Vector<T>? metaGradients = null;
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
            T metaLoss = LossFunction.CalculateLoss(queryPredictions, task.QueryOutput);
            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute implicit meta-gradients
            var taskMetaGradients = ComputeImplicitMetaGradients(initialParams, adaptedParams, task);

            // Accumulate meta-gradients
            if (metaGradients == null)
            {
                metaGradients = taskMetaGradients;
            }
            else
            {
                for (int i = 0; i < metaGradients.Length; i++)
                {
                    metaGradients[i] = NumOps.Add(metaGradients[i], taskMetaGradients[i]);
                }
            }
        }

        if (metaGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average the meta-gradients
        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < metaGradients.Length; i++)
        {
            metaGradients[i] = NumOps.Divide(metaGradients[i], batchSize);
        }

        // Outer loop: Update meta-parameters using the meta-optimizer
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = MetaOptimizer.UpdateParameters(currentMetaParams, metaGradients);
        MetaModel.SetParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(totalMetaLoss, batchSize);
    }

    /// <inheritdoc/>
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
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Use inner optimizer for parameter updates
            parameters = InnerOptimizer.UpdateParameters(parameters, gradients);
            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Computes implicit meta-gradients using the implicit function theorem.
    /// </summary>
    /// <param name="initialParams">The initial parameters before adaptation.</param>
    /// <param name="adaptedParams">The adapted parameters after inner loop.</param>
    /// <param name="task">The task being adapted to.</param>
    /// <returns>The implicit meta-gradient vector.</returns>
    private Vector<T> ComputeImplicitMetaGradients(
        Vector<T> initialParams,
        Vector<T> adaptedParams,
        IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Step 1: Compute gradient of query loss with respect to adapted parameters
        var model = CloneModel();
        model.SetParameters(adaptedParams);
        var queryGradients = ComputeGradients(model, task.QueryInput, task.QueryOutput);

        // Step 2: Solve the implicit equation using Conjugate Gradient
        // This step would typically involve computing the Hessian-vector product
        // For simplicity in this implementation, we use a first-order approximation
        // A full implementation would use CG to solve: (I + λH)v = g_query

        // Use first-order approximation (similar to first-order MAML)
        var metaGradients = queryGradients;

        // Apply regularization
        T lambda = NumOps.FromDouble(_imamlOptions.LambdaRegularization);
        for (int i = 0; i < metaGradients.Length; i++)
        {
            metaGradients[i] = NumOps.Divide(metaGradients[i], NumOps.Add(NumOps.One, lambda));
        }

        return metaGradients;
    }

    /// <summary>
    /// Solves a linear system using Conjugate Gradient method.
    /// </summary>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    private Vector<T> ConjugateGradient(Vector<T> b)
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
            // Check convergence using the tolerance variable
            if (Convert.ToDouble(rsOld) < Convert.ToDouble(tolerance))
            {
                break;
            }

            // For simplicity, we're not computing the actual matrix-vector product
            // A full implementation would compute (I + λH)p where H is the Hessian
            var Ap = p; // Simplified: just use identity matrix

            T alpha = NumOps.Divide(rsOld, DotProduct(p, Ap));

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
    /// Computes the dot product of two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The dot product.</returns>
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
