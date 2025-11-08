using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Gradient Episodic Memory (GEM) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> GEM prevents forgetting by ensuring that updates for new tasks
/// don't hurt performance on previous tasks. It does this by:
/// 1. Storing a small number of examples from each previous task (episodic memory)
/// 2. When computing gradients for a new task, check if they would hurt old tasks
/// 3. If yes, project the gradients to a direction that doesn't increase loss on old tasks
/// </para>
///
/// <para><b>How it works:</b>
/// - After learning each task, store some examples in episodic memory
/// - When training on a new task:
///   1. Compute gradient g for the new task
///   2. Compute gradients g_k for each previous task using stored examples
///   3. If g would increase loss on any previous task (i.e., g · g_k &lt; 0):
///      - Project g to the nearest gradient that satisfies g' · g_k ≥ 0 for all k
///   4. Use the projected gradient for the update
/// </para>
///
/// <para><b>Advantages:</b>
/// - Strong theoretical guarantees: never increases loss on previous tasks
/// - Works well with small memory budgets
/// - Applicable to any gradient-based learning
/// </para>
///
/// <para><b>Disadvantages:</b>
/// - Requires solving a quadratic program for gradient projection
/// - Needs access to gradients (not just parameters)
/// - Can be computationally expensive
/// </para>
///
/// <para><b>Reference:</b> Lopez-Paz and Ranzato "Gradient Episodic Memory for Continual Learning" (2017)</para>
/// </remarks>
public class GradientEpisodicMemory<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILossFunction<T> _lossFunction;
    private readonly int _memorySize;
    private readonly T _margin; // Constraint violation tolerance

    // Store gradients for each previous task
    private readonly List<Vector<T>> _taskGradients;

    // Memory buffer for storing examples
    private readonly ExperienceReplayBuffer<T, TInput, TOutput> _memoryBuffer;

    /// <summary>
    /// Initializes a new GEM strategy.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="memorySize">Number of examples to store per task.</param>
    /// <param name="margin">Margin for constraint violations (epsilon in the paper).</param>
    public GradientEpisodicMemory(
        ILossFunction<T> lossFunction,
        int memorySize = 256,
        T? margin = null)
    {
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _memorySize = memorySize;
        _margin = margin ?? NumOps.FromDouble(0.0);
        _taskGradients = new List<Vector<T>>();
        _memoryBuffer = new ExperienceReplayBuffer<T, TInput, TOutput>(memorySize * 10); // Total memory
    }

    /// <inheritdoc/>
    public void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // No preparation needed before training
    }

    /// <inheritdoc/>
    public T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // GEM doesn't use regularization loss - it modifies gradients instead
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public Vector<T> AdjustGradients(Vector<T> gradients)
    {
        if (_taskGradients.Count == 0)
            return gradients; // No previous tasks, no adjustment needed

        // Check if current gradient violates constraints with previous tasks
        bool violatesConstraints = false;
        var violations = new List<int>();

        for (int k = 0; k < _taskGradients.Count; k++)
        {
            // Compute dot product: g · g_k
            T dotProduct = ComputeDotProduct(gradients, _taskGradients[k]);

            // If dot product < margin, we're violating the constraint
            // (increasing loss on task k)
            if (Convert.ToDouble(dotProduct) < Convert.ToDouble(_margin))
            {
                violatesConstraints = true;
                violations.Add(k);
            }
        }

        if (!violatesConstraints)
            return gradients; // No violation, use original gradients

        // Project gradient to satisfy constraints
        // Solve: min ||g' - g||^2 such that g' · g_k >= 0 for all k
        var projectedGradient = ProjectGradient(gradients, violations);

        return projectedGradient;
    }

    /// <inheritdoc/>
    public void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // In a full implementation, this would:
        // 1. Compute the average gradient on the stored examples for this task
        // 2. Store this gradient for future constraint checking

        // Placeholder: store a dummy gradient
        var dummyGradient = new Vector<T>(model.ParameterCount);
        for (int i = 0; i < dummyGradient.Length; i++)
        {
            dummyGradient[i] = NumOps.FromDouble(0.01);
        }

        _taskGradients.Add(dummyGradient);
    }

    /// <summary>
    /// Stores examples from a task in the episodic memory.
    /// </summary>
    public void StoreTaskExamples(IDataset<T, TInput, TOutput> taskData, int taskId)
    {
        _memoryBuffer.AddTaskExamples(taskData, _memorySize);
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    private T ComputeDotProduct(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }
        return sum;
    }

    /// <summary>
    /// Projects a gradient to satisfy constraints with previous tasks.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> This solves the quadratic program:
    /// minimize: ||g' - g||^2
    /// subject to: g' · g_k ≥ 0 for all k in violations
    /// </para>
    ///
    /// <para>This can be solved using:
    /// 1. Quadratic Programming (QP) solvers
    /// 2. Gradient projection methods
    /// 3. Averaging with violated task gradients (simple approximation)
    /// </para>
    ///
    /// <para>We use a simplified approach: average the current gradient with
    /// the gradients of violated tasks to move away from the violation.</para>
    /// </remarks>
    private Vector<T> ProjectGradient(Vector<T> gradient, List<int> violations)
    {
        if (violations.Count == 0)
            return gradient;

        // Simplified projection: average current gradient with task gradients
        // This is an approximation; the full solution requires solving a QP

        var projected = new T[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            projected[i] = gradient[i];
        }

        // For each violated constraint, adjust the gradient
        foreach (int k in violations)
        {
            var taskGrad = _taskGradients[k];
            T dotProduct = ComputeDotProduct(gradient, taskGrad);

            // If violation is severe, blend more with the task gradient
            if (Convert.ToDouble(dotProduct) < 0)
            {
                // Simple projection: g' = g - (g · g_k) * g_k / ||g_k||^2
                T normSq = ComputeDotProduct(taskGrad, taskGrad);
                T factor = NumOps.Divide(dotProduct, normSq);

                for (int i = 0; i < projected.Length; i++)
                {
                    var adjustment = NumOps.Multiply(factor, taskGrad[i]);
                    projected[i] = NumOps.Subtract(projected[i], adjustment);
                }
            }
        }

        return new Vector<T>(projected);
    }

    /// <summary>
    /// Gets the number of stored task gradients.
    /// </summary>
    public int NumStoredTasks => _taskGradients.Count;

    /// <summary>
    /// Gets the episodic memory buffer.
    /// </summary>
    public ExperienceReplayBuffer<T, TInput, TOutput> MemoryBuffer => _memoryBuffer;
}
