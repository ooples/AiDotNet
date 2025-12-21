using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
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
///   3. If g would increase loss on any previous task (i.e., g · g_k &lt; margin):
///      - Project g to the nearest gradient that satisfies g' · g_k ≥ margin for all k
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
    /// <param name="margin">Margin for constraint violations (epsilon in the paper). Default is 0.0.</param>
    public GradientEpisodicMemory(
        ILossFunction<T> lossFunction,
        int memorySize = 256,
        double margin = 0.0)
    {
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _memorySize = memorySize;
        _margin = NumOps.FromDouble(margin);
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
    /// <remarks>
    /// <para><b>GEM-Specific Implementation:</b> This method should store the average gradient
    /// computed on examples from the completed task for future constraint checking.</para>
    ///
    /// <para><b>Current Limitation:</b> The IFullModel interface does not expose gradient
    /// computation methods. To use GEM properly in production, you need to:
    /// 1. Compute gradients during training using your optimizer/framework
    /// 2. Call a separate method to store these gradients (see <see cref="StoreTaskGradient"/>)
    /// </para>
    ///
    /// <para>This method currently stores a zero gradient as a placeholder to maintain
    /// the task count. The gradient projection in <see cref="AdjustGradients"/> will still
    /// function but without actual constraint enforcement until real gradients are provided.</para>
    /// </remarks>
    public void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // Store a zero gradient placeholder to maintain task count.
        // In practice, use StoreTaskGradient() during training to provide real gradients.
        var zeroGradient = new Vector<T>(model.ParameterCount);
        // Initialize to zero (default(T) for each element)
        for (int i = 0; i < zeroGradient.Length; i++)
        {
            zeroGradient[i] = NumOps.Zero;
        }

        _taskGradients.Add(zeroGradient);
    }

    /// <summary>
    /// Stores the reference gradient for a completed task.
    /// </summary>
    /// <param name="taskGradient">The average gradient on task examples.</param>
    /// <remarks>
    /// <para><b>For Production Use:</b> Call this method after computing the average gradient
    /// on examples from a completed task. This gradient will be used as a constraint
    /// in future tasks to prevent catastrophic forgetting.</para>
    ///
    /// <para>To compute the task gradient:
    /// 1. Sample examples from the task (stored in episodic memory)
    /// 2. Compute loss on these examples
    /// 3. Compute gradients via backpropagation
    /// 4. Average the gradients across examples
    /// 5. Pass the result to this method
    /// </para>
    /// </remarks>
    public void StoreTaskGradient(Vector<T> taskGradient)
    {
        if (taskGradient == null)
            throw new ArgumentNullException(nameof(taskGradient));

        // Replace the most recent placeholder gradient or add new one
        if (_taskGradients.Count > 0)
        {
            _taskGradients[_taskGradients.Count - 1] = taskGradient;
        }
        else
        {
            _taskGradients.Add(taskGradient);
        }
    }

    /// <summary>
    /// Stores examples from a task in the episodic memory.
    /// </summary>
    public void StoreTaskExamples(IDataset<T, TInput, TOutput> taskData)
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
    /// subject to: g' · g_k ≥ margin for all k in violations
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

        var projected = gradient.ToArray();

        // For each violated constraint, adjust the gradient
        foreach (var taskGrad in violations.Select(k => _taskGradients[k]))
        {
            T dotProduct = ComputeDotProduct(gradient, taskGrad);

            // If violation is severe, blend more with the task gradient
            if (Convert.ToDouble(dotProduct) < 0)
            {
                // Simple projection: g' = g - (g · g_k) * g_k / ||g_k||^2
                T normSq = ComputeDotProduct(taskGrad, taskGrad);
                T factor = NumOps.Divide(dotProduct, normSq);

                projected = projected.Select((value, i) =>
                {
                    var adjustment = NumOps.Multiply(factor, taskGrad[i]);
                    return NumOps.Subtract(value, adjustment);
                }).ToArray();
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
