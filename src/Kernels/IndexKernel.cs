namespace AiDotNet.Kernels;

/// <summary>
/// Index kernel for multi-task/multi-output Gaussian Processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The IndexKernel enables GPs to model multiple related tasks or outputs
/// simultaneously, sharing information between them.
///
/// In multi-task learning, you have multiple related prediction tasks. For example:
/// - Predicting exam scores for multiple subjects (math, science, english)
/// - Forecasting sales for multiple products
/// - Estimating properties of multiple chemical compounds
///
/// The IndexKernel models the correlation between tasks using a covariance matrix:
/// k_task(t, t') = B[t, t']
///
/// Where B is a positive semi-definite "task covariance matrix" that captures:
/// - B[t, t]: The variance of task t
/// - B[t, t']: The covariance between tasks t and t'
///
/// When combined with an input kernel (e.g., RBF), the full multi-task kernel is:
/// k((x, t), (x', t')) = k_input(x, x') × k_task(t, t')
///                     = k_input(x, x') × B[t, t']
///
/// This allows the GP to:
/// - Share information between similar tasks
/// - Leverage data from data-rich tasks to help data-poor tasks
/// - Learn task relationships from data
/// </para>
/// <para>
/// Usage patterns:
/// 1. Initialize with a task covariance matrix (if known)
/// 2. Initialize randomly and optimize the task covariance
/// 3. Use with ProductKernel to combine with input kernel
/// </para>
/// </remarks>
public class IndexKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The task covariance matrix (B).
    /// </summary>
    private readonly double[,] _taskCovariance;

    /// <summary>
    /// Number of tasks.
    /// </summary>
    private readonly int _numTasks;

    /// <summary>
    /// Lower Cholesky factor of the task covariance (for parameterization).
    /// </summary>
    private readonly double[,]? _choleskyFactor;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes an IndexKernel with a specified task covariance matrix.
    /// </summary>
    /// <param name="taskCovariance">The task covariance matrix B.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an IndexKernel with known task correlations.
    ///
    /// The task covariance matrix should be:
    /// - Symmetric: B[i,j] = B[j,i]
    /// - Positive semi-definite: All eigenvalues ≥ 0
    ///
    /// Example for 3 tasks with known structure:
    /// var B = new double[,] {
    ///     { 1.0, 0.8, 0.2 },  // Task 0 correlated with task 1
    ///     { 0.8, 1.0, 0.3 },  // Task 1 correlated with task 0
    ///     { 0.2, 0.3, 1.0 }   // Task 2 somewhat independent
    /// };
    /// var kernel = new IndexKernel&lt;double&gt;(B);
    /// </para>
    /// </remarks>
    public IndexKernel(double[,] taskCovariance)
    {
        if (taskCovariance is null) throw new ArgumentNullException(nameof(taskCovariance));
        if (taskCovariance.GetLength(0) != taskCovariance.GetLength(1))
            throw new ArgumentException("Task covariance matrix must be square.", nameof(taskCovariance));

        int n = taskCovariance.GetLength(0);
        if (n < 1)
            throw new ArgumentException("Must have at least one task.", nameof(taskCovariance));

        _numTasks = n;
        _taskCovariance = (double[,])taskCovariance.Clone();
        _numOps = MathHelper.GetNumericOperations<T>();

        // Verify symmetry and compute Cholesky for future updates
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (Math.Abs(_taskCovariance[i, j] - _taskCovariance[j, i]) > 1e-10)
                    throw new ArgumentException("Task covariance matrix must be symmetric.");
            }
        }
    }

    /// <summary>
    /// Initializes an IndexKernel with specified number of tasks using random initialization.
    /// </summary>
    /// <param name="numTasks">Number of tasks.</param>
    /// <param name="rank">Rank of the task covariance matrix. If null, uses full rank.</param>
    /// <param name="variance">Initial variance for each task.</param>
    /// <param name="seed">Random seed for initialization.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an IndexKernel with random task correlations.
    ///
    /// The rank parameter controls the complexity of task relationships:
    /// - Full rank (rank = numTasks): Can model arbitrary task correlations
    /// - Low rank (rank &lt; numTasks): Assumes tasks share some underlying factors
    ///
    /// Low-rank models are useful when:
    /// - You have many tasks but expect them to share structure
    /// - You want to prevent overfitting
    /// - You have prior knowledge that tasks are driven by a few common factors
    /// </para>
    /// </remarks>
    public IndexKernel(int numTasks, int? rank = null, double variance = 1.0, int? seed = null)
    {
        if (numTasks < 1)
            throw new ArgumentException("Must have at least one task.", nameof(numTasks));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        _numTasks = numTasks;
        int effectiveRank = rank ?? numTasks;
        if (effectiveRank < 1 || effectiveRank > numTasks)
            throw new ArgumentException("Rank must be between 1 and numTasks.", nameof(rank));

        var rand = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize via low-rank factorization: B = LLᵀ + diag(v)
        _choleskyFactor = new double[numTasks, effectiveRank];
        _taskCovariance = new double[numTasks, numTasks];

        // Random factor matrix
        double factorScale = Math.Sqrt(variance / effectiveRank);
        for (int i = 0; i < numTasks; i++)
        {
            for (int j = 0; j < effectiveRank; j++)
            {
                _choleskyFactor[i, j] = (rand.NextDouble() - 0.5) * 2 * factorScale;
            }
        }

        // Compute B = LLᵀ + small diagonal for stability
        for (int i = 0; i < numTasks; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double val = 0;
                for (int k = 0; k < effectiveRank; k++)
                {
                    val += _choleskyFactor[i, k] * _choleskyFactor[j, k];
                }

                if (i == j)
                {
                    val += 0.1 * variance; // Diagonal jitter
                }

                _taskCovariance[i, j] = val;
                _taskCovariance[j, i] = val;
            }
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the number of tasks.
    /// </summary>
    public int NumTasks => _numTasks;

    /// <summary>
    /// Gets a copy of the task covariance matrix.
    /// </summary>
    /// <returns>The task covariance matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the matrix B that captures task relationships.
    /// After fitting a multi-task GP, inspect this to understand how tasks relate.
    /// </para>
    /// </remarks>
    public double[,] GetTaskCovariance()
    {
        return (double[,])_taskCovariance.Clone();
    }

    /// <summary>
    /// Gets the correlation between two tasks.
    /// </summary>
    /// <param name="task1">First task index.</param>
    /// <param name="task2">Second task index.</param>
    /// <returns>Correlation coefficient between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the correlation (not covariance) between two tasks.
    /// - 1.0: Perfectly positively correlated (tasks move together)
    /// - 0.0: Uncorrelated (tasks are independent)
    /// - -1.0: Perfectly negatively correlated (tasks move opposite)
    /// </para>
    /// </remarks>
    public double GetTaskCorrelation(int task1, int task2)
    {
        if (task1 < 0 || task1 >= _numTasks)
            throw new ArgumentOutOfRangeException(nameof(task1));
        if (task2 < 0 || task2 >= _numTasks)
            throw new ArgumentOutOfRangeException(nameof(task2));

        double cov = _taskCovariance[task1, task2];
        double std1 = Math.Sqrt(_taskCovariance[task1, task1]);
        double std2 = Math.Sqrt(_taskCovariance[task2, task2]);

        if (std1 < 1e-10 || std2 < 1e-10) return 0;
        return cov / (std1 * std2);
    }

    /// <summary>
    /// Calculates the task kernel value between two task indices.
    /// </summary>
    /// <param name="x1">Vector containing the task index in the first element.</param>
    /// <param name="x2">Vector containing the task index in the first element.</param>
    /// <returns>The task covariance B[task1, task2].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The input vectors should contain the task index as their first element.
    /// The kernel value is simply the corresponding entry from the task covariance matrix.
    ///
    /// Note: This expects task indices as the first element of the vectors. When using with
    /// ProductKernel, ensure your data format matches this convention.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        // Extract task indices from first element
        int task1 = (int)Math.Round(_numOps.ToDouble(x1[0]));
        int task2 = (int)Math.Round(_numOps.ToDouble(x2[0]));

        if (task1 < 0 || task1 >= _numTasks || task2 < 0 || task2 >= _numTasks)
        {
            // Return 0 for out-of-range tasks
            return _numOps.FromDouble(0);
        }

        return _numOps.FromDouble(_taskCovariance[task1, task2]);
    }

    /// <summary>
    /// Gets the task covariance for explicit task indices.
    /// </summary>
    /// <param name="task1">First task index.</param>
    /// <param name="task2">Second task index.</param>
    /// <returns>The covariance between the two tasks.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Direct access to the task covariance without vector wrapping.
    /// Use this for efficiency when you already have task indices.
    /// </para>
    /// </remarks>
    public double CalculateForTasks(int task1, int task2)
    {
        if (task1 < 0 || task1 >= _numTasks)
            throw new ArgumentOutOfRangeException(nameof(task1));
        if (task2 < 0 || task2 >= _numTasks)
            throw new ArgumentOutOfRangeException(nameof(task2));

        return _taskCovariance[task1, task2];
    }

    /// <summary>
    /// Creates an IndexKernel with identity task covariance (independent tasks).
    /// </summary>
    /// <param name="numTasks">Number of tasks.</param>
    /// <param name="variance">Variance for each task.</param>
    /// <returns>An IndexKernel with B = variance × I.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates independent tasks (no information sharing).
    /// This is equivalent to running separate GPs for each task.
    ///
    /// Use as a baseline or when you believe tasks are truly independent.
    /// </para>
    /// </remarks>
    public static IndexKernel<T> Independent(int numTasks, double variance = 1.0)
    {
        var B = new double[numTasks, numTasks];
        for (int i = 0; i < numTasks; i++)
        {
            B[i, i] = variance;
        }
        return new IndexKernel<T>(B);
    }

    /// <summary>
    /// Creates an IndexKernel with uniform task correlation.
    /// </summary>
    /// <param name="numTasks">Number of tasks.</param>
    /// <param name="correlation">Correlation between all task pairs.</param>
    /// <param name="variance">Variance for each task.</param>
    /// <returns>An IndexKernel with uniform off-diagonal correlation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates tasks that are all equally correlated with each other.
    ///
    /// For example, with correlation = 0.5:
    /// - All tasks have the same variance
    /// - All pairs of tasks have correlation 0.5
    ///
    /// This is a simple model when you believe all tasks are roughly equally related.
    /// </para>
    /// </remarks>
    public static IndexKernel<T> UniformCorrelation(int numTasks, double correlation, double variance = 1.0)
    {
        if (correlation < -1.0 / (numTasks - 1) || correlation > 1.0)
            throw new ArgumentException(
                $"Correlation must be between {-1.0 / (numTasks - 1):F3} and 1.0 for the matrix to be positive definite.",
                nameof(correlation));

        var B = new double[numTasks, numTasks];
        double covariance = correlation * variance;

        for (int i = 0; i < numTasks; i++)
        {
            for (int j = 0; j < numTasks; j++)
            {
                B[i, j] = (i == j) ? variance : covariance;
            }
        }

        return new IndexKernel<T>(B);
    }
}
