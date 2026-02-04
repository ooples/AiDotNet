namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Multi-Task Gaussian Process for modeling multiple correlated outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Multi-Task GP models multiple related output variables simultaneously,
/// learning the correlations between tasks to improve predictions for all of them.
///
/// Example scenarios:
/// - Predicting temperature, humidity, and pressure at weather stations (related measurements)
/// - Forecasting sales across multiple product lines (correlated markets)
/// - Modeling grades across subjects for students (abilities correlate)
///
/// Why use Multi-Task GP instead of separate GPs?
///
/// 1. **Information sharing**: If task A has lots of data and task B has little,
///    the Multi-Task GP can use A's data to help predict B
///
/// 2. **Correlation modeling**: Learns how tasks relate (e.g., when temperature rises,
///    ice cream sales increase)
///
/// 3. **Better uncertainty**: More accurate confidence intervals by considering
///    task relationships
///
/// The model uses a coregionalization approach:
/// - A base kernel captures input similarity
/// - A task correlation matrix captures how tasks relate
/// - The combined kernel is their product (Kronecker structure)
/// </para>
/// </remarks>
public class MultiTaskGaussianProcess<T>
{
    /// <summary>
    /// The base kernel for input similarity.
    /// </summary>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The training input data.
    /// </summary>
    private Matrix<T> _X;

    /// <summary>
    /// The training target values (multi-output).
    /// </summary>
    private Matrix<T> _Y;

    /// <summary>
    /// The number of tasks (output dimensions).
    /// </summary>
    private readonly int _numTasks;

    /// <summary>
    /// The task correlation matrix (B matrix in ICM/LMC models).
    /// </summary>
    private Matrix<T> _taskCovariance;

    /// <summary>
    /// Cholesky factor of the task covariance.
    /// </summary>
    private Matrix<T> _taskCovCholesky;

    /// <summary>
    /// The combined kernel matrix.
    /// </summary>
    private Matrix<T> _K;

    /// <summary>
    /// The alpha vector for predictions (K^(-1) * y).
    /// </summary>
    private Vector<T> _alpha;

    /// <summary>
    /// Operations for numeric calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Matrix decomposition method.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// Observation noise variance.
    /// </summary>
    private readonly double _noiseVariance;

    /// <summary>
    /// Whether to learn task correlations from data.
    /// </summary>
    private readonly bool _learnTaskCorrelations;

    /// <summary>
    /// Initializes a new Multi-Task Gaussian Process.
    /// </summary>
    /// <param name="kernel">The base kernel for input similarity.</param>
    /// <param name="numTasks">The number of tasks (output dimensions).</param>
    /// <param name="noiseVariance">Observation noise variance. Default is 1e-4.</param>
    /// <param name="learnTaskCorrelations">Whether to learn task correlations. Default is true.</param>
    /// <param name="decompositionType">Matrix decomposition method. Default is Cholesky.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Multi-Task GP for modeling correlated outputs.
    ///
    /// Parameters:
    /// - kernel: How to measure input similarity (RBF, Matern, etc.)
    /// - numTasks: How many outputs you're predicting simultaneously
    /// - noiseVariance: Expected measurement noise
    /// - learnTaskCorrelations: If true, learns how tasks relate from data;
    ///   if false, assumes tasks are equally correlated
    ///
    /// Example: Predicting 3 related quantities
    /// var mtgp = new MultiTaskGaussianProcess&lt;double&gt;(rbfKernel, numTasks: 3);
    /// </para>
    /// </remarks>
    public MultiTaskGaussianProcess(
        IKernelFunction<T> kernel,
        int numTasks,
        double noiseVariance = 1e-4,
        bool learnTaskCorrelations = true,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        if (kernel is null)
            throw new ArgumentNullException(nameof(kernel));
        if (numTasks < 1)
            throw new ArgumentException("Must have at least one task.", nameof(numTasks));
        if (noiseVariance < 0)
            throw new ArgumentException("Noise variance must be non-negative.", nameof(noiseVariance));

        _kernel = kernel;
        _numTasks = numTasks;
        _noiseVariance = noiseVariance;
        _learnTaskCorrelations = learnTaskCorrelations;
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();

        _X = Matrix<T>.Empty();
        _Y = Matrix<T>.Empty();
        _K = Matrix<T>.Empty();
        _alpha = Vector<T>.Empty();

        // Initialize task covariance as identity (independent tasks)
        _taskCovariance = CreateIdentityMatrix(numTasks);
        _taskCovCholesky = CreateIdentityMatrix(numTasks);
    }

    /// <summary>
    /// Trains the Multi-Task GP on the provided data.
    /// </summary>
    /// <param name="X">The input features matrix (n × d).</param>
    /// <param name="Y">The multi-output target matrix (n × numTasks).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model to predict multiple outputs simultaneously.
    ///
    /// Input format:
    /// - X: Each row is one input point (e.g., location, time)
    /// - Y: Each row has values for all tasks at that input point
    ///   - Y[i, 0] = task 0's value at input X[i]
    ///   - Y[i, 1] = task 1's value at input X[i]
    ///   - etc.
    ///
    /// The training process:
    /// 1. If enabled, learn task correlations from the data
    /// 2. Build the combined covariance matrix (Kronecker product structure)
    /// 3. Solve for prediction weights
    ///
    /// After training, the model can predict all tasks at new input points.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Matrix<T> Y)
    {
        if (Y.Columns != _numTasks)
            throw new ArgumentException($"Y must have {_numTasks} columns (one per task).", nameof(Y));
        if (X.Rows != Y.Rows)
            throw new ArgumentException("X and Y must have the same number of rows.", nameof(Y));

        _X = X;
        _Y = Y;

        // Learn task correlations if enabled
        if (_learnTaskCorrelations)
        {
            LearnTaskCorrelations();
        }

        // Build combined covariance matrix
        BuildCombinedKernel();

        // Solve for alpha
        ComputeAlpha();
    }

    /// <summary>
    /// Learns the task correlation matrix from the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This estimates how different tasks relate to each other.
    ///
    /// The method computes the empirical covariance between tasks:
    /// - If tasks are positively correlated, they tend to increase/decrease together
    /// - If negatively correlated, one increases when the other decreases
    /// - If uncorrelated (zero), they're independent
    ///
    /// This information helps the model share information between tasks:
    /// - Highly correlated tasks benefit a lot from each other's data
    /// - Uncorrelated tasks are essentially independent
    /// </para>
    /// </remarks>
    private void LearnTaskCorrelations()
    {
        int n = _Y.Rows;

        // Compute empirical covariance between tasks
        var taskMeans = new Vector<T>(_numTasks);
        for (int t = 0; t < _numTasks; t++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = _numOps.Add(sum, _Y[i, t]);
            }
            taskMeans[t] = _numOps.Divide(sum, _numOps.FromDouble(n));
        }

        _taskCovariance = new Matrix<T>(_numTasks, _numTasks);
        for (int t1 = 0; t1 < _numTasks; t1++)
        {
            for (int t2 = 0; t2 < _numTasks; t2++)
            {
                T cov = _numOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T diff1 = _numOps.Subtract(_Y[i, t1], taskMeans[t1]);
                    T diff2 = _numOps.Subtract(_Y[i, t2], taskMeans[t2]);
                    cov = _numOps.Add(cov, _numOps.Multiply(diff1, diff2));
                }
                _taskCovariance[t1, t2] = _numOps.Divide(cov, _numOps.FromDouble(n - 1));
            }
        }

        // Add jitter for stability
        for (int t = 0; t < _numTasks; t++)
        {
            _taskCovariance[t, t] = _numOps.Add(_taskCovariance[t, t], _numOps.FromDouble(1e-6));
        }

        // Compute Cholesky factor
        try
        {
            var chol = new CholeskyDecomposition<T>(_taskCovariance);
            _taskCovCholesky = chol.L;
        }
        catch (ArgumentException ex)
        {
            // Fall back to identity if not positive definite
            System.Diagnostics.Debug.WriteLine($"Task covariance Cholesky failed: {ex.Message}. Using identity.");
            _taskCovariance = CreateIdentityMatrix(_numTasks);
            _taskCovCholesky = CreateIdentityMatrix(_numTasks);
        }
    }

    /// <summary>
    /// Builds the combined covariance matrix using Kronecker structure.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The combined kernel has a special structure:
    ///
    /// K_combined = K_tasks ⊗ K_inputs
    ///
    /// Where ⊗ is the Kronecker product. This means the full covariance between
    /// (input i, task t) and (input j, task s) is:
    ///
    /// K_combined[(i,t), (j,s)] = K_tasks[t,s] × K_inputs[i,j]
    ///
    /// This structure:
    /// - Captures input similarity (K_inputs)
    /// - Captures task relationships (K_tasks)
    /// - Allows efficient computation
    /// </para>
    /// </remarks>
    private void BuildCombinedKernel()
    {
        int n = _X.Rows;
        int totalSize = n * _numTasks;

        // Compute input kernel matrix
        var Kx = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Kx[i, j] = _kernel.Calculate(_X.GetRow(i), _X.GetRow(j));
            }
        }

        // Build combined kernel using Kronecker structure
        _K = new Matrix<T>(totalSize, totalSize);
        for (int t1 = 0; t1 < _numTasks; t1++)
        {
            for (int t2 = 0; t2 < _numTasks; t2++)
            {
                T taskCov = _taskCovariance[t1, t2];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        int row = t1 * n + i;
                        int col = t2 * n + j;
                        _K[row, col] = _numOps.Multiply(taskCov, Kx[i, j]);
                    }
                }
            }
        }

        // Add noise to diagonal
        T noise = _numOps.FromDouble(_noiseVariance);
        for (int i = 0; i < totalSize; i++)
        {
            _K[i, i] = _numOps.Add(_K[i, i], noise);
        }

        // Add jitter
        T jitter = _numOps.FromDouble(1e-6);
        for (int i = 0; i < totalSize; i++)
        {
            _K[i, i] = _numOps.Add(_K[i, i], jitter);
        }
    }

    /// <summary>
    /// Computes the alpha vector for predictions.
    /// </summary>
    private void ComputeAlpha()
    {
        int n = _X.Rows;

        // Flatten Y into a vector (task-major order)
        var yFlat = new Vector<T>(n * _numTasks);
        for (int t = 0; t < _numTasks; t++)
        {
            for (int i = 0; i < n; i++)
            {
                yFlat[t * n + i] = _Y[i, t];
            }
        }

        // Solve K * alpha = y
        _alpha = MatrixSolutionHelper.SolveLinearSystem(_K, yFlat, _decompositionType);
    }

    /// <summary>
    /// Predicts all task outputs for a new input point.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>Tuple of means and variances for each task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Predicts all output variables at once for a new input.
    ///
    /// Returns:
    /// - means: Predicted value for each task (length = numTasks)
    /// - variances: Uncertainty for each task (length = numTasks)
    ///
    /// The predictions account for correlations between tasks, so if one task
    /// has more data, it can help improve predictions for related tasks.
    /// </para>
    /// </remarks>
    public (Vector<T> means, Vector<T> variances) Predict(Vector<T> x)
    {
        if (_X.IsEmpty || _alpha.IsEmpty)
        {
            throw new InvalidOperationException("Model must be trained before prediction. Call Fit() first.");
        }

        int n = _X.Rows;

        // Compute kernel vector between test point and training points
        var kStar = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            kStar[i] = _kernel.Calculate(x, _X.GetRow(i));
        }

        // Compute predictions for each task
        var means = new Vector<T>(_numTasks);
        var variances = new Vector<T>(_numTasks);

        T kStarStar = _kernel.Calculate(x, x);

        for (int t = 0; t < _numTasks; t++)
        {
            // Mean: sum over all (task, datapoint) pairs weighted by alpha
            T mean = _numOps.Zero;
            for (int t2 = 0; t2 < _numTasks; t2++)
            {
                T taskCov = _taskCovariance[t, t2];
                for (int i = 0; i < n; i++)
                {
                    int idx = t2 * n + i;
                    T contribution = _numOps.Multiply(taskCov, _numOps.Multiply(kStar[i], _alpha[idx]));
                    mean = _numOps.Add(mean, contribution);
                }
            }
            means[t] = mean;

            // Variance (simplified - full computation would use Kronecker structure)
            T variance = _numOps.Multiply(_taskCovariance[t, t], kStarStar);

            // Subtract reduction from training data
            var kStarTask = new Vector<T>(n * _numTasks);
            for (int t2 = 0; t2 < _numTasks; t2++)
            {
                T taskCov = _taskCovariance[t, t2];
                for (int i = 0; i < n; i++)
                {
                    kStarTask[t2 * n + i] = _numOps.Multiply(taskCov, kStar[i]);
                }
            }

            var v = MatrixSolutionHelper.SolveLinearSystem(_K, kStarTask, _decompositionType);
            T reduction = _numOps.Zero;
            for (int i = 0; i < kStarTask.Length; i++)
            {
                reduction = _numOps.Add(reduction, _numOps.Multiply(kStarTask[i], v[i]));
            }

            variance = _numOps.Subtract(variance, reduction);
            variance = _numOps.FromDouble(Math.Max(_numOps.ToDouble(variance), 1e-10));
            variances[t] = variance;
        }

        return (means, variances);
    }

    /// <summary>
    /// Gets the learned task correlation matrix.
    /// </summary>
    /// <returns>The task covariance matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the matrix showing how tasks relate.
    ///
    /// Interpreting the matrix:
    /// - Diagonal elements: Variance of each task
    /// - Off-diagonal elements: Covariance between tasks
    ///   - Positive: Tasks tend to move together
    ///   - Negative: Tasks tend to move oppositely
    ///   - Near zero: Tasks are independent
    ///
    /// You can convert to correlations by dividing by sqrt(var1 * var2).
    /// </para>
    /// </remarks>
    public Matrix<T> GetTaskCorrelations() => _taskCovariance;

    /// <summary>
    /// Updates the kernel for this multi-task GP.
    /// </summary>
    /// <param name="kernel">The new kernel function.</param>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        if (!_X.IsEmpty && !_Y.IsEmpty)
        {
            BuildCombinedKernel();
            ComputeAlpha();
        }
    }

    private Matrix<T> CreateIdentityMatrix(int size)
    {
        var matrix = new Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            matrix[i, i] = _numOps.One;
        }
        return matrix;
    }
}
