using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Multi-task Gaussian Process Prediction", "https://doi.org/10.5555/2981562.2981672", Year = 2008, Authors = "Edwin V. Bonilla, Kian Ming A. Chai, Christopher K. I. Williams")]
public partial class MultiTaskGaussianProcess<T> : GaussianProcessBase<T>
{
    /// <summary>
    /// The base kernel for input similarity.
    /// </summary>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The training input data.
    /// </summary>
    [Buffer]
    private Matrix<T> _X;

    /// <summary>
    /// The training target values (multi-output).
    /// </summary>
    [Buffer]
    private Matrix<T> _Y;

    /// <summary>
    /// The number of tasks (output dimensions).
    /// </summary>
    private readonly int _numTasks;

    /// <summary>
    /// The task correlation matrix (B matrix in ICM/LMC models).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _taskCovariance;

    /// <summary>
    /// Cholesky factor of the task covariance.
    /// </summary>
    [Buffer]
    private Matrix<T> _taskCovCholesky;

    /// <summary>
    /// The combined kernel matrix.
    /// </summary>
    [Buffer]
    private Matrix<T> _K;

    /// <summary>
    /// The alpha vector for predictions (K^(-1) * y).
    /// </summary>
    [Buffer]
    private Vector<T> _alpha;

    /// <summary>
    /// Per-task mean of the training targets. The zero-mean prior models each task's deviations
    /// from it, which is also what LearnTaskCorrelations measures when it estimates the amplitude.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _taskMeans;

    /// <summary>
    /// The observation noise variance the model was fitted with: the caller's fixed value, or the
    /// marginal-likelihood estimate when none was given. A single entry.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _fittedNoiseVariance;

    /// <summary>
    /// Operations for numeric calculations.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Matrix decomposition method.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// The caller's fixed observation noise variance, or null to learn it by maximizing the marginal
    /// likelihood.
    /// </summary>
    private readonly double? _noiseVariance;

    /// <summary>
    /// Gradient steps of the marginal-likelihood hyperparameter search.
    /// </summary>
    private readonly int _hyperparameterOptimizationSteps;

    /// <summary>
    /// Whether to learn task correlations from data.
    /// </summary>
    private readonly bool _learnTaskCorrelations;

    /// <summary>
    /// Initializes a new Multi-Task Gaussian Process.
    /// </summary>
    /// <param name="kernel">The base kernel for input similarity.</param>
    /// <param name="numTasks">The number of tasks (output dimensions).</param>
    /// <param name="noiseVariance">Observation noise variance. Null (the default) learns it by
    /// maximizing the marginal likelihood, as Bonilla et al. (2008) do; a value fixes it.</param>
    /// <param name="learnTaskCorrelations">Whether to learn the task covariance B. When true (the
    /// default) B starts at the empirical covariance of the targets and is refined jointly with the
    /// noise by maximizing the marginal likelihood; when false the tasks are independent.</param>
    /// <param name="decompositionType">Matrix decomposition method. Default is Cholesky.</param>
    /// <param name="hyperparameterOptimizationSteps">Gradient steps of the marginal-likelihood search.
    /// Default 100; 0 keeps the starting estimates.</param>
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
        double? noiseVariance = null,
        bool learnTaskCorrelations = true,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky,
        int hyperparameterOptimizationSteps = 100)
    {
        if (kernel is null)
            throw new ArgumentNullException(nameof(kernel));
        if (numTasks < 1)
            throw new ArgumentException("Must have at least one task.", nameof(numTasks));
        if (noiseVariance < 0)
            throw new ArgumentException("Noise variance must be non-negative.", nameof(noiseVariance));
        if (hyperparameterOptimizationSteps < 0)
            throw new ArgumentException("Optimization steps must be non-negative.", nameof(hyperparameterOptimizationSteps));

        _kernel = kernel;
        _numTasks = numTasks;
        _noiseVariance = noiseVariance;
        _hyperparameterOptimizationSteps = hyperparameterOptimizationSteps;
        _learnTaskCorrelations = learnTaskCorrelations;
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();

        _X = Matrix<T>.Empty();
        _Y = Matrix<T>.Empty();
        _K = Matrix<T>.Empty();
        _alpha = Vector<T>.Empty();
        _taskMeans = new Vector<T>(numTasks);
        _fittedNoiseVariance = new Vector<T>(1);

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

        // The prior is zero-mean, so model each task's deviations from its training mean: away from the
        // data a prediction then reverts to that mean rather than to zero. (Inside the data this changes
        // little - a smooth kernel absorbs a constant offset - so it is not a calibration fix.)
        _taskMeans = ComputeTaskMeans(Y);

        // The empirical task covariance is the starting point of the marginal-likelihood search.
        if (_learnTaskCorrelations)
        {
            LearnTaskCorrelations();
        }

        OptimizeHyperparameters();

        // Build combined covariance matrix
        BuildCombinedKernel();

        // Solve for alpha
        ComputeAlpha();
    }

    /// <summary>
    /// Computes each task's mean over the training rows.
    /// </summary>
    private Vector<T> ComputeTaskMeans(Matrix<T> Y)
    {
        var means = new Vector<T>(_numTasks);
        for (int t = 0; t < _numTasks; t++)
        {
            T sum = _numOps.Zero;
            for (int i = 0; i < Y.Rows; i++)
            {
                sum = _numOps.Add(sum, Y[i, t]);
            }
            means[t] = _numOps.Divide(sum, _numOps.FromDouble(Y.Rows));
        }

        return means;
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

        // Compute empirical covariance between tasks, around the means Fit already took.
        var taskMeans = _taskMeans;

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
    /// Learns the task covariance B and the observation noise by type-II maximum likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Maximizes log N(y | 0, K) with K = B (x) K_x + (sigma_n^2 + jitter) I, over B = L L^T (L lower
    /// triangular, diagonal on a log scale) and log sigma_n^2, by Adam on the analytic gradient
    /// d log p / d theta = 1/2 tr((a a^T - K^-1) dK/dtheta), a = K^-1 y. With W = a a^T - K^-1 and G the
    /// task-by-task sums of W's blocks against K_x, that gradient is (G L) for L and 1/2 sigma_n^2 tr(W)
    /// for the log noise. The best point seen is kept. Which of the two is learned follows the
    /// constructor: a fixed noise is not moved, and with task correlations off B stays the identity.
    /// </para>
    /// <para>
    /// Fixing the noise used to be the only option, at 1e-4 by default. On data whose noise variance is
    /// 2.5e-3 that made the predictive uncertainty far too narrow: 3 of 10 points inside their 95% band.
    /// </para>
    /// </remarks>
    private void OptimizeHyperparameters()
    {
        int n = _X.Rows;
        int tasks = _numTasks;
        int size = n * tasks;
        const double Jitter = 1e-6;

        var inputKernel = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inputKernel[i, j] = _numOps.ToDouble(_kernel.Calculate(_X.GetRow(i), _X.GetRow(j)));
            }
        }

        var targets = new double[size];
        for (int t = 0; t < tasks; t++)
        {
            for (int i = 0; i < n; i++)
            {
                targets[t * n + i] = _numOps.ToDouble(_Y[i, t]) - _numOps.ToDouble(_taskMeans[t]);
            }
        }

        // Starting point: the current B (empirical, or the identity) and, when the noise is learned, a
        // tenth of the average task variance.
        var taskFactor = new double[tasks, tasks];
        var startingCovariance = new double[tasks, tasks];
        double averageTaskVariance = 0;
        for (int a = 0; a < tasks; a++)
        {
            for (int b = 0; b < tasks; b++)
            {
                startingCovariance[a, b] = _numOps.ToDouble(_taskCovariance[a, b]);
            }

            averageTaskVariance += startingCovariance[a, a] / tasks;
        }

        var startingFactor = GaussianProcessEvidence.Cholesky(startingCovariance);
        for (int a = 0; a < tasks; a++)
        {
            for (int b = 0; b <= a; b++)
            {
                taskFactor[a, b] = startingFactor is not null ? startingFactor[a, b] : (a == b ? 1.0 : 0.0);
            }
        }

        bool learnNoise = !_noiseVariance.HasValue;
        bool learnTasks = _learnTaskCorrelations;
        double fixedNoise = _noiseVariance ?? 0.0;
        double logNoise = learnNoise ? Math.Log(Math.Max(0.1 * averageTaskVariance, 1e-8)) : 0.0;

        // Parameter vector: the lower triangle of L (diagonal as a log) when B is learned, then log noise.
        var slots = new List<(int Row, int Column)>();
        if (learnTasks)
        {
            for (int a = 0; a < tasks; a++)
            {
                for (int b = 0; b <= a; b++)
                {
                    slots.Add((a, b));
                }
            }
        }

        int count = slots.Count + (learnNoise ? 1 : 0);
        var theta = new double[count];
        for (int s = 0; s < slots.Count; s++)
        {
            var (row, column) = slots[s];
            theta[s] = row == column ? Math.Log(Math.Max(taskFactor[row, row], 1e-8)) : taskFactor[row, column];
        }

        if (learnNoise)
        {
            theta[count - 1] = logNoise;
        }

        double[,] FactorOf(double[] parameters)
        {
            if (!learnTasks)
                return taskFactor;

            var factor = new double[tasks, tasks];
            for (int s = 0; s < slots.Count; s++)
            {
                var (row, column) = slots[s];
                factor[row, column] = row == column ? Math.Exp(parameters[s]) : parameters[s];
            }

            return factor;
        }

        double NoiseOf(double[] parameters) => learnNoise ? Math.Exp(parameters[count - 1]) : fixedNoise;

        double[,]? CombinedFactor(double[,] factor, double noise, out double[,] covariance)
        {
            covariance = new double[tasks, tasks];
            for (int a = 0; a < tasks; a++)
            {
                for (int b = 0; b < tasks; b++)
                {
                    double sum = 0;
                    for (int k = 0; k < tasks; k++)
                    {
                        sum += factor[a, k] * factor[b, k];
                    }

                    covariance[a, b] = sum;
                }
            }

            var combined = new double[size, size];
            for (int a = 0; a < tasks; a++)
            {
                for (int b = 0; b < tasks; b++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            combined[a * n + i, b * n + j] = covariance[a, b] * inputKernel[i, j];
                        }
                    }
                }
            }

            for (int i = 0; i < size; i++)
            {
                combined[i, i] += noise + Jitter;
            }

            return GaussianProcessEvidence.Cholesky(combined);
        }

        var best = (double[])theta.Clone();
        double bestEvidence = double.NegativeInfinity;
        var firstMoment = new double[count];
        var secondMoment = new double[count];
        const double LearningRate = 0.05;
        const double Beta1 = 0.9;
        const double Beta2 = 0.999;
        const double Epsilon = 1e-8;
        var columns = new[] { targets };

        for (int step = 0; step <= _hyperparameterOptimizationSteps && count > 0; step++)
        {
            var factor = FactorOf(theta);
            double noise = NoiseOf(theta);
            var lower = CombinedFactor(factor, noise, out _);
            if (lower is null)
            {
                break;
            }

            double evidence = GaussianProcessEvidence.LogMarginalLikelihood(lower, columns);
            if (evidence > bestEvidence)
            {
                bestEvidence = evidence;
                best = (double[])theta.Clone();
            }

            if (step == _hyperparameterOptimizationSteps)
            {
                break;
            }

            // W = a a^T - K^-1.
            var alpha = GaussianProcessEvidence.Solve(lower, targets);
            var inverse = GaussianProcessEvidence.Inverse(lower);
            var gradient = new double[count];

            if (learnTasks)
            {
                // G[a, b] = sum_ij W[(a,i),(b,j)] K_x[j,i]; the gradient for L is (G L).
                var g = new double[tasks, tasks];
                for (int a = 0; a < tasks; a++)
                {
                    for (int b = 0; b < tasks; b++)
                    {
                        double sum = 0;
                        for (int i = 0; i < n; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                int p = a * n + i;
                                int q = b * n + j;
                                sum += (alpha[p] * alpha[q] - inverse[p, q]) * inputKernel[j, i];
                            }
                        }

                        g[a, b] = sum;
                    }
                }

                for (int s = 0; s < slots.Count; s++)
                {
                    var (row, column) = slots[s];
                    double sum = 0;
                    for (int k = 0; k < tasks; k++)
                    {
                        sum += g[row, k] * factor[k, column];
                    }

                    // The diagonal is parameterized by its log, so the chain rule multiplies by it.
                    gradient[s] = row == column ? sum * factor[row, row] : sum;
                }
            }

            if (learnNoise)
            {
                double trace = 0;
                for (int i = 0; i < size; i++)
                {
                    trace += alpha[i] * alpha[i] - inverse[i, i];
                }

                gradient[count - 1] = 0.5 * noise * trace;
            }

            // Adam, ascending the evidence.
            for (int k = 0; k < count; k++)
            {
                firstMoment[k] = Beta1 * firstMoment[k] + (1 - Beta1) * gradient[k];
                secondMoment[k] = Beta2 * secondMoment[k] + (1 - Beta2) * gradient[k] * gradient[k];
                double firstHat = firstMoment[k] / (1 - Math.Pow(Beta1, step + 1));
                double secondHat = secondMoment[k] / (1 - Math.Pow(Beta2, step + 1));
                theta[k] += LearningRate * firstHat / (Math.Sqrt(secondHat) + Epsilon);
            }
        }

        var bestFactor = FactorOf(best);
        double bestNoise = NoiseOf(best);
        if (count == 0 || CombinedFactor(bestFactor, bestNoise, out var bestCovariance) is null)
        {
            // Nothing to learn, or no point of the search was positive definite: keep the starting B
            // and the fixed (or starting) noise.
            _fittedNoiseVariance[0] = _numOps.FromDouble(learnNoise ? Math.Exp(logNoise) : fixedNoise);
            return;
        }

        _fittedNoiseVariance[0] = _numOps.FromDouble(bestNoise);
        if (learnTasks)
        {
            _taskCovariance = new Matrix<T>(tasks, tasks);
            _taskCovCholesky = new Matrix<T>(tasks, tasks);
            for (int a = 0; a < tasks; a++)
            {
                for (int b = 0; b < tasks; b++)
                {
                    _taskCovariance[a, b] = _numOps.FromDouble(bestCovariance[a, b]);
                    _taskCovCholesky[a, b] = _numOps.FromDouble(bestFactor[a, b]);
                }
            }
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

        // Add the fitted observation noise to the diagonal
        T noise = _fittedNoiseVariance[0];
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
                // Deviations from the task mean - what the zero-mean prior describes.
                yFlat[t * n + i] = _numOps.Subtract(_Y[i, t], _taskMeans[t]);
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
    public (Vector<T> means, Vector<T> variances) PredictMultiTask(Vector<T> x)
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
            // The prior explains deviations from the task mean; add the mean back.
            means[t] = _numOps.Add(mean, _taskMeans[t]);

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
    public override void UpdateKernel(IKernelFunction<T> kernel)
    {
        Guard.NotNull(kernel);
        _kernel = kernel;
        if (!_X.IsEmpty && !_Y.IsEmpty)
        {
            // A new kernel moves the marginal-likelihood optimum, so re-learn from the current estimates.
            OptimizeHyperparameters();
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

    /// <summary>
    /// IFullModel compliance: Fit with single-output vector (uses first task only).
    /// For multi-task, prefer Fit(Matrix, Matrix) directly.
    /// </summary>
    public override void Fit(Matrix<T> X, Vector<T> y)
    {
        var yMatrix = new Matrix<T>(y.Length, 1);
        for (int i = 0; i < y.Length; i++)
            yMatrix[i, 0] = y[i];
        Fit(X, yMatrix);
    }

    /// <summary>
    /// IFullModel compliance: Predict single point returning first task's mean.
    /// </summary>
    public override (T mean, T variance) Predict(Vector<T> x)
    {
        var (means, variances) = PredictMultiTask(x);
        return (means[0], variances[0]);
    }
}
