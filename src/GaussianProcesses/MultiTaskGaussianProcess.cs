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
        return Engine.ReduceMean(Tensor<T>.FromMatrix(Y), new[] { 0 }, keepDims: false).ToVector();
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
        // cov = Cᵀ C / (n - 1) with C the task-centered targets
        var engine = Engine;
        var centered = CenteredTargets();
        var covariance = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), _numOps.FromDouble(n - 1));

        // Add jitter for stability
        _taskCovariance = GaussianProcessEngineOps.AddToDiagonal(covariance.ToMatrix(), _numOps.FromDouble(1e-6));

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
        var engine = Engine;
        int n = _X.Rows;
        int tasks = _numTasks;
        int size = n * tasks;
        T jitter = _numOps.FromDouble(1e-6);

        bool learnNoise = !_noiseVariance.HasValue;
        bool learnTasks = _learnTaskCorrelations;
        T fixedNoise = _numOps.FromDouble(_noiseVariance ?? 0.0);
        int slots = learnTasks ? tasks * (tasks + 1) / 2 : 0;
        int count = slots + (learnNoise ? 1 : 0);
        if (count == 0)
        {
            // Nothing to learn: keep the starting B and the fixed noise.
            _fittedNoiseVariance[0] = fixedNoise;
            return;
        }

        var inputKernel = Tensor<T>.FromMatrix(InputKernelMatrix());

        // Targets stacked task by task: y[t*n + i] = Y[i, t] - mean_t
        var targets = engine.TensorTranspose(CenteredTargets()).Reshape(new[] { size, 1 });

        // Starting point: the current B (empirical, or the identity) and, when the noise is learned, a
        // tenth of the average task variance.
        var startingCovariance = Tensor<T>.FromMatrix(_taskCovariance);
        T averageTaskVariance = _numOps.Divide(engine.TensorTrace(startingCovariance), _numOps.FromDouble(tasks));
        var startingFactor = GaussianProcessEvidence.Cholesky(startingCovariance);
        var taskFactor = startingFactor is not null ? engine.TensorTril(startingFactor) : engine.TensorEye<T>(tasks);

        T floor = _numOps.FromDouble(1e-8);
        T logNoise = learnNoise
            ? _numOps.Log(MathHelper.Max(_numOps.Multiply(_numOps.FromDouble(0.1), averageTaskVariance), floor))
            : _numOps.Zero;

        // Parameter vector: the lower triangle of L row by row (diagonal as a log) when B is learned, then log noise.
        var positions = new int[slots];
        int slot = 0;
        for (int a = 0; a < tasks && learnTasks; a++)
        {
            for (int b = 0; b <= a; b++)
            {
                positions[slot++] = a * tasks + b;
            }
        }

        var slotIndex = new Tensor<int>(positions, new[] { slots });
        var diagonalMask = slots > 0
            ? engine.TensorIndexSelect(engine.TensorEye<T>(tasks).Reshape(new[] { tasks * tasks }), slotIndex, 0)
            : new Tensor<T>(new[] { 0 });

        var initialParts = new List<Tensor<T>>();
        if (slots > 0)
        {
            var factorValues = engine.TensorIndexSelect(taskFactor.Reshape(new[] { tasks * tasks }), slotIndex, 0);
            initialParts.Add(engine.TensorWhere(
                diagonalMask, engine.TensorLog(engine.TensorMax(factorValues, floor)), factorValues));
        }

        if (learnNoise)
        {
            initialParts.Add(Tensor<T>.CreateDefault(new[] { 1 }, logNoise));
        }

        var theta = engine.TensorConcatenate(initialParts.ToArray(), 0);

        Tensor<T> FactorOf(Tensor<T> parameters)
        {
            if (!learnTasks)
                return taskFactor;

            var slotParameters = engine.TensorSlice(parameters, new[] { 0 }, new[] { slots });
            var values = engine.TensorWhere(diagonalMask, engine.TensorExp(slotParameters), slotParameters);
            return engine.TensorIndexAdd(new Tensor<T>(new[] { tasks * tasks }), 0, slotIndex, values)
                .Reshape(new[] { tasks, tasks });
        }

        T NoiseOf(Tensor<T> parameters) => learnNoise
            ? _numOps.Exp(engine.TensorSum(engine.TensorSlice(parameters, new[] { count - 1 }, new[] { 1 })))
            : fixedNoise;

        // K = B (x) K_x + (sigma_n^2 + jitter) I, with K[(a,i),(b,j)] = B[a,b] K_x[i,j]
        var identity = engine.TensorEye<T>(size);
        var blockShape = new[] { tasks, n, tasks, n };
        var inputBlocks = engine.TensorBroadcastTo(inputKernel.Reshape(new[] { 1, n, 1, n }), blockShape);
        Tensor<T>? CombinedFactor(Tensor<T> factor, T noise, out Tensor<T> covariance)
        {
            covariance = engine.TensorMatMul(factor, engine.TensorTranspose(factor));
            var taskBlocks = engine.TensorBroadcastTo(covariance.Reshape(new[] { tasks, 1, tasks, 1 }), blockShape);
            var combined = engine.TensorMultiply(taskBlocks, inputBlocks).Reshape(new[] { size, size });
            combined = engine.TensorAdd(combined, engine.TensorMultiplyScalar(identity, _numOps.Add(noise, jitter)));
            return GaussianProcessEvidence.Cholesky(combined);
        }

        // Block (a, b) of the tiled matrix is K_x^T, so summing W * tile over a block gives
        // sum_ij W[(a,i),(b,j)] K_x[j,i].
        var kernelTiles = engine.TensorTile(engine.TensorTranspose(inputKernel), new[] { tasks, tasks });
        var best = theta;
        T bestEvidence = _numOps.FromDouble(double.NegativeInfinity);
        var firstMoment = engine.TensorZerosLike(theta);
        var secondMoment = engine.TensorZerosLike(theta);
        const double LearningRate = 0.05;
        const double Beta1 = 0.9;
        const double Beta2 = 0.999;
        const double Epsilon = 1e-8;
        T half = _numOps.FromDouble(0.5);

        for (int step = 0; step <= _hyperparameterOptimizationSteps; step++)
        {
            var factor = FactorOf(theta);
            T noise = NoiseOf(theta);
            var lower = CombinedFactor(factor, noise, out _);
            if (lower is null)
            {
                break;
            }

            T evidence = GaussianProcessEvidence.LogMarginalLikelihood(lower, targets);
            if (_numOps.GreaterThan(evidence, bestEvidence))
            {
                bestEvidence = evidence;
                best = theta;
            }

            if (step == _hyperparameterOptimizationSteps)
            {
                break;
            }

            // W = a a^T - K^-1.
            var alpha = GaussianProcessEvidence.Solve(lower, targets);
            var inverse = GaussianProcessEvidence.Inverse(lower);
            var w = engine.TensorSubtract(engine.TensorMatMul(alpha, engine.TensorTranspose(alpha)), inverse);
            var gradientParts = new List<Tensor<T>>();

            if (learnTasks)
            {
                // G[a, b] = sum_ij W[(a,i),(b,j)] K_x[j,i]; the gradient for L is (G L).
                var g = engine.ReduceSum(
                    engine.TensorMultiply(w, kernelTiles).Reshape(blockShape), new[] { 1, 3 }, keepDims: false);
                var gl = engine.TensorIndexSelect(
                    engine.TensorMatMul(g, factor).Reshape(new[] { tasks * tasks }), slotIndex, 0);

                // The diagonal is parameterized by its log, so the chain rule multiplies by it.
                var factorSlots = engine.TensorIndexSelect(factor.Reshape(new[] { tasks * tasks }), slotIndex, 0);
                gradientParts.Add(engine.TensorWhere(diagonalMask, engine.TensorMultiply(gl, factorSlots), gl));
            }

            if (learnNoise)
            {
                T trace = _numOps.Subtract(engine.TensorSumOfSquares(alpha), engine.TensorTrace(inverse));
                gradientParts.Add(Tensor<T>.CreateDefault(
                    new[] { 1 }, _numOps.Multiply(half, _numOps.Multiply(noise, trace))));
            }

            var gradient = engine.TensorConcatenate(gradientParts.ToArray(), 0);

            // Adam, ascending the evidence.
            firstMoment = engine.TensorAdd(
                engine.TensorMultiplyScalar(firstMoment, _numOps.FromDouble(Beta1)),
                engine.TensorMultiplyScalar(gradient, _numOps.FromDouble(1 - Beta1)));
            secondMoment = engine.TensorAdd(
                engine.TensorMultiplyScalar(secondMoment, _numOps.FromDouble(Beta2)),
                engine.TensorMultiplyScalar(engine.TensorSquare(gradient), _numOps.FromDouble(1 - Beta2)));
            var firstHat = engine.TensorDivideScalar(firstMoment, _numOps.FromDouble(1 - Math.Pow(Beta1, step + 1)));
            var secondHat = engine.TensorDivideScalar(secondMoment, _numOps.FromDouble(1 - Math.Pow(Beta2, step + 1)));
            var stepSize = engine.TensorDivide(
                engine.TensorMultiplyScalar(firstHat, _numOps.FromDouble(LearningRate)),
                engine.TensorAddScalar(engine.TensorSqrt(secondHat), _numOps.FromDouble(Epsilon)));
            theta = engine.TensorAdd(theta, stepSize);
        }

        var bestFactor = FactorOf(best);
        T bestNoise = NoiseOf(best);
        if (CombinedFactor(bestFactor, bestNoise, out var bestCovariance) is null)
        {
            // No point of the search was positive definite: keep the starting B and the starting noise.
            _fittedNoiseVariance[0] = learnNoise ? _numOps.Exp(logNoise) : fixedNoise;
            return;
        }

        _fittedNoiseVariance[0] = bestNoise;
        if (learnTasks)
        {
            _taskCovariance = bestCovariance.ToMatrix();
            _taskCovCholesky = bestFactor.ToMatrix();
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
        var Kx = InputKernelMatrix();

        // Build combined kernel using Kronecker structure: K[t1·n + i, t2·n + j] = B[t1, t2] · Kx[i, j]
        var engine = Engine;
        var blockShape = new[] { _numTasks, n, _numTasks, n };
        var taskBlocks = engine.TensorBroadcastTo(
            Tensor<T>.FromMatrix(_taskCovariance).Reshape(new[] { _numTasks, 1, _numTasks, 1 }), blockShape);
        var inputBlocks = engine.TensorBroadcastTo(
            Tensor<T>.FromMatrix(Kx).Reshape(new[] { 1, n, 1, n }), blockShape);
        var combined = engine.TensorMultiply(taskBlocks, inputBlocks).Reshape(new[] { totalSize, totalSize }).ToMatrix();

        // Add the fitted observation noise to the diagonal, then jitter
        combined = GaussianProcessEngineOps.AddToDiagonal(combined, _fittedNoiseVariance[0]);
        _K = GaussianProcessEngineOps.AddToDiagonal(combined, _numOps.FromDouble(1e-6));
    }

    /// <summary>
    /// Computes the alpha vector for predictions.
    /// </summary>
    /// <summary>
    /// Targets minus their task means, [n, tasks].
    /// </summary>
    private Tensor<T> CenteredTargets()
    {
        var engine = Engine;
        var targets = Tensor<T>.FromMatrix(_Y);
        var means = Tensor<T>.FromVector(_taskMeans).Reshape(new[] { 1, _numTasks });
        return engine.TensorSubtract(targets, engine.TensorBroadcastTo(means, new[] { _Y.Rows, _numTasks }));
    }

    /// <summary>
    /// Input kernel matrix Kx[i, j] = k(x_i, x_j) over the training inputs.
    /// </summary>
    private Matrix<T> InputKernelMatrix()
    {
        int n = _X.Rows;
        var Kx = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Kx[i, j] = _kernel.Calculate(_X.GetRow(i), _X.GetRow(j));
            }
        }

        return Kx;
    }

    /// <summary>
    /// Cross kernel vector k(x, x_i) over the training inputs.
    /// </summary>
    private Vector<T> CrossKernelVector(Vector<T> x)
    {
        int n = _X.Rows;
        var kStar = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            kStar[i] = _kernel.Calculate(x, _X.GetRow(i));
        }

        return kStar;
    }

    private void ComputeAlpha()
    {
        int n = _X.Rows;

        // Flatten Y into a vector (task-major order, index t·n + i).
        // Deviations from the task mean - what the zero-mean prior describes.
        var yFlat = Engine.TensorTranspose(CenteredTargets()).Reshape(new[] { n * _numTasks }).ToVector();

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
        var kStar = CrossKernelVector(x);
        var engine = Engine;
        var taskCovariance = Tensor<T>.FromMatrix(_taskCovariance);
        T kStarStar = _kernel.Calculate(x, x);

        // Mean_t = Σ_t2 B[t, t2] Σ_i k*_i α[t2·n + i] = (B · (A k*))_t with A = α reshaped to [tasks, n].
        // The prior explains deviations from the task mean; add the mean back.
        var alphaByTask = Tensor<T>.FromVector(_alpha).Reshape(new[] { _numTasks, n });
        var kStarColumn = Tensor<T>.FromVector(kStar).Reshape(new[] { n, 1 });
        var taskProjections = engine.TensorMatMul(alphaByTask, kStarColumn);
        var means = engine.TensorAdd(
            engine.TensorMatMul(taskCovariance, taskProjections).Reshape(new[] { _numTasks }),
            Tensor<T>.FromVector(_taskMeans)).ToVector();

        // Variance (simplified - full computation would use Kronecker structure):
        // B[t, t] k** - k*_tᵀ K⁻¹ k*_t with k*_t[t2·n + i] = B[t, t2] k*_i (column t of kStarTasks).
        int totalSize = n * _numTasks;
        var columnShape = new[] { _numTasks, n, _numTasks };
        var kStarTasks = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.TensorTranspose(taskCovariance).Reshape(new[] { _numTasks, 1, _numTasks }), columnShape),
            engine.TensorBroadcastTo(kStarColumn.Reshape(new[] { 1, n, 1 }), columnShape)).Reshape(new[] { totalSize, _numTasks });

        var kStarTaskMatrix = kStarTasks.ToMatrix();
        var solved = new Tensor<T>[_numTasks];
        for (int t = 0; t < _numTasks; t++)
        {
            solved[t] = Tensor<T>.FromVector(
                MatrixSolutionHelper.SolveLinearSystem(_K, kStarTaskMatrix.GetColumn(t), _decompositionType));
        }

        var reduction = engine.ReduceSum(
            engine.TensorMultiply(kStarTasks, engine.TensorStack(solved, 1)), new[] { 0 }, keepDims: false);
        var priorVariance = engine.TensorMultiplyScalar(engine.TensorDiagonal(taskCovariance), kStarStar);
        var variances = engine.TensorMax(engine.TensorSubtract(priorVariance, reduction), _numOps.FromDouble(1e-10)).ToVector();

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
        return Engine.TensorEye<T>(size).ToMatrix();
    }

    /// <summary>
    /// IFullModel compliance: Fit with single-output vector (uses first task only).
    /// For multi-task, prefer Fit(Matrix, Matrix) directly.
    /// </summary>
    public override void Fit(Matrix<T> X, Vector<T> y)
    {
        var yMatrix = Tensor<T>.FromVector(y).Reshape(new[] { y.Length, 1 }).ToMatrix();
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
