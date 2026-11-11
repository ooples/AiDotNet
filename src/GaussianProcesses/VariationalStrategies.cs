using AiDotNet.Helpers;

namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Provides variational inference strategies for scalable Gaussian Process inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard GP inference scales as O(n³), which becomes impractical for
/// large datasets. Variational inference provides approximations that scale better:
///
/// Key ideas:
/// 1. Introduce "inducing points" that summarize the data
/// 2. Approximate the true posterior with a simpler variational distribution
/// 3. Optimize the variational parameters to minimize KL divergence
///
/// Common strategies:
/// - SVGP (Sparse Variational GP): Full variational approximation
/// - FITC: Fully Independent Training Conditional
/// - VFE: Variational Free Energy
/// - KISS-GP: Kernel Interpolation for Scalable Structured GPs
///
/// Trade-offs:
/// - More inducing points = better approximation but slower
/// - Fewer inducing points = faster but less accurate
/// </para>
/// </remarks>
public class VariationalStrategies<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new variational strategies helper.
    /// </summary>
    public VariationalStrategies()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Selects inducing points using k-means clustering on training data.
    /// </summary>
    /// <param name="X">Training data matrix (N x D).</param>
    /// <param name="numInducingPoints">Number of inducing points (M).</param>
    /// <param name="maxIterations">Maximum k-means iterations. Default is 100.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Matrix of inducing points (M x D).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inducing points should be representative of your data.
    /// K-means clustering finds M cluster centers that cover the input space well.
    ///
    /// Why k-means works well:
    /// - Centers are spread across the data
    /// - Dense regions get more inducing points
    /// - Empty regions are avoided
    ///
    /// Tips:
    /// - M = sqrt(N) is a reasonable starting point
    /// - Increase M if you need more accuracy
    /// - Decrease M for faster inference
    /// </para>
    /// </remarks>
    public Matrix<T> SelectInducingPointsKMeans(
        Matrix<T> X,
        int numInducingPoints,
        int maxIterations = 100,
        int seed = 42)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (numInducingPoints < 1)
            throw new ArgumentException("Number of inducing points must be at least 1.", nameof(numInducingPoints));
        if (numInducingPoints > X.Rows)
            throw new ArgumentException("Number of inducing points cannot exceed data size.", nameof(numInducingPoints));

        int n = X.Rows;
        int d = X.Columns;
        int m = numInducingPoints;

        var rand = RandomHelper.CreateSeededRandom(seed);

        var engine = AiDotNetEngine.Current;
        var data = Tensor<T>.FromMatrix(X);

        // Initialize centers using k-means++
        var centers = new Matrix<T>(m, d);
        var usedIndices = new HashSet<int>();

        // First center: random point
        int firstIdx = rand.Next(n);
        usedIndices.Add(firstIdx);
        centers.SetRow(0, X.GetRow(firstIdx));

        // Remaining centers: k-means++ initialization
        var minDistances = Tensor<T>.CreateDefault(new[] { n }, _numOps.FromDouble(double.MaxValue));
        T tinyTotal = _numOps.FromDouble(1e-10);

        for (int k = 1; k < m; k++)
        {
            // Update distances to nearest center
            minDistances = engine.TensorMin(minDistances, SquaredDistancesToCenters(data, centers.GetRow(k - 1)));

            // Sample next center proportional to squared distance
            T totalDist = engine.TensorSum(minDistances);
            int selectedIdx;
            if (_numOps.LessThan(totalDist, tinyTotal))
            {
                // All points are very close to existing centers
                // Just pick a random unused point
                do
                {
                    selectedIdx = rand.Next(n);
                } while (usedIndices.Contains(selectedIdx));
            }
            else
            {
                // First index whose running distance total reaches threshold (index 0 if none does)
                var threshold = Tensor<T>.FromScalar(_numOps.Multiply(_numOps.FromDouble(rand.NextDouble()), totalDist))
                    .Reshape(new[] { 1 });
                var cumulative = engine.TensorCumSum(minDistances, 0);
                selectedIdx = engine.TensorSearchSorted(cumulative, threshold).AsSpan()[0];
                if (selectedIdx >= n) selectedIdx = 0;
            }

            usedIndices.Add(selectedIdx);
            centers.SetRow(k, X.GetRow(selectedIdx));
        }

        // Run k-means iterations
        var assignments = new int[n];
        var ones = Tensor<T>.CreateDefault(new[] { n }, _numOps.One);
        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assignment step: nearest center (first one on ties) for every point
            var distances = SquaredDistancesToCenters(data, centers);
            var nearest = engine.TensorArgMin(distances, 1);
            var nearestCenters = nearest.AsSpan().ToArray();
            bool changed = !nearestCenters.AsSpan().SequenceEqual(assignments);
            assignments = nearestCenters;

            if (!changed)
                break;

            // Update step: mean of the points assigned to each center (empty clusters keep their center)
            var sums = engine.TensorIndexAdd(new Tensor<T>(new[] { m, d }), 0, nearest, data);
            var counts = engine.TensorIndexAdd(new Tensor<T>(new[] { m }), 0, nearest, ones);
            var hasPoints = engine.TensorGreaterThan(counts, _numOps.Zero);
            var safeCounts = engine.TensorWhere(hasPoints, counts, engine.TensorOnesLike(counts));
            var means = engine.TensorDivide(
                sums, engine.TensorBroadcastTo(safeCounts.Reshape(new[] { m, 1 }), new[] { m, d }));
            var keep = engine.TensorBroadcastTo(hasPoints.Reshape(new[] { m, 1 }), new[] { m, d });
            centers = engine.TensorWhere(keep, means, Tensor<T>.FromMatrix(centers)).ToMatrix();
        }

        return centers;
    }

    /// <summary>
    /// Computes squared distance between a data point and a center.
    /// </summary>
    private static Tensor<T> SquaredDistancesToCenters(Tensor<T> data, Vector<T> center)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Shape[0];
        int d = data.Shape[1];
        var diff = engine.TensorSubtract(
            data, engine.TensorBroadcastTo(Tensor<T>.FromVector(center).Reshape(new[] { 1, d }), new[] { n, d }));
        return engine.ReduceSum(engine.TensorSquare(diff), new[] { 1 }, keepDims: false);
    }

    /// <summary>
    /// Squared Euclidean distance from every data row to every center, [n, m].
    /// </summary>
    private static Tensor<T> SquaredDistancesToCenters(Tensor<T> data, Matrix<T> centers)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Shape[0];
        int d = data.Shape[1];
        int m = centers.Rows;
        var shape = new[] { n, m, d };
        var diff = engine.TensorSubtract(
            engine.TensorBroadcastTo(data.Reshape(new[] { n, 1, d }), shape),
            engine.TensorBroadcastTo(Tensor<T>.FromMatrix(centers).Reshape(new[] { 1, m, d }), shape));
        return engine.ReduceSum(engine.TensorSquare(diff), new[] { 2 }, keepDims: false);
    }

    /// <summary>
    /// Selects inducing points using greedy variance reduction.
    /// </summary>
    /// <param name="X">Training data matrix (N x D).</param>
    /// <param name="kernel">The kernel function.</param>
    /// <param name="numInducingPoints">Number of inducing points (M).</param>
    /// <returns>Matrix of inducing points (M x D).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method greedily selects inducing points to maximize
    /// the reduction in posterior variance.
    ///
    /// Algorithm:
    /// 1. Start with no inducing points
    /// 2. Repeatedly add the point that most reduces total variance
    /// 3. Stop when we have M inducing points
    ///
    /// This tends to place inducing points where the GP is most uncertain,
    /// which is often more effective than uniform placement.
    /// </para>
    /// </remarks>
    public Matrix<T> SelectInducingPointsGreedyVariance(
        Matrix<T> X,
        IKernelFunction<T> kernel,
        int numInducingPoints)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (numInducingPoints < 1)
            throw new ArgumentException("Number of inducing points must be at least 1.", nameof(numInducingPoints));

        int n = X.Rows;
        int d = X.Columns;
        int m = Math.Min(numInducingPoints, n);

        var selectedIndices = new List<int>();
        var remainingIndices = new HashSet<int>(Enumerable.Range(0, n));

        // Compute diagonal of kernel matrix (prior variances)
        var priorVariances = new double[n];
        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(X, i);
            priorVariances[i] = _numOps.ToDouble(kernel.Calculate(xi, xi));
        }

        // Greedy selection
        for (int k = 0; k < m; k++)
        {
            // Find point with maximum remaining variance
            int bestIdx = -1;
            double maxVariance = -1;

            foreach (int i in remainingIndices)
            {
                double variance = priorVariances[i];

                // Subtract variance explained by current inducing points
                if (selectedIndices.Count > 0)
                {
                    var xi = GetRow(X, i);
                    var Kiu = new Vector<T>(selectedIndices.Count);
                    for (int j = 0; j < selectedIndices.Count; j++)
                    {
                        var xj = GetRow(X, selectedIndices[j]);
                        Kiu[j] = kernel.Calculate(xi, xj);
                    }

                    // Approximate variance reduction
                    double reduction = 0;
                    for (int j = 0; j < Kiu.Length; j++)
                    {
                        double kij = _numOps.ToDouble(Kiu[j]);
                        reduction += kij * kij;
                    }
                    variance -= reduction / (selectedIndices.Count + 1e-6);
                }

                if (variance > maxVariance)
                {
                    maxVariance = variance;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0)
            {
                selectedIndices.Add(bestIdx);
                remainingIndices.Remove(bestIdx);
            }
        }

        // Extract selected points
        return X.GetRows(selectedIndices.Take(m));
    }

    /// <summary>
    /// Gets a row from a matrix as a vector.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> M, int row)
    {
        return M.GetRow(row);
    }

    /// <summary>
    /// Computes the ELBO (Evidence Lower BOund) for sparse GP.
    /// </summary>
    /// <param name="X">Training inputs (N x D).</param>
    /// <param name="y">Training targets (N).</param>
    /// <param name="Z">Inducing points (M x D).</param>
    /// <param name="kernel">The kernel function.</param>
    /// <param name="noiseVariance">Observation noise variance.</param>
    /// <param name="variationalMean">Variational mean (M).</param>
    /// <param name="variationalCovChol">Cholesky of variational covariance (M x M).</param>
    /// <returns>The ELBO value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The ELBO is a lower bound on the log marginal likelihood.
    /// Maximizing ELBO is equivalent to minimizing KL divergence between the
    /// variational approximation and the true posterior.
    ///
    /// ELBO = E_q[log p(y|f)] - KL(q(u) || p(u))
    ///
    /// Where:
    /// - First term: How well the model explains the data
    /// - Second term: How far the variational distribution is from the prior
    ///
    /// Higher ELBO = better model. Use this to:
    /// - Compare different inducing point configurations
    /// - Optimize variational parameters
    /// - Select hyperparameters
    /// </para>
    /// </remarks>
    public T ComputeELBO(
        Matrix<T> X,
        Vector<T> y,
        Matrix<T> Z,
        IKernelFunction<T> kernel,
        double noiseVariance,
        Vector<T> variationalMean,
        Matrix<T> variationalCovChol)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (Z is null) throw new ArgumentNullException(nameof(Z));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (variationalMean is null) throw new ArgumentNullException(nameof(variationalMean));
        if (variationalCovChol is null) throw new ArgumentNullException(nameof(variationalCovChol));

        // Validate noise variance to prevent divide-by-zero
        if (noiseVariance <= 0)
            throw new ArgumentOutOfRangeException(nameof(noiseVariance),
                "Noise variance must be positive to prevent divide-by-zero in ELBO computation.");

        int n = X.Rows;
        int m = Z.Rows;

        // Validate dimension consistency
        if (y.Length != n)
            throw new ArgumentException($"y length ({y.Length}) must match X rows ({n}).", nameof(y));
        if (variationalMean.Length != m)
            throw new ArgumentException($"variationalMean length ({variationalMean.Length}) must match Z rows ({m}).", nameof(variationalMean));
        if (variationalCovChol.Rows != m || variationalCovChol.Columns != m)
            throw new ArgumentException($"variationalCovChol must be {m}x{m} to match Z rows.", nameof(variationalCovChol));
        if (X.Columns != Z.Columns)
            throw new ArgumentException($"X columns ({X.Columns}) must match Z columns ({Z.Columns}).", nameof(Z));

        // Compute kernel matrices
        var Kuu = ComputeKernelMatrix(Z, Z, kernel);
        var Kuf = ComputeKernelMatrix(Z, X, kernel);

        var engine = AiDotNetEngine.Current;
        T negativeInfinity = _numOps.FromDouble(double.NegativeInfinity);

        // Add jitter to Kuu
        Kuu = GaussianProcessEngineOps.AddToDiagonal(Kuu, _numOps.FromDouble(1e-6));

        // Cholesky decomposition of Kuu; a factor that does not exist or has a pivot below 1e-10
        // makes the triangular solves below undefined.
        var (luu, info) = Linalg.CholeskyEx(Tensor<T>.FromMatrix(Kuu), upper: false);
        if (engine.TensorCountNonzero(info) > 0 || !GaussianProcessEngineOps.AllFinite(luu))
        {
            return negativeInfinity;
        }

        var luuDiagonal = engine.TensorAbs(engine.TensorDiagonal(luu));
        if (m > 0 && _numOps.LessThan(engine.TensorMinValue(luuDiagonal), _numOps.FromDouble(1e-10)))
        {
            return negativeInfinity;
        }

        // alpha = Luu \ Kuf (whitened cross-covariance), Kuu^{-1} m and A = Kuu^{-1} Kuf
        var kuf = Tensor<T>.FromMatrix(Kuf);
        var alpha = Linalg.SolveTriangular(luu, kuf, upper: false);
        var mean = Tensor<T>.FromVector(variationalMean).Reshape(new[] { m, 1 });
        var KuuInvM = Linalg.CholeskySolve(mean, luu, upper: false);
        var KuuInvKuf = Linalg.CholeskySolve(kuf, luu, upper: false);

        // Expected log likelihood term (Hensman et al. 2013):
        // Σ_i log N(y_i | k_iᵀ Kuu⁻¹ m, σ²) - β/2 Σ_i (k_ii - q_ii) - β/2 Σ_i k_iᵀ Kuu⁻¹ S Kuu⁻¹ k_i
        T halfBeta = _numOps.FromDouble(0.5 / noiseVariance);

        // Predictive mean: Kfu @ Kuu^{-1} @ m
        var predictiveMean = engine.TensorMatMul(engine.TensorTranspose(kuf), KuuInvM).Reshape(new[] { n });
        var residuals = engine.TensorSubtract(Tensor<T>.FromVector(y), predictiveMean);
        T logLikeTerm = _numOps.Subtract(
            _numOps.FromDouble(n * -0.5 * Math.Log(2 * Math.PI * noiseVariance)),
            _numOps.Multiply(halfBeta, engine.TensorSumOfSquares(residuals)));

        // Trace term for variance correction: Σ_i (k_ii - Σ_j alpha[j,i]²)
        var kernelDiagonal = Tensor<T>.FromVector(KernelDiagonal(X, kernel));
        T traceTerm = _numOps.Subtract(engine.TensorSum(kernelDiagonal), engine.TensorSumOfSquares(alpha));
        logLikeTerm = _numOps.Subtract(logLikeTerm, _numOps.Multiply(halfBeta, traceTerm));

        // Variance of q(u) carried to f: Σ_i a_iᵀ S a_i = ||L_Sᵀ A||_F² with A = Kuu⁻¹ Kuf
        var covChol = Tensor<T>.FromMatrix(variationalCovChol);
        var projected = engine.TensorMatMul(engine.TensorTranspose(covChol), KuuInvKuf);
        logLikeTerm = _numOps.Subtract(logLikeTerm, _numOps.Multiply(halfBeta, engine.TensorSumOfSquares(projected)));

        // KL divergence term: KL(q(u) || p(u))
        // = 0.5 * (tr(Kuu^{-1} S) + m^T Kuu^{-1} m - M + log|Kuu| - log|S|)
        T floor = _numOps.FromDouble(1e-10);
        T two = _numOps.FromDouble(2.0);
        T logDetKuu = _numOps.Multiply(two, engine.TensorSum(engine.TensorLog(engine.TensorMax(luuDiagonal, floor))));
        T logDetS = _numOps.Multiply(two, engine.TensorSum(engine.TensorLog(
            engine.TensorMax(engine.TensorAbs(engine.TensorDiagonal(covChol)), floor))));
        T mKuuInvM = engine.TensorVecDot(mean.Reshape(new[] { m }), KuuInvM.Reshape(new[] { m }));

        // tr(Kuu^{-1} S) = ||Luu^{-1} @ S_chol||_F^2 where S = S_chol @ S_chol^T and Kuu = Luu @ Luu^T
        T traceKuuInvS = engine.TensorSumOfSquares(Linalg.SolveTriangular(luu, covChol, upper: false));

        T klTerm = _numOps.Multiply(
            _numOps.FromDouble(0.5),
            _numOps.Add(
                _numOps.Subtract(_numOps.Add(traceKuuInvS, mKuuInvM), _numOps.FromDouble(m)),
                _numOps.Subtract(logDetKuu, logDetS)));

        return _numOps.Subtract(logLikeTerm, klTerm);
    }

    /// <summary>
    /// Computes a kernel matrix between two sets of points.
    /// </summary>
    /// <summary>
    /// Kernel self-similarities k(x_i, x_i) for every row of <paramref name="X"/>.
    /// </summary>
    private Vector<T> KernelDiagonal(Matrix<T> X, IKernelFunction<T> kernel)
    {
        var diagonal = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            var xi = GetRow(X, i);
            diagonal[i] = kernel.Calculate(xi, xi);
        }

        return diagonal;
    }

    private Matrix<T> ComputeKernelMatrix(Matrix<T> A, Matrix<T> B, IKernelFunction<T> kernel)
    {
        int n = A.Rows;
        int m = B.Rows;
        var K = new Matrix<T>(n, m);

        for (int i = 0; i < n; i++)
        {
            var ai = GetRow(A, i);
            for (int j = 0; j < m; j++)
            {
                var bj = GetRow(B, j);
                K[i, j] = kernel.Calculate(ai, bj);
            }
        }

        return K;
    }

    /// <summary>
    /// Initializes variational parameters for SVGP.
    /// </summary>
    /// <param name="Z">Inducing points (M x D).</param>
    /// <param name="y">Training targets (N).</param>
    /// <param name="Kuf">Cross-covariance matrix (M x N).</param>
    /// <param name="noiseVariance">Observation noise variance.</param>
    /// <returns>Initial variational mean and covariance Cholesky factor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Good initialization of variational parameters speeds up
    /// convergence. This method uses a simple heuristic based on the inducing points
    /// and training data.
    ///
    /// Returns:
    /// - Mean: Weighted average of nearby training targets
    /// - Covariance Cholesky: Identity scaled by prior variance
    /// </para>
    /// </remarks>
    public (Vector<T> mean, Matrix<T> covChol) InitializeVariationalParameters(
        Matrix<T> Z,
        Vector<T> y,
        Matrix<T> Kuf,
        double noiseVariance)
    {
        if (Z is null) throw new ArgumentNullException(nameof(Z));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (Kuf is null) throw new ArgumentNullException(nameof(Kuf));

        int m = Z.Rows;
        int n = y.Length;

        // Initialize mean: weighted average of nearby targets
        // mean_i = Σ_j |Kuf_ij| y_j / Σ_j |Kuf_ij|, or 0 when the weights sum to at most 1e-10
        var engine = AiDotNetEngine.Current;
        var weights = engine.TensorAbs(Tensor<T>.FromMatrix(Kuf));
        var weightSums = engine.ReduceSum(weights, new[] { 1 }, keepDims: false);
        var valueSums = engine.TensorMatMul(weights, Tensor<T>.FromVector(y).Reshape(new[] { n, 1 })).Reshape(new[] { m });
        var hasWeight = engine.TensorGreaterThan(weightSums, _numOps.FromDouble(1e-10));
        var safeSums = engine.TensorWhere(hasWeight, weightSums, engine.TensorOnesLike(weightSums));
        var mean = engine.TensorWhere(
            hasWeight, engine.TensorDivide(valueSums, safeSums), engine.TensorZerosLike(weightSums)).ToVector();

        // Initialize covariance Cholesky: scaled identity
        double scale = Math.Sqrt(Math.Max(noiseVariance, 0.1));
        var covChol = engine.TensorMultiplyScalar(engine.TensorEye<T>(m), _numOps.FromDouble(scale)).ToMatrix();

        return (mean, covChol);
    }
}
