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

        // Initialize centers using k-means++
        var centers = new Matrix<T>(m, d);
        var usedIndices = new HashSet<int>();

        // First center: random point
        int firstIdx = rand.Next(n);
        usedIndices.Add(firstIdx);
        for (int j = 0; j < d; j++)
        {
            centers[0, j] = X[firstIdx, j];
        }

        // Remaining centers: k-means++ initialization
        var minDistances = new double[n];
        for (int i = 0; i < n; i++)
        {
            minDistances[i] = double.MaxValue;
        }

        for (int k = 1; k < m; k++)
        {
            // Update distances to nearest center
            for (int i = 0; i < n; i++)
            {
                double dist = ComputeSquaredDistance(X, i, centers, k - 1);
                minDistances[i] = Math.Min(minDistances[i], dist);
            }

            // Sample next center proportional to squared distance
            double totalDist = minDistances.Sum();
            if (totalDist < 1e-10)
            {
                // All points are very close to existing centers
                // Just pick a random unused point
                int nextIdx;
                do
                {
                    nextIdx = rand.Next(n);
                } while (usedIndices.Contains(nextIdx));

                usedIndices.Add(nextIdx);
                for (int j = 0; j < d; j++)
                {
                    centers[k, j] = X[nextIdx, j];
                }
            }
            else
            {
                double threshold = rand.NextDouble() * totalDist;
                double cumSum = 0;
                int selectedIdx = 0;

                for (int i = 0; i < n; i++)
                {
                    cumSum += minDistances[i];
                    if (cumSum >= threshold)
                    {
                        selectedIdx = i;
                        break;
                    }
                }

                usedIndices.Add(selectedIdx);
                for (int j = 0; j < d; j++)
                {
                    centers[k, j] = X[selectedIdx, j];
                }
            }
        }

        // Run k-means iterations
        var assignments = new int[n];

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Assignment step
            bool changed = false;
            for (int i = 0; i < n; i++)
            {
                int nearestCenter = 0;
                double minDist = double.MaxValue;

                for (int k = 0; k < m; k++)
                {
                    double dist = ComputeSquaredDistance(X, i, centers, k);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestCenter = k;
                    }
                }

                if (assignments[i] != nearestCenter)
                {
                    assignments[i] = nearestCenter;
                    changed = true;
                }
            }

            if (!changed)
                break;

            // Update step: compute new centers
            var newCenters = new Matrix<T>(m, d);
            var counts = new int[m];

            for (int i = 0; i < n; i++)
            {
                int k = assignments[i];
                counts[k]++;
                for (int j = 0; j < d; j++)
                {
                    newCenters[k, j] = _numOps.Add(newCenters[k, j], X[i, j]);
                }
            }

            for (int k = 0; k < m; k++)
            {
                if (counts[k] > 0)
                {
                    for (int j = 0; j < d; j++)
                    {
                        centers[k, j] = _numOps.Divide(newCenters[k, j], _numOps.FromDouble(counts[k]));
                    }
                }
            }
        }

        return centers;
    }

    /// <summary>
    /// Computes squared distance between a data point and a center.
    /// </summary>
    private double ComputeSquaredDistance(Matrix<T> X, int dataIdx, Matrix<T> centers, int centerIdx)
    {
        double dist = 0;
        for (int j = 0; j < X.Columns; j++)
        {
            double diff = _numOps.ToDouble(X[dataIdx, j]) - _numOps.ToDouble(centers[centerIdx, j]);
            dist += diff * diff;
        }
        return dist;
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
        var inducingPoints = new Matrix<T>(m, d);
        for (int k = 0; k < m; k++)
        {
            int idx = selectedIndices[k];
            for (int j = 0; j < d; j++)
            {
                inducingPoints[k, j] = X[idx, j];
            }
        }

        return inducingPoints;
    }

    /// <summary>
    /// Gets a row from a matrix as a vector.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> M, int row)
    {
        var result = new Vector<T>(M.Columns);
        for (int j = 0; j < M.Columns; j++)
        {
            result[j] = M[row, j];
        }
        return result;
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

        int n = X.Rows;
        int m = Z.Rows;

        // Compute kernel matrices
        var Kuu = ComputeKernelMatrix(Z, Z, kernel);
        var Kuf = ComputeKernelMatrix(Z, X, kernel);

        // Add jitter to Kuu
        for (int i = 0; i < m; i++)
        {
            Kuu[i, i] = _numOps.Add(Kuu[i, i], _numOps.FromDouble(1e-6));
        }

        // Cholesky decomposition of Kuu
        var Luu = CholeskyDecomposition(Kuu);
        if (Luu is null)
        {
            return _numOps.FromDouble(double.NegativeInfinity);
        }

        // Solve Luu \ Kuf
        var alpha = SolveTriangular(Luu, Kuf, lower: true);
        if (alpha is null)
        {
            return _numOps.FromDouble(double.NegativeInfinity);
        }

        // Expected log likelihood term
        double beta = 1.0 / noiseVariance;
        double logLikeTerm = 0;

        // Predictive mean: Kfu @ Kuu^{-1} @ m
        var KuuInvM = SolveTriangularSystem(Luu, variationalMean);
        if (KuuInvM is null)
        {
            return _numOps.FromDouble(double.NegativeInfinity);
        }

        for (int i = 0; i < n; i++)
        {
            // Mean at training point i
            double mean = 0;
            for (int j = 0; j < m; j++)
            {
                mean += _numOps.ToDouble(Kuf[j, i]) * _numOps.ToDouble(KuuInvM[j]);
            }

            double diff = _numOps.ToDouble(y[i]) - mean;
            logLikeTerm += -0.5 * Math.Log(2 * Math.PI / beta) - 0.5 * beta * diff * diff;
        }

        // Trace term for variance correction
        double traceTerm = 0;
        for (int i = 0; i < n; i++)
        {
            // Kii - sum_j alpha[j,i]^2
            var xi = GetRow(X, i);
            double kii = _numOps.ToDouble(kernel.Calculate(xi, xi));

            double qii = 0;
            for (int j = 0; j < m; j++)
            {
                double alphaJi = _numOps.ToDouble(alpha[j, i]);
                qii += alphaJi * alphaJi;
            }

            traceTerm += kii - qii;
        }
        logLikeTerm -= 0.5 * beta * traceTerm;

        // KL divergence term: KL(q(u) || p(u))
        // = 0.5 * (tr(Kuu^{-1} S) + m^T Kuu^{-1} m - M + log|Kuu| - log|S|)
        double klTerm = 0;

        // log|Kuu|
        double logDetKuu = 0;
        for (int i = 0; i < m; i++)
        {
            logDetKuu += 2 * Math.Log(Math.Max(1e-10, Math.Abs(_numOps.ToDouble(Luu[i, i]))));
        }

        // log|S|
        double logDetS = 0;
        for (int i = 0; i < m; i++)
        {
            logDetS += 2 * Math.Log(Math.Max(1e-10, Math.Abs(_numOps.ToDouble(variationalCovChol[i, i]))));
        }

        // m^T Kuu^{-1} m
        double mKuuInvM = 0;
        for (int i = 0; i < m; i++)
        {
            mKuuInvM += _numOps.ToDouble(variationalMean[i]) * _numOps.ToDouble(KuuInvM[i]);
        }

        // Simplified trace term (assumes S is well-conditioned)
        double traceKuuInvS = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                traceKuuInvS += _numOps.ToDouble(variationalCovChol[i, j]) * _numOps.ToDouble(variationalCovChol[i, j]);
            }
        }
        traceKuuInvS /= (logDetKuu / (2 * m) + 1e-6);

        klTerm = 0.5 * (traceKuuInvS + mKuuInvM - m + logDetKuu - logDetS);

        double elbo = logLikeTerm - klTerm;
        return _numOps.FromDouble(elbo);
    }

    /// <summary>
    /// Computes a kernel matrix between two sets of points.
    /// </summary>
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
    /// Performs Cholesky decomposition of a symmetric positive definite matrix.
    /// </summary>
    private Matrix<T>? CholeskyDecomposition(Matrix<T> A)
    {
        int n = A.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = _numOps.ToDouble(A[i, j]);

                for (int k = 0; k < j; k++)
                {
                    sum -= _numOps.ToDouble(L[i, k]) * _numOps.ToDouble(L[j, k]);
                }

                if (i == j)
                {
                    if (sum <= 0)
                    {
                        return null; // Not positive definite
                    }
                    L[i, j] = _numOps.FromDouble(Math.Sqrt(sum));
                }
                else
                {
                    double ljj = _numOps.ToDouble(L[j, j]);
                    if (Math.Abs(ljj) < 1e-10)
                    {
                        return null;
                    }
                    L[i, j] = _numOps.FromDouble(sum / ljj);
                }
            }
        }

        return L;
    }

    /// <summary>
    /// Solves L @ X = B for X where L is lower triangular.
    /// </summary>
    private Matrix<T>? SolveTriangular(Matrix<T> L, Matrix<T> B, bool lower = true)
    {
        int m = L.Rows;
        int n = B.Columns;
        var X = new Matrix<T>(m, n);

        if (lower)
        {
            for (int col = 0; col < n; col++)
            {
                for (int i = 0; i < m; i++)
                {
                    double sum = _numOps.ToDouble(B[i, col]);
                    for (int j = 0; j < i; j++)
                    {
                        sum -= _numOps.ToDouble(L[i, j]) * _numOps.ToDouble(X[j, col]);
                    }
                    double lii = _numOps.ToDouble(L[i, i]);
                    if (Math.Abs(lii) < 1e-10)
                    {
                        return null;
                    }
                    X[i, col] = _numOps.FromDouble(sum / lii);
                }
            }
        }
        else
        {
            for (int col = 0; col < n; col++)
            {
                for (int i = m - 1; i >= 0; i--)
                {
                    double sum = _numOps.ToDouble(B[i, col]);
                    for (int j = i + 1; j < m; j++)
                    {
                        sum -= _numOps.ToDouble(L[i, j]) * _numOps.ToDouble(X[j, col]);
                    }
                    double lii = _numOps.ToDouble(L[i, i]);
                    if (Math.Abs(lii) < 1e-10)
                    {
                        return null;
                    }
                    X[i, col] = _numOps.FromDouble(sum / lii);
                }
            }
        }

        return X;
    }

    /// <summary>
    /// Solves L @ L^T @ x = b for x.
    /// </summary>
    private Vector<T>? SolveTriangularSystem(Matrix<T> L, Vector<T> b)
    {
        int n = L.Rows;

        // Forward substitution: L @ z = b
        var z = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double sum = _numOps.ToDouble(b[i]);
            for (int j = 0; j < i; j++)
            {
                sum -= _numOps.ToDouble(L[i, j]) * _numOps.ToDouble(z[j]);
            }
            double lii = _numOps.ToDouble(L[i, i]);
            if (Math.Abs(lii) < 1e-10)
            {
                return null;
            }
            z[i] = _numOps.FromDouble(sum / lii);
        }

        // Backward substitution: L^T @ x = z
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = _numOps.ToDouble(z[i]);
            for (int j = i + 1; j < n; j++)
            {
                sum -= _numOps.ToDouble(L[j, i]) * _numOps.ToDouble(x[j]);
            }
            double lii = _numOps.ToDouble(L[i, i]);
            if (Math.Abs(lii) < 1e-10)
            {
                return null;
            }
            x[i] = _numOps.FromDouble(sum / lii);
        }

        return x;
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
        var mean = new Vector<T>(m);
        for (int i = 0; i < m; i++)
        {
            double weightSum = 0;
            double valueSum = 0;

            for (int j = 0; j < n; j++)
            {
                double weight = Math.Abs(_numOps.ToDouble(Kuf[i, j]));
                weightSum += weight;
                valueSum += weight * _numOps.ToDouble(y[j]);
            }

            if (weightSum > 1e-10)
            {
                mean[i] = _numOps.FromDouble(valueSum / weightSum);
            }
        }

        // Initialize covariance Cholesky: scaled identity
        var covChol = new Matrix<T>(m, m);
        double scale = Math.Sqrt(Math.Max(noiseVariance, 0.1));
        for (int i = 0; i < m; i++)
        {
            covChol[i, i] = _numOps.FromDouble(scale);
        }

        return (mean, covChol);
    }
}
