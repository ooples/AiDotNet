using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Sparse Variational Gaussian Process (SVGP) for scalable GP regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard Gaussian Processes are powerful but slow - they require O(n³)
/// computation time, making them impractical for datasets larger than a few thousand points.
///
/// SVGP solves this problem using two key ideas:
///
/// 1. **Inducing Points**: Instead of using all training points, we use a smaller set of
///    "representative" points (inducing points) that summarize the data. If we have n training
///    points but only m inducing points (where m &lt;&lt; n), computation becomes O(nm²) instead of O(n³).
///
/// 2. **Variational Inference**: We approximate the true posterior distribution with a simpler
///    distribution (variational distribution) that's easier to work with. We optimize this
///    approximation to be as close as possible to the true posterior.
///
/// The result is a GP that can handle millions of data points while still providing
/// uncertainty estimates and probabilistic predictions.
/// </para>
/// <para>
/// When to use SVGP:
/// - Large datasets (thousands to millions of points)
/// - When you need mini-batch training (can't fit all data in memory)
/// - When you want uncertainty estimates but standard GP is too slow
/// - When inducing points can reasonably summarize your data
///
/// Limitations:
/// - The approximation quality depends on the number and placement of inducing points
/// - More hyperparameters to tune compared to standard GP
/// - May require more iterations to converge
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Scalable Variational Gaussian Process Classification", "https://doi.org/10.48550/arXiv.1411.2005", Year = 2015, Authors = "James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani")]
public class SparseVariationalGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The kernel function that determines similarity between data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel function measures how similar two points are.
    /// In SVGP, it's used to compute correlations between:
    /// - Inducing points and training points
    /// - Inducing points with each other
    /// - Test points and inducing points
    /// </para>
    /// </remarks>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The matrix of input features from the training data.
    /// </summary>
    private Matrix<T> _X;

    /// <summary>
    /// The vector of target values from the training data.
    /// </summary>
    private Vector<T> _y;

    /// <summary>
    /// The inducing points that summarize the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Inducing points are a smaller set of "summary" points that
    /// represent the key patterns in your training data. Instead of using all n training
    /// points (which is expensive), we use m inducing points where m is much smaller.
    ///
    /// Think of them like cluster centroids or strategic sampling points that capture
    /// the essential structure of your data.
    ///
    /// The locations of inducing points can be:
    /// - Fixed (initialized randomly or via k-means and kept constant)
    /// - Optimized (learned during training to best summarize the data)
    /// </para>
    /// </remarks>
    private Matrix<T> _inducingPoints;

    /// <summary>
    /// The variational mean of the approximate posterior at inducing points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In variational inference, we approximate the true posterior
    /// (which is complex) with a simpler distribution called the variational distribution.
    ///
    /// The variational mean (m) represents the expected value of the latent function
    /// at each inducing point. These are the "best guesses" for the function values.
    ///
    /// During optimization, we adjust these values to make our approximation better.
    /// </para>
    /// </remarks>
    private Vector<T> _variationalMean;

    /// <summary>
    /// The Cholesky factor of the variational covariance matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variational covariance captures our uncertainty about
    /// the function values at inducing points. We store the Cholesky factor (L) instead
    /// of the full covariance matrix (S = L * L^T) because:
    ///
    /// 1. It's more numerically stable
    /// 2. It guarantees the covariance stays positive definite (valid)
    /// 3. It's more efficient for certain computations
    ///
    /// A larger covariance means more uncertainty; smaller means more confidence.
    /// </para>
    /// </remarks>
    private Matrix<T> _variationalCovCholesky;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The method used for matrix decomposition in linear system solving.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// The observation noise variance (likelihood parameter).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter represents how much random noise you expect
    /// in your observations. If your measurements are precise, use a small value.
    /// If your measurements are noisy, use a larger value.
    ///
    /// The noise variance affects both:
    /// - How closely the model fits the training data (larger noise = looser fit)
    /// - The uncertainty in predictions (larger noise = more uncertainty)
    /// </para>
    /// </remarks>
    private readonly double _noiseVariance;

    /// <summary>
    /// The number of inducing points to use.
    /// </summary>
    private readonly int _numInducingPoints;

    /// <summary>
    /// Learning rate for variational parameter updates.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Number of optimization iterations.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Kernel matrix between inducing points (Kuu).
    /// </summary>
    private Matrix<T> _Kuu;

    /// <summary>
    /// Cholesky factor of Kuu for efficient computation.
    /// </summary>
    private Matrix<T> _LKuu;

    /// <summary>
    /// Initializes a new instance of the SparseVariationalGaussianProcess class.
    /// </summary>
    /// <param name="kernel">The kernel function to use for measuring similarity.</param>
    /// <param name="numInducingPoints">Number of inducing points. Default is 100.</param>
    /// <param name="noiseVariance">Observation noise variance. Default is 1e-4.</param>
    /// <param name="learningRate">Learning rate for optimization. Default is 0.01.</param>
    /// <param name="maxIterations">Maximum optimization iterations. Default is 1000.</param>
    /// <param name="decompositionType">Matrix decomposition method. Default is Cholesky.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the SVGP model with your chosen settings.
    ///
    /// Key parameters to consider:
    ///
    /// - numInducingPoints: More points = better approximation but slower. Start with 100-500.
    ///   As a rule of thumb, use 5-20% of your dataset size, capped at around 1000.
    ///
    /// - noiseVariance: Reflects how noisy your data is. If your measurements are precise,
    ///   use a small value (1e-6). If noisy, use a larger value (1e-2 or higher).
    ///
    /// - learningRate: How fast to update parameters. Too large = unstable; too small = slow.
    ///   Start with 0.01 and adjust if training is unstable or slow.
    ///
    /// - maxIterations: How many optimization steps. More iterations = better fit but slower.
    ///   Monitor the ELBO to see if more iterations are needed.
    /// </para>
    /// </remarks>
    public SparseVariationalGaussianProcess(
        IKernelFunction<T> kernel,
        int numInducingPoints = 100,
        double noiseVariance = 1e-4,
        double learningRate = 0.01,
        int maxIterations = 1000,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        if (numInducingPoints < 1)
            throw new ArgumentException("Number of inducing points must be at least 1.", nameof(numInducingPoints));
        if (noiseVariance < 0)
            throw new ArgumentException("Noise variance must be non-negative.", nameof(noiseVariance));
        if (learningRate <= 0)
            throw new ArgumentException("Learning rate must be positive.", nameof(learningRate));

        _kernel = kernel;
        _numInducingPoints = numInducingPoints;
        _noiseVariance = noiseVariance;
        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize empty matrices/vectors
        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _inducingPoints = Matrix<T>.Empty();
        _variationalMean = Vector<T>.Empty();
        _variationalCovCholesky = Matrix<T>.Empty();
        _Kuu = Matrix<T>.Empty();
        _LKuu = Matrix<T>.Empty();
    }

    /// <summary>
    /// Trains the SVGP model using variational inference and gradient-based optimization.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a data point.</param>
    /// <param name="y">The target values corresponding to each input point.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains the SVGP model on your data using variational inference.
    ///
    /// The training process:
    ///
    /// 1. **Initialize inducing points**: Select m representative points from your data
    ///
    /// 2. **Initialize variational parameters**:
    ///    - Variational mean (m): starts at zero
    ///    - Variational covariance (S): starts as identity matrix
    ///
    /// 3. **Optimize the ELBO** (Evidence Lower Bound):
    ///    - The ELBO is a lower bound on the log marginal likelihood
    ///    - Maximizing ELBO makes our approximation closer to the true posterior
    ///    - We use gradient ascent to iteratively improve the parameters
    ///
    /// The ELBO has two terms:
    /// - Data fit: How well the model explains the observed data
    /// - KL divergence: Penalty for deviating from the prior (prevents overfitting)
    ///
    /// After training, the model can make predictions with uncertainty estimates.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _y = y;

        // Initialize inducing points from training data
        int numInduce = Math.Min(_numInducingPoints, X.Rows);
        _inducingPoints = SelectInducingPoints(X, numInduce);

        // Compute kernel matrix between inducing points
        _Kuu = CalculateKernelMatrix(_inducingPoints, _inducingPoints);
        AddJitter(_Kuu);

        // Cholesky decomposition of Kuu for efficient computation
        var choleskyKuu = new CholeskyDecomposition<T>(_Kuu);
        _LKuu = choleskyKuu.L;

        // Initialize variational parameters
        _variationalMean = new Vector<T>(numInduce);
        _variationalCovCholesky = CreateIdentityMatrix(numInduce);

        // Optimize variational parameters
        OptimizeVariationalParameters();
    }

    /// <summary>
    /// Selects inducing points from the training data using random sampling.
    /// </summary>
    /// <param name="X">The input features matrix.</param>
    /// <param name="numPoints">Number of inducing points to select.</param>
    /// <returns>Matrix of inducing points.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method selects which training points will serve as inducing points.
    ///
    /// We use random sampling, which is simple and often works well. More sophisticated
    /// methods include:
    /// - K-means clustering (use cluster centers as inducing points)
    /// - Greedy selection (iteratively add points that maximize information)
    /// - Learning the locations during optimization
    ///
    /// Random sampling is a good starting point. If your model isn't performing well,
    /// try increasing the number of inducing points or using k-means initialization.
    /// </para>
    /// </remarks>
    private Matrix<T> SelectInducingPoints(Matrix<T> X, int numPoints)
    {
        if (numPoints >= X.Rows)
        {
            return X;
        }

        var indices = new List<int>();
        var random = RandomHelper.CreateSecureRandom();

        while (indices.Count < numPoints)
        {
            int index = random.Next(0, X.Rows);
            if (!indices.Contains(index))
            {
                indices.Add(index);
            }
        }

        return X.GetRows(indices);
    }

    /// <summary>
    /// Optimizes the variational parameters using gradient ascent on the ELBO.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is the core of SVGP training. It adjusts the
    /// variational parameters (mean and covariance) to maximize the ELBO.
    ///
    /// The ELBO (Evidence Lower Bound) is:
    ///   ELBO = E[log p(y|f)] - KL(q(u)||p(u))
    ///
    /// Where:
    /// - E[log p(y|f)]: Expected log-likelihood (how well we fit the data)
    /// - KL(...): KL divergence (penalty for deviating from prior)
    ///
    /// We use gradient ascent:
    /// 1. Compute the gradient of ELBO with respect to variational parameters
    /// 2. Update parameters in the direction of the gradient
    /// 3. Repeat until convergence or max iterations
    ///
    /// The learning rate controls how big each step is. Too large = unstable;
    /// too small = slow convergence.
    /// </para>
    /// </remarks>
    private void OptimizeVariationalParameters()
    {
        // Compute kernel matrix between inducing points and training points
        var Kuf = CalculateKernelMatrix(_inducingPoints, _X);

        // Precompute Kuu^(-1) * Kuf using Cholesky solve
        var KuuInvKuf = new Matrix<T>(_Kuu.Rows, Kuf.Columns);
        for (int j = 0; j < Kuf.Columns; j++)
        {
            var col = Kuf.GetColumn(j);
            var solved = MatrixSolutionHelper.SolveLinearSystem(_Kuu, col, _decompositionType);
            KuuInvKuf.SetColumn(j, solved);
        }

        double noiseVariance = Math.Max(_noiseVariance, 1e-10);
        T noisePrecision = _numOps.FromDouble(1.0 / noiseVariance);

        // For Gaussian likelihood, the optimal variational parameters have closed-form solutions:
        //   S* = (Kuu^{-1} + σ^{-2} Kuf Kuf^T)^{-1}
        //   m* = σ^{-2} S* Kuf y
        //
        // Using iterative gradient descent is unnecessary and numerically unstable (the gradient
        // magnitude is proportional to σ^{-2} which can be 10^4+, causing divergence).
        // The closed-form solution is exact and avoids all convergence issues.

        // Step 1: Compute S (variational covariance) — already numerically stable
        UpdateVariationalCovariance(Kuf, noisePrecision);

        // Step 2: Compute m* = σ^{-2} S Kuf y using the Cholesky factor L where S = L L^T
        //   m* = σ^{-2} L L^T Kuf y = L (σ^{-2} L^T Kuf y)
        var Kufy = Kuf.Multiply(_y);
        var scaledKufy = new Vector<T>(Kufy.Length);
        for (int i = 0; i < Kufy.Length; i++)
        {
            scaledKufy[i] = _numOps.Multiply(noisePrecision, Kufy[i]);
        }

        // Compute L^T * scaledKufy
        var LT = _variationalCovCholesky.Transpose();
        var LTKufy = LT.Multiply(scaledKufy);

        // Compute m = L * LTKufy
        _variationalMean = _variationalCovCholesky.Multiply(LTKufy);
    }

    /// <summary>
    /// Updates the variational covariance based on the data.
    /// </summary>
    /// <param name="Kuf">Kernel matrix between inducing and training points.</param>
    /// <param name="noisePrecision">Precision (inverse variance) of observation noise.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variational covariance represents our uncertainty about
    /// the function values at inducing points. After seeing data, our uncertainty should decrease.
    ///
    /// The optimal variational covariance (in closed form for Gaussian likelihood) is:
    ///   S = (Kuu^(-1) + Kuf * Kfu / σ²)^(-1)
    ///
    /// This balances:
    /// - Prior uncertainty (from Kuu)
    /// - Information from data (from Kuf terms)
    ///
    /// We store the Cholesky factor of S for numerical stability.
    /// </para>
    /// </remarks>
    private void UpdateVariationalCovariance(Matrix<T> Kuf, T noisePrecision)
    {
        // Numerically stable computation of S = (Kuu^{-1} + σ^{-2} Kuf Kuf^T)^{-1}
        //
        // Instead of explicitly computing SInv then inverting (which amplifies numerical error),
        // we use the Cholesky of Kuu and work in a transformed space:
        //   Let Luu = chol(Kuu), so Kuu = Luu Luu^T
        //   Define A = Luu^{-1} Kuf / σ  (scaled, whitened cross-covariance)
        //   Then SInv = Kuu^{-1} + σ^{-2} Kuf Kuf^T
        //             = Luu^{-T} (I + A A^T) Luu^{-1}
        //   So S = Luu (I + A A^T)^{-1} Luu^T
        //   And chol(S) = Luu * chol((I + A A^T)^{-1})
        //
        // The matrix (I + A A^T) is better conditioned than SInv because its eigenvalues
        // are ≥ 1, avoiding the large condition numbers from noisePrecision * Kuf * Kuf^T.

        int m = _Kuu.Rows;
        int n = Kuf.Columns;

        // Compute A = Luu^{-1} Kuf / σ  (forward-solve each column)
        double sqrtPrecision = Math.Sqrt(Math.Max(_numOps.ToDouble(noisePrecision), 0.0));
        var A = new Matrix<T>(m, n);
        for (int j = 0; j < n; j++)
        {
            var col = Kuf.GetColumn(j);
            // Forward-solve: Luu * x = col  =>  x = Luu^{-1} * col
            var solved = MatrixSolutionHelper.SolveLinearSystem(_LKuu.Multiply(_LKuu.Transpose()), col, _decompositionType);
            // Apply Luu^{-1}: since Kuu = Luu Luu^T, Kuu^{-1} col = Luu^{-T} Luu^{-1} col
            // We need Luu^{-1} col, so solve Luu * x = col using forward substitution
            // Approximate: use _LKuu to forward-solve
            var whitened = ForwardSolve(_LKuu, col);
            for (int i = 0; i < m; i++)
            {
                A[i, j] = _numOps.Multiply(_numOps.FromDouble(sqrtPrecision), whitened[i]);
            }
        }

        // Compute B = I + A A^T (m×m, well-conditioned)
        var AAT = A.Multiply(A.Transpose());
        var B = CreateIdentityMatrix(m);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                B[i, j] = _numOps.Add(B[i, j], AAT[i, j]);
            }
        }

        // Compute S_white = B^{-1} (in whitened space)
        AddJitter(B);
        try
        {
            var choleskyB = new CholeskyDecomposition<T>(B);

            // S_white = B^{-1}, compute its Cholesky: chol(B^{-1})
            // Since B = chol_B * chol_B^T, B^{-1} = chol_B^{-T} * chol_B^{-1}
            // So chol(B^{-1}) = chol_B^{-T} (the inverse-transpose of the Cholesky factor)
            // We can get this by solving chol_B^T * L_inv = I column by column
            var identity = CreateIdentityMatrix(m);
            var L_B = choleskyB.L;
            var L_BInvT = new Matrix<T>(m, m);
            for (int j = 0; j < m; j++)
            {
                // Solve L_B^T * x = e_j (back-substitution)
                var ej = identity.GetColumn(j);
                var solved = BackSolve(L_B.Transpose(), ej);
                L_BInvT.SetColumn(j, solved);
            }

            // chol(S) = Luu * L_BInvT
            _variationalCovCholesky = _LKuu.Multiply(L_BInvT);
        }
        catch (ArgumentException)
        {
            // Fallback: if B is not positive definite (shouldn't happen since eigenvalues ≥ 1),
            // use a scaled identity covariance
            _variationalCovCholesky = CreateIdentityMatrix(m);
            for (int i = 0; i < m; i++)
            {
                _variationalCovCholesky[i, i] = _numOps.FromDouble(0.01);
            }
        }
    }

    /// <summary>
    /// Forward-solves L * x = b for lower-triangular L.
    /// </summary>
    private Vector<T> ForwardSolve(Matrix<T> L, Vector<T> b)
    {
        int n = b.Length;
        var x = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            T sum = b[i];
            for (int j = 0; j < i; j++)
            {
                sum = _numOps.Subtract(sum, _numOps.Multiply(L[i, j], x[j]));
            }
            T diag = L[i, i];
            // Guard against zero diagonal
            if (_numOps.ToDouble(diag) == 0.0)
            {
                diag = _numOps.FromDouble(1e-10);
            }
            x[i] = _numOps.Divide(sum, diag);
        }
        return x;
    }

    /// <summary>
    /// Back-solves U * x = b for upper-triangular U.
    /// </summary>
    private Vector<T> BackSolve(Matrix<T> U, Vector<T> b)
    {
        int n = b.Length;
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            T sum = b[i];
            for (int j = i + 1; j < n; j++)
            {
                sum = _numOps.Subtract(sum, _numOps.Multiply(U[i, j], x[j]));
            }
            T diag = U[i, i];
            if (_numOps.ToDouble(diag) == 0.0)
            {
                diag = _numOps.FromDouble(1e-10);
            }
            x[i] = _numOps.Divide(sum, diag);
        }
        return x;
    }

    /// <summary>
    /// Computes the predictive mean at training points.
    /// </summary>
    /// <param name="KuuInvKuf">Precomputed Kuu^(-1) * Kuf matrix.</param>
    /// <returns>Predictive mean vector at training points.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The predictive mean is our best guess for the function value
    /// at each training point, given the current variational parameters.
    ///
    /// f_mean = Kfu * Kuu^(-1) * m
    ///
    /// Where m is the variational mean at inducing points. This formula "interpolates"
    /// the function values from inducing points to training points using the kernel.
    /// </para>
    /// </remarks>
    private Vector<T> ComputePredictiveMean(Matrix<T> KuuInvKuf)
    {
        // predictive_mean = Kfu * Kuu^(-1) * m = (Kuu^(-1) * Kuf)^T * m
        var predictiveMean = new Vector<T>(KuuInvKuf.Columns);

        for (int i = 0; i < KuuInvKuf.Columns; i++)
        {
            T sum = _numOps.Zero;
            for (int j = 0; j < _variationalMean.Length; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(KuuInvKuf[j, i], _variationalMean[j]));
            }
            predictiveMean[i] = sum;
        }

        return predictiveMean;
    }

    /// <inheritdoc/>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        if (_inducingPoints.IsEmpty || _variationalMean.IsEmpty)
        {
            throw new InvalidOperationException("Model must be trained before prediction. Call Fit() first.");
        }

        // Compute kernel vector between test point and inducing points
        var kStar = CalculateKernelVector(_inducingPoints, x);

        // Compute Kuu^(-1) * k*
        var KuuInvKStar = MatrixSolutionHelper.SolveLinearSystem(_Kuu, kStar, _decompositionType);

        // Predictive mean: k*^T * Kuu^(-1) * m
        T mean = _numOps.Zero;
        for (int i = 0; i < _variationalMean.Length; i++)
        {
            mean = _numOps.Add(mean, _numOps.Multiply(KuuInvKStar[i], _variationalMean[i]));
        }

        // Predictive variance: k** - k*^T * Kuu^(-1) * k* + k*^T * Kuu^(-1) * S * Kuu^(-1) * k*
        T kStarStar = _kernel.Calculate(x, x);

        // First term: k** - k*^T * Kuu^(-1) * k*
        T variance = kStarStar;
        for (int i = 0; i < kStar.Length; i++)
        {
            variance = _numOps.Subtract(variance, _numOps.Multiply(kStar[i], KuuInvKStar[i]));
        }

        // Second term: k*^T * Kuu^(-1) * S * Kuu^(-1) * k*
        // S = L * L^T, so this is ||L^T * Kuu^(-1) * k*||²
        var LTKuuInvKStar = _variationalCovCholesky.Transpose().Multiply(KuuInvKStar);
        T additionalVar = _numOps.Zero;
        for (int i = 0; i < LTKuuInvKStar.Length; i++)
        {
            additionalVar = _numOps.Add(additionalVar, _numOps.Multiply(LTKuuInvKStar[i], LTKuuInvKStar[i]));
        }
        variance = _numOps.Add(variance, additionalVar);

        // Add observation noise
        variance = _numOps.Add(variance, _numOps.FromDouble(_noiseVariance));

        return (mean, variance);
    }

    /// <inheritdoc/>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_X.IsEmpty && !_y.IsEmpty)
        {
            Fit(_X, _y);
        }
    }

    /// <summary>
    /// Computes the Evidence Lower Bound (ELBO) for the current variational approximation.
    /// </summary>
    /// <returns>The ELBO value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The ELBO is a score that tells us how good our approximation is.
    ///
    /// ELBO = E[log p(y|f)] - KL(q(u)||p(u))
    ///
    /// Higher ELBO = better approximation. During training, we maximize the ELBO.
    ///
    /// The two terms:
    /// - E[log p(y|f)]: Expected likelihood - how well we fit the data
    /// - KL(...): Regularization - penalty for deviating from prior
    ///
    /// You can use the ELBO for:
    /// - Monitoring training progress
    /// - Model comparison (higher ELBO = better model)
    /// - Hyperparameter selection
    /// </para>
    /// </remarks>
    public T ComputeELBO()
    {
        if (_X.IsEmpty)
            return _numOps.Zero;

        // Compute kernel matrices
        var Kuf = CalculateKernelMatrix(_inducingPoints, _X);

        // Compute predictive mean and variance at training points
        var KuuInvKuf = new Matrix<T>(_Kuu.Rows, Kuf.Columns);
        for (int j = 0; j < Kuf.Columns; j++)
        {
            var col = Kuf.GetColumn(j);
            var solved = MatrixSolutionHelper.SolveLinearSystem(_Kuu, col, _decompositionType);
            KuuInvKuf.SetColumn(j, solved);
        }

        var predictiveMean = ComputePredictiveMean(KuuInvKuf);

        // Expected log-likelihood (Gaussian)
        double logLik = 0.0;
        double noiseVar = Math.Max(_noiseVariance, 1e-10);
        for (int i = 0; i < _y.Length; i++)
        {
            double diff = _numOps.ToDouble(_y[i]) - _numOps.ToDouble(predictiveMean[i]);
            logLik -= 0.5 * diff * diff / noiseVar;
            logLik -= 0.5 * Math.Log(2 * Math.PI * noiseVar);
        }

        // KL divergence: KL(q(u)||p(u))
        // KL = 0.5 * (tr(Kuu^(-1) * S) + m^T * Kuu^(-1) * m - m + log|Kuu| - log|S|)
        var S = _variationalCovCholesky.Multiply(_variationalCovCholesky.Transpose());

        // tr(Kuu^(-1) * S)
        double trace = 0.0;
        for (int i = 0; i < _Kuu.Rows; i++)
        {
            var ei = new Vector<T>(_Kuu.Rows);
            ei[i] = _numOps.One;
            var KuuInvEi = MatrixSolutionHelper.SolveLinearSystem(_Kuu, ei, _decompositionType);
            for (int j = 0; j < S.Columns; j++)
            {
                trace += _numOps.ToDouble(KuuInvEi[j]) * _numOps.ToDouble(S[j, i]);
            }
        }

        // m^T * Kuu^(-1) * m
        var KuuInvM = MatrixSolutionHelper.SolveLinearSystem(_Kuu, _variationalMean, _decompositionType);
        double quadForm = 0.0;
        for (int i = 0; i < _variationalMean.Length; i++)
        {
            quadForm += _numOps.ToDouble(_variationalMean[i]) * _numOps.ToDouble(KuuInvM[i]);
        }

        // log|Kuu| = 2 * sum(log(diag(LKuu)))
        double logDetKuu = 0.0;
        for (int i = 0; i < _LKuu.Rows; i++)
        {
            logDetKuu += 2.0 * Math.Log(Math.Abs(_numOps.ToDouble(_LKuu[i, i])));
        }

        // log|S| = 2 * sum(log(diag(L_S)))
        double logDetS = 0.0;
        for (int i = 0; i < _variationalCovCholesky.Rows; i++)
        {
            logDetS += 2.0 * Math.Log(Math.Abs(_numOps.ToDouble(_variationalCovCholesky[i, i])));
        }

        double kl = 0.5 * (trace + quadForm - _variationalMean.Length + logDetKuu - logDetS);

        double elbo = logLik - kl;
        return _numOps.FromDouble(elbo);
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of data points.
    /// </summary>
    private Matrix<T> CalculateKernelMatrix(Matrix<T> X1, Matrix<T> X2)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);
        for (int i = 0; i < X1.Rows; i++)
        {
            for (int j = 0; j < X2.Rows; j++)
            {
                K[i, j] = _kernel.Calculate(X1.GetRow(i), X2.GetRow(j));
            }
        }
        return K;
    }

    /// <summary>
    /// Calculates kernel values between a matrix of points and a single vector.
    /// </summary>
    private Vector<T> CalculateKernelVector(Matrix<T> X, Vector<T> x)
    {
        var k = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            k[i] = _kernel.Calculate(X.GetRow(i), x);
        }
        return k;
    }

    /// <summary>
    /// Creates an identity matrix of the specified size.
    /// </summary>
    private Matrix<T> CreateIdentityMatrix(int size)
    {
        var identity = new Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            identity[i, i] = _numOps.One;
        }
        return identity;
    }

    /// <summary>
    /// Adds jitter to the diagonal for numerical stability.
    /// </summary>
    private void AddJitter(Matrix<T> K)
    {
        var jitter = _numOps.FromDouble(1e-6);
        for (int i = 0; i < K.Rows; i++)
        {
            K[i, i] = _numOps.Add(K[i, i], jitter);
        }
    }
}
