namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Variational Gaussian Process (VGP) using variational inference for exact data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Variational Gaussian Process (VGP) is a probabilistic model that uses
/// variational inference to approximate the posterior distribution over functions.
///
/// Unlike SVGP which uses inducing points for scalability, VGP works with all training points
/// but uses variational inference to handle non-Gaussian likelihoods (like for classification
/// or robust regression with non-Gaussian noise).
///
/// Key differences from standard GP:
/// - Standard GP: Assumes Gaussian likelihood, has closed-form solution
/// - VGP: Can handle any likelihood, uses optimization to find approximate posterior
///
/// When to use VGP:
/// - When you have non-Gaussian likelihoods (classification, count data, etc.)
/// - When you want uncertainty quantification with flexible likelihood models
/// - When your dataset is small enough to use all points (up to ~5000 points)
///
/// For large datasets with Gaussian likelihood, use SparseVariationalGaussianProcess instead.
/// </para>
/// </remarks>
public class VariationalGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The kernel function that determines similarity between data points.
    /// </summary>
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
    /// The kernel matrix computed from training data.
    /// </summary>
    private Matrix<T> _K;

    /// <summary>
    /// The variational mean of the approximate posterior.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variational mean represents our best estimate of the
    /// latent function values at each training point. These are optimized during training
    /// to maximize the evidence lower bound (ELBO).
    /// </para>
    /// </remarks>
    private Vector<T> _variationalMean;

    /// <summary>
    /// The Cholesky factor of the variational covariance matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The variational covariance captures our uncertainty about
    /// the latent function values. We store the Cholesky factor for numerical stability
    /// and to ensure the covariance remains positive definite.
    /// </para>
    /// </remarks>
    private Matrix<T> _variationalCovCholesky;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The method used for matrix decomposition.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// The observation noise variance.
    /// </summary>
    private readonly double _noiseVariance;

    /// <summary>
    /// Learning rate for optimization.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Maximum number of optimization iterations.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance for optimization.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// The likelihood type for the model.
    /// </summary>
    private readonly VGPLikelihood _likelihood;

    /// <summary>
    /// Cholesky factor of the kernel matrix for efficient computation.
    /// </summary>
    private Matrix<T> _LK;

    /// <summary>
    /// Initializes a new instance of the VariationalGaussianProcess class.
    /// </summary>
    /// <param name="kernel">The kernel function to use.</param>
    /// <param name="likelihood">The likelihood function type. Default is Gaussian.</param>
    /// <param name="noiseVariance">Observation noise variance. Default is 1e-4.</param>
    /// <param name="learningRate">Learning rate for optimization. Default is 0.01.</param>
    /// <param name="maxIterations">Maximum optimization iterations. Default is 500.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-6.</param>
    /// <param name="decompositionType">Matrix decomposition method. Default is Cholesky.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the VGP with your chosen configuration.
    ///
    /// Key parameters:
    ///
    /// - likelihood: What type of output you're modeling
    ///   - Gaussian: Continuous values with normal noise (regression)
    ///   - Bernoulli: Binary outcomes (classification)
    ///   - Poisson: Count data (0, 1, 2, ...)
    ///
    /// - noiseVariance: How much random noise in observations (for Gaussian likelihood)
    ///
    /// - learningRate: How fast to update during optimization
    ///   - Too large: Training becomes unstable
    ///   - Too small: Training takes too long
    ///
    /// - maxIterations: When to stop if not converged
    /// </para>
    /// </remarks>
    public VariationalGaussianProcess(
        IKernelFunction<T> kernel,
        VGPLikelihood likelihood = VGPLikelihood.Gaussian,
        double noiseVariance = 1e-4,
        double learningRate = 0.01,
        int maxIterations = 500,
        double tolerance = 1e-6,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky)
    {
        if (noiseVariance < 0)
            throw new ArgumentException("Noise variance must be non-negative.", nameof(noiseVariance));
        if (learningRate <= 0)
            throw new ArgumentException("Learning rate must be positive.", nameof(learningRate));

        _kernel = kernel;
        _likelihood = likelihood;
        _noiseVariance = noiseVariance;
        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _decompositionType = decompositionType;
        _numOps = MathHelper.GetNumericOperations<T>();

        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _K = Matrix<T>.Empty();
        _variationalMean = Vector<T>.Empty();
        _variationalCovCholesky = Matrix<T>.Empty();
        _LK = Matrix<T>.Empty();
    }

    /// <summary>
    /// Trains the VGP model using variational inference.
    /// </summary>
    /// <param name="X">The input features matrix.</param>
    /// <param name="y">The target values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains the VGP on your data.
    ///
    /// The training process:
    /// 1. Compute the kernel matrix between all training points
    /// 2. Initialize variational parameters (mean = 0, covariance = prior)
    /// 3. Iteratively optimize to maximize the ELBO:
    ///    - Compute expected log-likelihood
    ///    - Compute KL divergence from prior
    ///    - Update parameters in direction that improves ELBO
    ///
    /// For Gaussian likelihood, there's a closed-form optimal solution.
    /// For other likelihoods, we use gradient-based optimization.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _y = y;

        // Compute kernel matrix
        _K = CalculateKernelMatrix(X, X);
        AddJitter(_K);

        // Cholesky decomposition for efficient computation
        var choleskyK = new CholeskyDecomposition<T>(_K);
        _LK = choleskyK.L;

        // Initialize variational parameters
        int n = X.Rows;
        _variationalMean = new Vector<T>(n);
        _variationalCovCholesky = CreateScaledIdentityMatrix(n, 1.0);

        // Optimize based on likelihood type
        if (_likelihood == VGPLikelihood.Gaussian)
        {
            // Closed-form solution for Gaussian likelihood
            OptimizeGaussianLikelihood();
        }
        else
        {
            // Gradient-based optimization for non-Gaussian likelihoods
            OptimizeNonGaussianLikelihood();
        }
    }

    /// <summary>
    /// Optimizes variational parameters for Gaussian likelihood (closed-form).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Gaussian likelihood, the optimal variational distribution
    /// can be computed exactly without iteration.
    ///
    /// The optimal posterior is:
    /// - Mean: m = (K + σ²I)^(-1) * K * y / σ² (simplified: just the GP posterior mean)
    /// - Covariance: S = K - K * (K + σ²I)^(-1) * K (the GP posterior covariance)
    ///
    /// This is equivalent to standard GP regression but expressed in variational form.
    /// </para>
    /// </remarks>
    private void OptimizeGaussianLikelihood()
    {
        int n = _X.Rows;
        double noiseVar = Math.Max(_noiseVariance, 1e-10);

        // Compute (K + σ²I)^(-1)
        var KPlusNoise = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                KPlusNoise[i, j] = _K[i, j];
            }
            KPlusNoise[i, i] = _numOps.Add(KPlusNoise[i, i], _numOps.FromDouble(noiseVar));
        }

        // Solve (K + σ²I) * alpha = y
        var alpha = MatrixSolutionHelper.SolveLinearSystem(KPlusNoise, _y, _decompositionType);

        // Variational mean = K * alpha
        _variationalMean = _K.Multiply(alpha);

        // Compute posterior covariance Cholesky
        // S = K - K * (K + σ²I)^(-1) * K
        var KKplusNoiseInv = new Matrix<T>(n, n);
        for (int j = 0; j < n; j++)
        {
            var col = _K.GetColumn(j);
            var solved = MatrixSolutionHelper.SolveLinearSystem(KPlusNoise, col, _decompositionType);
            KKplusNoiseInv.SetColumn(j, solved);
        }

        var S = new Matrix<T>(n, n);
        var KKpn = _K.Multiply(KKplusNoiseInv);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                S[i, j] = _numOps.Subtract(_K[i, j], KKpn[i, j]);
            }
        }

        AddJitter(S);
        try
        {
            var choleskyS = new CholeskyDecomposition<T>(S);
            _variationalCovCholesky = choleskyS.L;
        }
        catch (Exception ex)
        {
            // If Cholesky fails, use scaled identity (matrix may not be positive definite)
            System.Diagnostics.Debug.WriteLine($"Cholesky decomposition failed: {ex.Message}. Using scaled identity.");
            _variationalCovCholesky = CreateScaledIdentityMatrix(n, 0.1);
        }
    }

    /// <summary>
    /// Optimizes variational parameters for non-Gaussian likelihoods using gradient ascent.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For non-Gaussian likelihoods (like Bernoulli for classification),
    /// there's no closed-form solution, so we use iterative optimization.
    ///
    /// At each iteration:
    /// 1. Compute the gradient of ELBO with respect to variational mean
    /// 2. Update the mean in the gradient direction
    /// 3. Update the covariance based on the likelihood curvature
    /// 4. Check for convergence
    ///
    /// The process continues until the ELBO stops improving significantly.
    /// </para>
    /// </remarks>
    private void OptimizeNonGaussianLikelihood()
    {
        int n = _X.Rows;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute gradient of expected log-likelihood
            var gradLogLik = ComputeLikelihoodGradient();

            // Compute gradient from KL divergence (prior term)
            var KInvM = MatrixSolutionHelper.SolveLinearSystem(_K, _variationalMean, _decompositionType);

            // Total gradient for mean: grad_loglik - K^(-1) * m
            var gradMean = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                gradMean[i] = _numOps.Subtract(gradLogLik[i], KInvM[i]);
            }

            // Update variational mean
            T maxChange = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T update = _numOps.Multiply(_numOps.FromDouble(_learningRate), gradMean[i]);
                _variationalMean[i] = _numOps.Add(_variationalMean[i], update);

                T absUpdate = _numOps.Abs(update);
                if (_numOps.ToDouble(absUpdate) > _numOps.ToDouble(maxChange))
                {
                    maxChange = absUpdate;
                }
            }

            // Update covariance based on likelihood Hessian (simplified)
            if (iter == 0 || iter % 50 == 0)
            {
                UpdateVariationalCovariance();
            }

            // Check convergence
            if (_numOps.ToDouble(maxChange) < _tolerance)
            {
                break;
            }
        }
    }

    /// <summary>
    /// Computes the gradient of the expected log-likelihood.
    /// </summary>
    /// <returns>Gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient tells us how to adjust the variational mean
    /// to better fit the observed data.
    ///
    /// For different likelihoods:
    /// - Gaussian: gradient = (y - m) / σ²
    /// - Bernoulli: gradient = y - sigmoid(m)
    /// - Poisson: gradient = y - exp(m)
    ///
    /// Each tells us which direction to push the mean to increase likelihood.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeLikelihoodGradient()
    {
        int n = _variationalMean.Length;
        var gradient = new Vector<T>(n);

        switch (_likelihood)
        {
            case VGPLikelihood.Gaussian:
                double noisePrecision = 1.0 / Math.Max(_noiseVariance, 1e-10);
                for (int i = 0; i < n; i++)
                {
                    double diff = _numOps.ToDouble(_y[i]) - _numOps.ToDouble(_variationalMean[i]);
                    gradient[i] = _numOps.FromDouble(diff * noisePrecision);
                }
                break;

            case VGPLikelihood.Bernoulli:
                for (int i = 0; i < n; i++)
                {
                    double m = _numOps.ToDouble(_variationalMean[i]);
                    double sigmoid = 1.0 / (1.0 + Math.Exp(-m));
                    double y = _numOps.ToDouble(_y[i]);
                    gradient[i] = _numOps.FromDouble(y - sigmoid);
                }
                break;

            case VGPLikelihood.Poisson:
                for (int i = 0; i < n; i++)
                {
                    double m = _numOps.ToDouble(_variationalMean[i]);
                    double expM = Math.Exp(Math.Min(m, 20)); // Clip for numerical stability
                    double y = _numOps.ToDouble(_y[i]);
                    gradient[i] = _numOps.FromDouble(y - expM);
                }
                break;
        }

        return gradient;
    }

    /// <summary>
    /// Updates the variational covariance based on likelihood curvature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The optimal variational covariance depends on how curved
    /// the log-likelihood is at the current mean estimate.
    ///
    /// For Bernoulli likelihood:
    ///   W[i,i] = sigmoid(m[i]) * (1 - sigmoid(m[i]))
    ///
    /// For Poisson likelihood:
    ///   W[i,i] = exp(m[i])
    ///
    /// Then: S^(-1) = K^(-1) + W
    ///
    /// Points with higher curvature (more data information) have smaller variance.
    /// </para>
    /// </remarks>
    private void UpdateVariationalCovariance()
    {
        int n = _variationalMean.Length;

        // Compute diagonal of negative Hessian of log-likelihood
        var W = new Vector<T>(n);

        switch (_likelihood)
        {
            case VGPLikelihood.Gaussian:
                double noisePrecision = 1.0 / Math.Max(_noiseVariance, 1e-10);
                for (int i = 0; i < n; i++)
                {
                    W[i] = _numOps.FromDouble(noisePrecision);
                }
                break;

            case VGPLikelihood.Bernoulli:
                for (int i = 0; i < n; i++)
                {
                    double m = _numOps.ToDouble(_variationalMean[i]);
                    double sigmoid = 1.0 / (1.0 + Math.Exp(-m));
                    W[i] = _numOps.FromDouble(sigmoid * (1.0 - sigmoid));
                }
                break;

            case VGPLikelihood.Poisson:
                for (int i = 0; i < n; i++)
                {
                    double m = _numOps.ToDouble(_variationalMean[i]);
                    double expM = Math.Exp(Math.Min(m, 20));
                    W[i] = _numOps.FromDouble(expM);
                }
                break;
        }

        // Compute S^(-1) = K^(-1) + diag(W)
        // First compute K^(-1)
        var eye = CreateIdentityMatrix(n);
        var KInv = new Matrix<T>(n, n);
        for (int j = 0; j < n; j++)
        {
            var col = eye.GetColumn(j);
            var solved = MatrixSolutionHelper.SolveLinearSystem(_K, col, _decompositionType);
            KInv.SetColumn(j, solved);
        }

        // Add W to diagonal
        var SInv = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                SInv[i, j] = KInv[i, j];
            }
            SInv[i, i] = _numOps.Add(SInv[i, i], W[i]);
        }

        AddJitter(SInv);

        // Invert to get S
        try
        {
            var choleskySInv = new CholeskyDecomposition<T>(SInv);
            var S = new Matrix<T>(n, n);
            for (int j = 0; j < n; j++)
            {
                var col = eye.GetColumn(j);
                var solved = choleskySInv.Solve(col);
                S.SetColumn(j, solved);
            }

            AddJitter(S);
            var choleskyS = new CholeskyDecomposition<T>(S);
            _variationalCovCholesky = choleskyS.L;
        }
        catch (Exception ex)
        {
            // Keep current covariance if decomposition fails
            System.Diagnostics.Debug.WriteLine($"Covariance Cholesky failed: {ex.Message}. Keeping current covariance.");
        }
    }

    /// <inheritdoc/>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        if (_X.IsEmpty || _variationalMean.IsEmpty)
        {
            throw new InvalidOperationException("Model must be trained before prediction. Call Fit() first.");
        }

        // Compute kernel vector between test point and training points
        var kStar = CalculateKernelVector(_X, x);

        // Solve K^(-1) * k*
        var KInvKStar = MatrixSolutionHelper.SolveLinearSystem(_K, kStar, _decompositionType);

        // Predictive mean: k*^T * K^(-1) * m_q
        // But m_q is already K * alpha for Gaussian, so we use a different formulation
        T mean = _numOps.Zero;
        for (int i = 0; i < _variationalMean.Length; i++)
        {
            mean = _numOps.Add(mean, _numOps.Multiply(KInvKStar[i], _variationalMean[i]));
        }

        // Predictive variance: k** - k*^T * K^(-1) * k* + k*^T * K^(-1) * S * K^(-1) * k*
        T kStarStar = _kernel.Calculate(x, x);

        // Prior variance reduction
        T variance = kStarStar;
        for (int i = 0; i < kStar.Length; i++)
        {
            variance = _numOps.Subtract(variance, _numOps.Multiply(kStar[i], KInvKStar[i]));
        }

        // Add posterior variance contribution
        var LTKInvKStar = _variationalCovCholesky.Transpose().Multiply(KInvKStar);
        T posteriorVar = _numOps.Zero;
        for (int i = 0; i < LTKInvKStar.Length; i++)
        {
            posteriorVar = _numOps.Add(posteriorVar, _numOps.Multiply(LTKInvKStar[i], LTKInvKStar[i]));
        }
        variance = _numOps.Add(variance, posteriorVar);

        // Ensure non-negative variance
        if (_numOps.ToDouble(variance) < 0)
        {
            variance = _numOps.FromDouble(1e-10);
        }

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
    /// <b>For Beginners:</b> The ELBO measures how good our approximation is.
    ///
    /// ELBO = Expected log-likelihood - KL divergence
    ///
    /// - Higher ELBO = better approximation
    /// - Can be used to compare different models or hyperparameters
    /// </para>
    /// </remarks>
    public T ComputeELBO()
    {
        if (_X.IsEmpty)
            return _numOps.Zero;

        // Expected log-likelihood
        double expLogLik = 0.0;
        int n = _y.Length;

        switch (_likelihood)
        {
            case VGPLikelihood.Gaussian:
                double noiseVar = Math.Max(_noiseVariance, 1e-10);
                for (int i = 0; i < n; i++)
                {
                    double diff = _numOps.ToDouble(_y[i]) - _numOps.ToDouble(_variationalMean[i]);
                    expLogLik -= 0.5 * diff * diff / noiseVar;
                    expLogLik -= 0.5 * Math.Log(2 * Math.PI * noiseVar);
                }
                break;

            case VGPLikelihood.Bernoulli:
                for (int i = 0; i < n; i++)
                {
                    double m = _numOps.ToDouble(_variationalMean[i]);
                    double y = _numOps.ToDouble(_y[i]);
                    // log p(y|f) = y*log(σ(f)) + (1-y)*log(1-σ(f)) = y*f - log(1+exp(f))
                    // Use numerically stable form: log(1+exp(m)) = m + log(1+exp(-m)) for m > 0
                    double logSigmoid = m > 0
                        ? -Math.Log(1 + Math.Exp(-m))
                        : m - Math.Log(1 + Math.Exp(m));
                    expLogLik += y * m + logSigmoid - m; // Simplifies to: y*m - log(1+exp(m))
                }
                break;

            case VGPLikelihood.Poisson:
                for (int i = 0; i < n; i++)
                {
                    double m = _numOps.ToDouble(_variationalMean[i]);
                    double y = _numOps.ToDouble(_y[i]);
                    double expM = Math.Exp(Math.Min(m, 20));
                    // log p(y|f) = y*f - exp(f) - log(y!)
                    expLogLik += y * m - expM - LogFactorial((int)y);
                }
                break;
        }

        // KL divergence (approximate)
        // Note: S = L_S * L_S^T is computed implicitly in trace term below
        var KInvM = MatrixSolutionHelper.SolveLinearSystem(_K, _variationalMean, _decompositionType);

        double quadForm = 0.0;
        for (int i = 0; i < n; i++)
        {
            quadForm += _numOps.ToDouble(_variationalMean[i]) * _numOps.ToDouble(KInvM[i]);
        }

        // log|K| using Cholesky
        double logDetK = 0.0;
        for (int i = 0; i < _LK.Rows; i++)
        {
            logDetK += 2.0 * Math.Log(Math.Abs(_numOps.ToDouble(_LK[i, i])));
        }

        // log|S| using Cholesky
        double logDetS = 0.0;
        for (int i = 0; i < _variationalCovCholesky.Rows; i++)
        {
            logDetS += 2.0 * Math.Log(Math.Abs(_numOps.ToDouble(_variationalCovCholesky[i, i])));
        }

        // Trace term: tr(K^(-1) * S)
        // Compute using S = L_S * L_S^T, so tr(K^(-1) * S) = ||L_K^(-1) * L_S||_F^2
        // For efficiency, we approximate using diagonal dominance when K is well-conditioned
        double trace = 0.0;
        try
        {
            // Solve L_K * Y = L_S for Y, then trace = ||Y||_F^2
            for (int j = 0; j < _variationalCovCholesky.Columns; j++)
            {
                var colS = _variationalCovCholesky.GetColumn(j);
                var solved = MatrixSolutionHelper.SolveLinearSystem(_LK, colS, _decompositionType);
                for (int i = 0; i < solved.Length; i++)
                {
                    trace += _numOps.ToDouble(_numOps.Multiply(solved[i], solved[i]));
                }
            }
        }
        catch (Exception)
        {
            // Fallback: use identity approximation (S ≈ K means trace ≈ n)
            trace = n;
        }

        double kl = 0.5 * (trace + quadForm - n + logDetK - logDetS);
        double elbo = expLogLik - kl;

        return _numOps.FromDouble(elbo);
    }

    /// <summary>
    /// Computes log factorial for Poisson likelihood.
    /// </summary>
    private static double LogFactorial(int n)
    {
        if (n <= 1) return 0;
        double result = 0;
        for (int i = 2; i <= n; i++)
        {
            result += Math.Log(i);
        }
        return result;
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of points.
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
    /// Calculates kernel values between a matrix and a vector.
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
    /// Creates an identity matrix.
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
    /// Creates a scaled identity matrix.
    /// </summary>
    private Matrix<T> CreateScaledIdentityMatrix(int size, double scale)
    {
        var matrix = new Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            matrix[i, i] = _numOps.FromDouble(scale);
        }
        return matrix;
    }

    /// <summary>
    /// Adds jitter to diagonal for numerical stability.
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

/// <summary>
/// Specifies the likelihood function type for Variational Gaussian Process.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The likelihood function describes what type of observations
/// you have and how they relate to the underlying latent function.
/// </para>
/// </remarks>
public enum VGPLikelihood
{
    /// <summary>
    /// Gaussian likelihood for continuous regression with normally distributed noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your target variable is continuous
    /// (like temperature, price, height) and you assume measurement errors
    /// follow a bell curve (normal distribution).
    /// </para>
    /// </remarks>
    Gaussian,

    /// <summary>
    /// Bernoulli likelihood for binary classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your target is binary (yes/no, 0/1, true/false).
    /// For example: spam detection, disease diagnosis, pass/fail prediction.
    /// </para>
    /// </remarks>
    Bernoulli,

    /// <summary>
    /// Poisson likelihood for count data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your target is a count (0, 1, 2, 3, ...).
    /// For example: number of customers, defect counts, event frequencies.
    /// </para>
    /// </remarks>
    Poisson
}
