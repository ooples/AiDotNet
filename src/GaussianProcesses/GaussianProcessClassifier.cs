namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Gaussian Process Classifier using Laplace approximation for probabilistic classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Gaussian Process Classifier (GPC) is a powerful machine learning method
/// that not only classifies data but also tells you how confident it is about each prediction.
///
/// Imagine you're building a spam filter:
/// - A regular classifier might say: "This email is spam"
/// - A GP classifier says: "This email is 95% likely to be spam, and I'm quite confident about this"
///
/// How does it work?
/// 1. It learns a "latent function" - a hidden score for each point in your data space
/// 2. This score is passed through a sigmoid function to get a probability
/// 3. The Laplace approximation helps us handle the mathematical complexity
///
/// The Laplace Approximation:
/// - GP classification doesn't have a nice closed-form solution like GP regression
/// - The Laplace approximation finds the most likely values (the "mode") of the latent function
/// - It then approximates the posterior as a Gaussian centered at this mode
/// - This gives us uncertainty estimates even though the true posterior is non-Gaussian
///
/// When to use GP Classification:
/// - When you need probability estimates, not just class labels
/// - When you have small to medium-sized datasets (up to a few thousand points)
/// - When uncertainty quantification is important (medical diagnosis, risk assessment)
/// - When your decision boundary might be non-linear
///
/// Limitations:
/// - Scales cubically O(n³) with dataset size due to matrix operations
/// - For larger datasets, consider Sparse GP Classification (SVGP)
/// </para>
/// </remarks>
public class GaussianProcessClassifier<T> : IGaussianProcessClassifier<T>
{
    /// <summary>
    /// The kernel function that determines how similarity between data points is calculated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel function is like a "similarity measure" that tells the model
    /// how related two data points are. Points that are similar according to the kernel will tend
    /// to have similar class labels.
    /// </para>
    /// </remarks>
    private IKernelFunction<T> _kernel;

    /// <summary>
    /// The matrix of input features from the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This stores all your training examples. Each row is one example,
    /// and each column is one feature (measurement) about that example.
    /// </para>
    /// </remarks>
    private Matrix<T> _X;

    /// <summary>
    /// The vector of class labels from the training data, transformed to +1/-1 for binary classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the "answers" for your training data. For binary classification,
    /// we internally convert labels 0/1 to -1/+1 because this makes the math work better with the
    /// sigmoid (logistic) function.
    /// </para>
    /// </remarks>
    private Vector<T> _y;

    /// <summary>
    /// The original class labels from training (before transformation).
    /// </summary>
    private Vector<T> _originalLabels;

    /// <summary>
    /// The kernel matrix calculated from the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a square matrix where each entry [i,j] tells us how similar
    /// training point i is to training point j. Think of it as a "similarity table" for all
    /// pairs of training examples.
    /// </para>
    /// </remarks>
    private Matrix<T> _K;

    /// <summary>
    /// The latent function values at training points (the mode of the posterior).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the "hidden scores" the model learns for each training point.
    /// A positive score means the model thinks that point belongs to class 1, negative means class 0.
    /// The magnitude indicates confidence - larger absolute values mean stronger predictions.
    /// </para>
    /// </remarks>
    private Vector<T> _f;

    /// <summary>
    /// The Hessian of the negative log-likelihood, used in the Laplace approximation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This matrix captures how "curved" the likelihood function is around
    /// the optimal latent values. It's used to approximate the posterior distribution and
    /// calculate uncertainties. Higher values on the diagonal indicate more certainty at those points.
    /// </para>
    /// </remarks>
    private Matrix<T> _W;

    /// <summary>
    /// Operations for performing numeric calculations with the generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The method used to decompose matrices for solving linear systems.
    /// </summary>
    private readonly MatrixDecompositionType _decompositionType;

    /// <summary>
    /// The cached log marginal likelihood from the last fit.
    /// </summary>
    private T _logMarginalLikelihood;

    /// <summary>
    /// Maximum number of iterations for the Newton-Raphson optimization.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance for the Newton-Raphson optimization.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// The number of classes detected during training.
    /// </summary>
    private int _numClasses;

    /// <inheritdoc/>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Initializes a new instance of the GaussianProcessClassifier class.
    /// </summary>
    /// <param name="kernel">The kernel function to use for measuring similarity between data points.</param>
    /// <param name="decompositionType">The matrix decomposition method to use for calculations.</param>
    /// <param name="maxIterations">Maximum iterations for Newton-Raphson optimization (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance for optimization (default: 1e-6).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up your GP classifier with the necessary components.
    ///
    /// The kernel function is the most important choice - it defines how the model sees similarity:
    /// - RBF/Gaussian kernel: Creates smooth decision boundaries (most common choice)
    /// - Matern kernel: Adjustable smoothness (good for real-world messy data)
    /// - Linear kernel: Creates linear decision boundaries (like logistic regression)
    ///
    /// The other parameters control the optimization process:
    /// - maxIterations: How many times to refine the solution (100 is usually plenty)
    /// - tolerance: When to stop refining (smaller = more precise but slower)
    ///
    /// For most uses, the defaults work well. Just pick an appropriate kernel for your data.
    /// </para>
    /// </remarks>
    public GaussianProcessClassifier(
        IKernelFunction<T> kernel,
        MatrixDecompositionType decompositionType = MatrixDecompositionType.Cholesky,
        int maxIterations = 100,
        double tolerance = 1e-6)
    {
        _kernel = kernel;
        _X = Matrix<T>.Empty();
        _y = Vector<T>.Empty();
        _originalLabels = Vector<T>.Empty();
        _K = Matrix<T>.Empty();
        _f = Vector<T>.Empty();
        _W = Matrix<T>.Empty();
        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionType = decompositionType;
        _logMarginalLikelihood = _numOps.Zero;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _numClasses = 0;
    }

    /// <summary>
    /// Trains the Gaussian Process classifier on the provided data using Laplace approximation.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a data point and each column is a feature.</param>
    /// <param name="y">The target class labels. For binary classification, should be 0 or 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the classifier using your training data.
    ///
    /// The training process uses the Laplace Approximation, which works in these steps:
    ///
    /// 1. Initialize latent function values (f) to zero
    /// 2. Use Newton-Raphson optimization to find the best f values:
    ///    - These are the values that maximize the posterior probability
    ///    - We iterate until the values stop changing significantly
    /// 3. Calculate the Hessian matrix (W) at the solution:
    ///    - This captures how curved the likelihood is
    ///    - Used later to estimate uncertainties
    /// 4. Compute the log marginal likelihood:
    ///    - This measures how well the model fits the data
    ///    - Useful for comparing different kernel configurations
    ///
    /// The Newton-Raphson algorithm is like finding the bottom of a valley:
    /// - Start somewhere (f = 0)
    /// - Look at the slope and curvature
    /// - Take a step toward the bottom
    /// - Repeat until you can't go any lower
    ///
    /// After training, the model stores everything it needs to make predictions on new data.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X;
        _originalLabels = y;

        // Determine number of classes
        var uniqueLabels = new HashSet<double>();
        for (int i = 0; i < y.Length; i++)
        {
            uniqueLabels.Add(_numOps.ToDouble(y[i]));
        }
        _numClasses = uniqueLabels.Count;

        // Transform labels from {0, 1} to {-1, +1} for binary classification
        // This makes the math work better with the logistic sigmoid
        _y = TransformLabels(y);

        // Calculate kernel matrix with jitter for numerical stability
        _K = CalculateKernelMatrix(X, X);
        AddJitter(_K);

        // Initialize latent function values to zero
        _f = new Vector<T>(X.Rows);
        for (int i = 0; i < _f.Length; i++)
        {
            _f[i] = _numOps.Zero;
        }

        // Newton-Raphson optimization to find the mode of the posterior
        // This is the core of the Laplace approximation
        OptimizeLatentFunction();

        // Calculate the log marginal likelihood
        _logMarginalLikelihood = CalculateLogMarginalLikelihood();
    }

    /// <summary>
    /// Transforms class labels from {0, 1} to {-1, +1} format.
    /// </summary>
    /// <param name="y">The original labels (0 or 1).</param>
    /// <returns>Transformed labels (-1 or +1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> We transform the labels because the logistic sigmoid function σ(x)
    /// has nice mathematical properties when labels are -1 and +1:
    /// - σ(f) gives P(y=+1|f)
    /// - σ(-f) gives P(y=-1|f)
    ///
    /// This symmetry makes the gradient and Hessian calculations cleaner.
    /// After predictions, we convert back to 0/1 for the final output.
    /// </para>
    /// </remarks>
    private Vector<T> TransformLabels(Vector<T> y)
    {
        var transformed = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            // Convert 0 -> -1, 1 -> +1 (use threshold comparison to avoid floating point equality)
            double label = _numOps.ToDouble(y[i]);
            transformed[i] = _numOps.FromDouble(label < 0.5 ? -1.0 : 1.0);
        }
        return transformed;
    }

    /// <summary>
    /// Adds a small jitter term to the diagonal for numerical stability.
    /// </summary>
    /// <param name="K">The kernel matrix to modify.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When data points are very similar, the kernel matrix can become
    /// "ill-conditioned" (nearly singular), causing numerical problems. Adding a tiny amount
    /// to the diagonal (called "jitter") prevents this without significantly affecting results.
    ///
    /// Think of it like adding a tiny bit of noise to prevent exact duplicates from
    /// causing mathematical issues.
    /// </para>
    /// </remarks>
    private void AddJitter(Matrix<T> K)
    {
        var jitter = _numOps.FromDouble(1e-6);
        for (int i = 0; i < K.Rows; i++)
        {
            K[i, i] = _numOps.Add(K[i, i], jitter);
        }
    }

    /// <summary>
    /// Performs Newton-Raphson optimization to find the mode of the posterior distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Newton-Raphson is an iterative optimization algorithm that finds
    /// where a function reaches its maximum (or minimum).
    ///
    /// At each iteration:
    /// 1. Calculate the gradient (which direction is "uphill")
    /// 2. Calculate the Hessian (how curved is the surface)
    /// 3. Use both to compute an optimal step
    /// 4. Take the step and repeat
    ///
    /// For GP classification, we're finding the latent function values (f) that maximize:
    ///    log p(y|f) + log p(f|X)
    ///
    /// Where:
    /// - log p(y|f): How well the current f values explain the class labels
    /// - log p(f|X): How consistent f is with our prior (the GP prior from the kernel)
    ///
    /// The algorithm stops when:
    /// - The change in f becomes very small (converged)
    /// - Maximum iterations reached (might need more iterations or different hyperparameters)
    /// </para>
    /// </remarks>
    private void OptimizeLatentFunction()
    {
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Calculate pi = sigmoid(f) for each training point
            var pi = CalculateSigmoid(_f);

            // W = pi * (1 - pi) (diagonal of the Hessian of negative log-likelihood)
            // This measures the "curvature" of the likelihood at each point
            _W = CalculateWMatrix(pi);

            // Calculate gradient of log-likelihood: grad = y_transformed - pi_adjusted
            // Where pi_adjusted maps -1/+1 predictions to match label format
            var grad = CalculateGradient(pi);

            // Calculate B = I + W^(1/2) * K * W^(1/2)
            // This combines the prior (K) with the likelihood curvature (W)
            var sqrtW = CalculateSqrtW(_W);
            var B = CalculateBMatrix(sqrtW);

            // Solve for the Newton direction
            // b = W * f + grad
            var Wf = _W.Multiply(_f);
            var b = new Vector<T>(_f.Length);
            for (int i = 0; i < b.Length; i++)
            {
                b[i] = _numOps.Add(Wf[i], grad[i]);
            }

            // a = b - W^(1/2) * B^(-1) * W^(1/2) * K * b
            var Kb = _K.Multiply(b);
            var sqrtWKb = sqrtW.Multiply(Kb);

            // Solve B * temp = sqrtWKb
            var temp = MatrixSolutionHelper.SolveLinearSystem(B, sqrtWKb, _decompositionType);

            var sqrtWtemp = sqrtW.Multiply(temp);
            var a = new Vector<T>(b.Length);
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = _numOps.Subtract(b[i], sqrtWtemp[i]);
            }

            // Update f = K * a
            var fNew = _K.Multiply(a);

            // Check convergence
            T maxDiff = _numOps.Zero;
            for (int i = 0; i < _f.Length; i++)
            {
                T diff = _numOps.Abs(_numOps.Subtract(fNew[i], _f[i]));
                if (_numOps.ToDouble(diff) > _numOps.ToDouble(maxDiff))
                {
                    maxDiff = diff;
                }
                _f[i] = fNew[i];
            }

            if (_numOps.ToDouble(maxDiff) < _tolerance)
            {
                break;
            }
        }
    }

    /// <summary>
    /// Calculates the logistic sigmoid function for a vector of values.
    /// </summary>
    /// <param name="f">The latent function values.</param>
    /// <returns>Sigmoid values (probabilities) for each element.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sigmoid function σ(x) = 1/(1 + exp(-x)) squashes any number
    /// to a value between 0 and 1, making it perfect for probabilities.
    ///
    /// - Large positive input → output close to 1
    /// - Large negative input → output close to 0
    /// - Input of 0 → output of 0.5 (maximum uncertainty)
    ///
    /// This function converts the latent scores (which can be any number) into probabilities
    /// that we can interpret as "confidence in class 1".
    /// </para>
    /// </remarks>
    private Vector<T> CalculateSigmoid(Vector<T> f)
    {
        var pi = new Vector<T>(f.Length);
        for (int i = 0; i < f.Length; i++)
        {
            double fVal = _numOps.ToDouble(f[i]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-fVal));
            pi[i] = _numOps.FromDouble(sigmoid);
        }
        return pi;
    }

    /// <summary>
    /// Calculates the W matrix (diagonal of the Hessian of negative log-likelihood).
    /// </summary>
    /// <param name="pi">The sigmoid probabilities at each training point.</param>
    /// <returns>The W matrix as a diagonal matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The W matrix captures how "certain" the model is at each training point.
    ///
    /// W[i] = pi[i] * (1 - pi[i])
    ///
    /// This is the variance of a Bernoulli distribution:
    /// - When pi = 0.5, W = 0.25 (maximum uncertainty)
    /// - When pi approaches 0 or 1, W approaches 0 (high certainty)
    ///
    /// Points with high uncertainty (W close to 0.25) have more influence on shaping the
    /// decision boundary, while certain points contribute less.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateWMatrix(Vector<T> pi)
    {
        var W = new Matrix<T>(pi.Length, pi.Length);
        for (int i = 0; i < pi.Length; i++)
        {
            T piVal = pi[i];
            T oneMinusPi = _numOps.Subtract(_numOps.One, piVal);
            W[i, i] = _numOps.Multiply(piVal, oneMinusPi);
        }
        return W;
    }

    /// <summary>
    /// Calculates the gradient of the log-likelihood with respect to f.
    /// </summary>
    /// <param name="pi">The sigmoid probabilities at each training point.</param>
    /// <returns>The gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient tells us which direction to move the latent values (f)
    /// to improve the likelihood of the observed labels.
    ///
    /// For the logistic likelihood with {-1, +1} labels:
    ///    gradient[i] = (y[i] + 1)/2 - pi[i]
    ///
    /// This simplifies to:
    /// - If y[i] = +1: gradient[i] = 1 - pi[i] (want to increase f to push pi toward 1)
    /// - If y[i] = -1: gradient[i] = 0 - pi[i] = -pi[i] (want to decrease f to push pi toward 0)
    ///
    /// The gradient is zero when the predictions perfectly match the labels.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateGradient(Vector<T> pi)
    {
        var grad = new Vector<T>(pi.Length);
        for (int i = 0; i < pi.Length; i++)
        {
            // Convert y from {-1, +1} to {0, 1} for gradient calculation
            double yVal = _numOps.ToDouble(_y[i]);
            double target = (yVal + 1.0) / 2.0; // -1 -> 0, +1 -> 1
            double piVal = _numOps.ToDouble(pi[i]);
            grad[i] = _numOps.FromDouble(target - piVal);
        }
        return grad;
    }

    /// <summary>
    /// Calculates the square root of the diagonal W matrix.
    /// </summary>
    /// <param name="W">The W matrix.</param>
    /// <returns>The square root of W (element-wise on diagonal).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since W is a diagonal matrix, its square root is simply the
    /// square root of each diagonal element. This is used in the Newton-Raphson update
    /// to efficiently compute matrix products.
    ///
    /// W^(1/2) * K * W^(1/2) is a symmetric positive definite matrix that we can
    /// factor efficiently using Cholesky decomposition.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateSqrtW(Matrix<T> W)
    {
        var sqrtW = new Matrix<T>(W.Rows, W.Columns);
        for (int i = 0; i < W.Rows; i++)
        {
            double diagVal = _numOps.ToDouble(W[i, i]);
            sqrtW[i, i] = _numOps.FromDouble(Math.Sqrt(Math.Max(diagVal, 0)));
        }
        return sqrtW;
    }

    /// <summary>
    /// Calculates the B matrix used in the Newton-Raphson update.
    /// </summary>
    /// <param name="sqrtW">The square root of the W matrix.</param>
    /// <returns>B = I + W^(1/2) * K * W^(1/2).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The B matrix combines information from:
    /// - The identity matrix I (prior assumption of independence)
    /// - The kernel matrix K (correlations between training points)
    /// - The W matrix (likelihood curvature at current solution)
    ///
    /// B captures how the prior and likelihood interact, and solving linear systems
    /// with B is the key step in computing the Newton update direction.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateBMatrix(Matrix<T> sqrtW)
    {
        // B = I + sqrtW * K * sqrtW
        var sqrtWK = sqrtW.Multiply(_K);
        var sqrtWKsqrtW = sqrtWK.Multiply(sqrtW);

        var B = new Matrix<T>(sqrtW.Rows, sqrtW.Columns);
        for (int i = 0; i < B.Rows; i++)
        {
            for (int j = 0; j < B.Columns; j++)
            {
                B[i, j] = i == j
                    ? _numOps.Add(_numOps.One, sqrtWKsqrtW[i, j])
                    : sqrtWKsqrtW[i, j];
            }
        }
        return B;
    }

    /// <summary>
    /// Calculates the log marginal likelihood of the model.
    /// </summary>
    /// <returns>The approximate log marginal likelihood using Laplace approximation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The log marginal likelihood (LML) measures how well the model
    /// explains the observed data, while accounting for model complexity.
    ///
    /// Higher LML (less negative) = better model fit
    ///
    /// The LML has three components:
    /// 1. Log-likelihood: How well the predictions match the labels
    /// 2. Prior term: Penalty for latent values that deviate from zero
    /// 3. Complexity term: Penalty based on the curvature of the posterior
    ///
    /// This is useful for:
    /// - Comparing different kernel configurations
    /// - Hyperparameter optimization (find parameters that maximize LML)
    /// - Model selection (choosing between different GP classifiers)
    ///
    /// The Laplace approximation gives us an analytical expression for the LML,
    /// which would otherwise require intractable integration.
    /// </para>
    /// </remarks>
    private T CalculateLogMarginalLikelihood()
    {
        // Log-likelihood term: sum of log p(y|f)
        double logLik = 0.0;
        for (int i = 0; i < _y.Length; i++)
        {
            double yVal = _numOps.ToDouble(_y[i]);
            double fVal = _numOps.ToDouble(_f[i]);
            // log p(y|f) = log σ(y*f) for {-1,+1} labels
            double logProb = -Math.Log(1.0 + Math.Exp(-yVal * fVal));
            logLik += logProb;
        }

        // Prior term: -0.5 * f^T * K^(-1) * f
        var Kinv_f = MatrixSolutionHelper.SolveLinearSystem(_K, _f, _decompositionType);
        double priorTerm = 0.0;
        for (int i = 0; i < _f.Length; i++)
        {
            priorTerm += _numOps.ToDouble(_f[i]) * _numOps.ToDouble(Kinv_f[i]);
        }
        priorTerm *= -0.5;

        // Complexity term: -0.5 * log|B|
        var sqrtW = CalculateSqrtW(_W);
        var B = CalculateBMatrix(sqrtW);
        double logDetB = 0.0;
        try
        {
            var choleskyB = new CholeskyDecomposition<T>(B);
            // log|B| = 2 * sum(log(diag(L))) where B = L*L^T
            for (int i = 0; i < choleskyB.L.Rows; i++)
            {
                logDetB += 2.0 * Math.Log(Math.Abs(_numOps.ToDouble(choleskyB.L[i, i])));
            }
        }
        catch (ArgumentException ex)
        {
            // If Cholesky fails, use a fallback estimate (matrix may not be positive definite)
            System.Diagnostics.Debug.WriteLine($"Cholesky decomposition failed in log marginal likelihood: {ex.Message}");
            logDetB = 0.0;
        }

        double lml = logLik + priorTerm - 0.5 * logDetB;
        return _numOps.FromDouble(lml);
    }

    /// <inheritdoc/>
    public T GetLogMarginalLikelihood()
    {
        return _logMarginalLikelihood;
    }

    /// <inheritdoc/>
    public (int predictedClass, T probability, T variance) Predict(Vector<T> x)
    {
        if (_X.IsEmpty || _f.IsEmpty)
        {
            throw new InvalidOperationException("Model must be trained before prediction. Call Fit() first.");
        }

        // Calculate kernel vector between test point and training points
        var kStar = CalculateKernelVector(_X, x);

        // Predictive mean: k* . α where α = K^(-1) * f
        var alpha = MatrixSolutionHelper.SolveLinearSystem(_K, _f, _decompositionType);
        T fMean = kStar.DotProduct(alpha);

        // Predictive variance: k** - k*^T * (K + W^(-1))^(-1) * k*
        T kStarStar = _kernel.Calculate(x, x);

        // Compute (K + W^(-1))^(-1) * k* efficiently
        var sqrtW = CalculateSqrtW(_W);
        var B = CalculateBMatrix(sqrtW);
        var sqrtWkStar = sqrtW.Multiply(kStar);
        var v = MatrixSolutionHelper.SolveLinearSystem(B, sqrtWkStar, _decompositionType);
        var sqrtWv = sqrtW.Multiply(v);

        T variance = kStarStar;
        for (int i = 0; i < kStar.Length; i++)
        {
            variance = _numOps.Subtract(variance, _numOps.Multiply(kStar[i], sqrtWv[i]));
        }

        // Convert latent mean to probability using probit approximation
        // This integrates out the uncertainty in f to get p(y=1|x)
        double fMeanVal = _numOps.ToDouble(fMean);
        double varVal = Math.Max(_numOps.ToDouble(variance), 1e-10);

        // Probit approximation: use sigmoid with scaled input
        // κ = 1 / sqrt(1 + π * var / 8)
        double kappa = 1.0 / Math.Sqrt(1.0 + Math.PI * varVal / 8.0);
        double probClass1 = 1.0 / (1.0 + Math.Exp(-kappa * fMeanVal));

        int predictedClass = probClass1 >= 0.5 ? 1 : 0;
        T probability = _numOps.FromDouble(predictedClass == 1 ? probClass1 : 1.0 - probClass1);

        return (predictedClass, probability, variance);
    }

    /// <inheritdoc/>
    public Matrix<T> PredictProbabilities(Matrix<T> X)
    {
        if (_numClasses != 2)
        {
            throw new NotSupportedException(
                $"PredictProbabilities currently supports binary classification only. " +
                $"Found {_numClasses} classes. For multi-class problems, consider One-vs-Rest or One-vs-One strategies.");
        }

        int nSamples = X.Rows;
        // Binary classification: always 2 columns (class 0 and class 1)
        var probabilities = new Matrix<T>(nSamples, 2);

        for (int i = 0; i < nSamples; i++)
        {
            // Compute probabilities directly from latent function
            var kStar = CalculateKernelVector(_X, X.GetRow(i));
            var alpha = MatrixSolutionHelper.SolveLinearSystem(_K, _f, _decompositionType);
            T fMean = kStar.DotProduct(alpha);

            T kStarStar = _kernel.Calculate(X.GetRow(i), X.GetRow(i));
            var sqrtW = CalculateSqrtW(_W);
            var B = CalculateBMatrix(sqrtW);
            var sqrtWkStar = sqrtW.Multiply(kStar);
            var v = MatrixSolutionHelper.SolveLinearSystem(B, sqrtWkStar, _decompositionType);
            var sqrtWv = sqrtW.Multiply(v);

            T variance = kStarStar;
            for (int j = 0; j < kStar.Length; j++)
            {
                variance = _numOps.Subtract(variance, _numOps.Multiply(kStar[j], sqrtWv[j]));
            }

            double fMeanVal = _numOps.ToDouble(fMean);
            double varVal = Math.Max(_numOps.ToDouble(variance), 1e-10);
            double kappa = 1.0 / Math.Sqrt(1.0 + Math.PI * varVal / 8.0);
            double p1 = 1.0 / (1.0 + Math.Exp(-kappa * fMeanVal));

            probabilities[i, 0] = _numOps.FromDouble(1.0 - p1);
            probabilities[i, 1] = _numOps.FromDouble(p1);
        }

        return probabilities;
    }

    /// <inheritdoc/>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        _kernel = kernel;
        if (!_X.IsEmpty && !_y.IsEmpty)
        {
            Fit(_X, _originalLabels);
        }
    }

    /// <summary>
    /// Calculates the kernel matrix between two sets of data points.
    /// </summary>
    /// <param name="X1">The first set of data points.</param>
    /// <param name="X2">The second set of data points.</param>
    /// <returns>A matrix where each element [i,j] represents the kernel value between the i-th point in X1 and the j-th point in X2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a "similarity table" between data points.
    ///
    /// The kernel matrix K where K[i,j] = kernel(X1[i], X2[j]) tells us how similar each
    /// pair of points is according to our chosen kernel function.
    ///
    /// For example, with an RBF kernel:
    /// - Nearby points have K[i,j] close to 1
    /// - Distant points have K[i,j] close to 0
    ///
    /// This similarity information is fundamental to how GPs work - they assume that
    /// similar inputs should produce similar outputs.
    /// </para>
    /// </remarks>
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
    /// Calculates the kernel values between a set of data points and a single point.
    /// </summary>
    /// <param name="X">A matrix where each row is a data point.</param>
    /// <param name="x">A single data point as a vector.</param>
    /// <returns>A vector where each element is the kernel value between the corresponding row in X and the point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When making predictions for a new point, we need to know how similar
    /// it is to each of our training points. This method computes those similarities.
    ///
    /// The resulting vector tells us which training points should have the most influence
    /// on our prediction for the new point:
    /// - High similarity = strong influence
    /// - Low similarity = weak influence
    ///
    /// This is how GPs "remember" the training data and use it for predictions.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateKernelVector(Matrix<T> X, Vector<T> x)
    {
        var k = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            k[i] = _kernel.Calculate(X.GetRow(i), x);
        }
        return k;
    }
}
