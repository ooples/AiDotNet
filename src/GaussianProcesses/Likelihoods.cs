namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Interface for likelihood functions in Gaussian Processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A likelihood function describes how observed data y relates to
/// the underlying GP function value f. It models the "noise" or observation process.
///
/// Common likelihoods:
/// - Gaussian: y = f + ε, where ε ~ N(0, σ²) - for regression
/// - Bernoulli: y ~ Bernoulli(sigmoid(f)) - for binary classification
/// - Poisson: y ~ Poisson(exp(f)) - for count data
///
/// The likelihood affects:
/// - How we interpret the GP output
/// - What inference method we need (exact vs approximate)
/// - What kind of predictions we can make
/// </para>
/// </remarks>
public interface ILikelihood<T>
{
    /// <summary>
    /// Gets the name of this likelihood.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Computes the log-likelihood of observations given latent function values.
    /// </summary>
    /// <param name="y">The observed values.</param>
    /// <param name="f">The latent function values.</param>
    /// <returns>The log-likelihood.</returns>
    T LogLikelihood(Vector<T> y, Vector<T> f);

    /// <summary>
    /// Computes the gradient of the log-likelihood with respect to f.
    /// </summary>
    /// <param name="y">The observed values.</param>
    /// <param name="f">The latent function values.</param>
    /// <returns>The gradient vector.</returns>
    Vector<T> LogLikelihoodGradient(Vector<T> y, Vector<T> f);

    /// <summary>
    /// Computes the Hessian diagonal of the log-likelihood with respect to f.
    /// </summary>
    /// <param name="y">The observed values.</param>
    /// <param name="f">The latent function values.</param>
    /// <returns>The diagonal of the Hessian (second derivatives).</returns>
    Vector<T> LogLikelihoodHessianDiag(Vector<T> y, Vector<T> f);

    /// <summary>
    /// Transforms latent function values to the expected observation value.
    /// </summary>
    /// <param name="f">The latent function value.</param>
    /// <returns>The expected observation.</returns>
    T TransformMean(T f);

    /// <summary>
    /// Computes the predictive variance given latent mean and variance.
    /// </summary>
    /// <param name="fMean">The latent mean.</param>
    /// <param name="fVariance">The latent variance.</param>
    /// <returns>The predictive variance.</returns>
    T PredictiveVariance(T fMean, T fVariance);
}

/// <summary>
/// Implements the Gaussian (Normal) likelihood for regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Gaussian likelihood is the standard choice for regression.
/// It assumes observations are the true function value plus Gaussian noise:
///
/// y = f(x) + ε, where ε ~ N(0, σ²)
///
/// This means:
/// - Errors are normally distributed
/// - Errors have constant variance (homoscedastic)
/// - Errors are independent
///
/// The noise variance σ² is a hyperparameter that:
/// - Large σ² → Smoother fit, more uncertainty
/// - Small σ² → Interpolates data more closely
///
/// Gaussian likelihood allows exact GP inference (no approximations needed).
/// </para>
/// </remarks>
public class GaussianLikelihood<T> : ILikelihood<T>
{
    private readonly T _noiseVariance;
    private readonly INumericOperations<T> _numOps;
    private readonly double _log2Pi = Math.Log(2.0 * Math.PI);

    /// <summary>
    /// Initializes a new Gaussian likelihood.
    /// </summary>
    /// <param name="noiseVariance">The observation noise variance σ².</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Gaussian likelihood with specified noise level.
    ///
    /// Choosing noise variance:
    /// - Start with a small value like 0.01 or 0.1
    /// - Optimize it along with other hyperparameters
    /// - Larger values if you have noisy data
    ///
    /// Example:
    /// var likelihood = new GaussianLikelihood&lt;double&gt;(0.1);
    /// </para>
    /// </remarks>
    public GaussianLikelihood(double noiseVariance = 0.1)
    {
        if (noiseVariance <= 0)
            throw new ArgumentException("Noise variance must be positive.", nameof(noiseVariance));

        _numOps = MathHelper.GetNumericOperations<T>();
        _noiseVariance = _numOps.FromDouble(noiseVariance);
    }

    /// <summary>
    /// Gets the name of this likelihood.
    /// </summary>
    public string Name => "Gaussian";

    /// <summary>
    /// Gets the noise variance.
    /// </summary>
    public T NoiseVariance => _noiseVariance;

    /// <summary>
    /// Computes the log-likelihood of observations given latent function values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how likely the observations are given
    /// the function values. For Gaussian likelihood:
    ///
    /// log p(y|f) = -0.5 × Σᵢ [(yᵢ - fᵢ)²/σ² + log(2πσ²)]
    ///
    /// Higher values mean better fit (less negative = better).
    /// </para>
    /// </remarks>
    public T LogLikelihood(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        double variance = _numOps.ToDouble(_noiseVariance);
        double logNorm = -0.5 * (_log2Pi + Math.Log(variance));

        double total = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double diff = _numOps.ToDouble(y[i]) - _numOps.ToDouble(f[i]);
            total += logNorm - 0.5 * diff * diff / variance;
        }

        return _numOps.FromDouble(total);
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood with respect to f.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient tells us how to adjust f to increase
    /// the likelihood. For Gaussian likelihood:
    ///
    /// ∂log p(y|f)/∂fᵢ = (yᵢ - fᵢ)/σ²
    ///
    /// If y > f, gradient is positive (increase f).
    /// If y &lt; f, gradient is negative (decrease f).
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodGradient(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        double variance = _numOps.ToDouble(_noiseVariance);
        var grad = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            double diff = _numOps.ToDouble(y[i]) - _numOps.ToDouble(f[i]);
            grad[i] = _numOps.FromDouble(diff / variance);
        }

        return grad;
    }

    /// <summary>
    /// Computes the Hessian diagonal of the log-likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hessian tells us about the curvature of the
    /// likelihood. For Gaussian likelihood, it's constant:
    ///
    /// ∂²log p(y|f)/∂fᵢ² = -1/σ²
    ///
    /// This constant Hessian is why Gaussian likelihood is "nice" - it leads
    /// to exact GP inference with closed-form solutions.
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodHessianDiag(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));

        double variance = _numOps.ToDouble(_noiseVariance);
        var hess = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            hess[i] = _numOps.FromDouble(-1.0 / variance);
        }

        return hess;
    }

    /// <summary>
    /// For Gaussian likelihood, the mean is just f (identity transform).
    /// </summary>
    public T TransformMean(T f) => f;

    /// <summary>
    /// Computes predictive variance (adds noise variance to latent variance).
    /// </summary>
    public T PredictiveVariance(T fMean, T fVariance)
    {
        return _numOps.Add(fVariance, _noiseVariance);
    }
}

/// <summary>
/// Implements the Bernoulli likelihood for binary classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Bernoulli likelihood is used for binary classification
/// (yes/no, 0/1, positive/negative outcomes).
///
/// The latent function f is passed through a sigmoid to get probability:
/// p(y=1|f) = σ(f) = 1 / (1 + exp(-f))
///
/// This means:
/// - f → +∞ gives p(y=1) → 1
/// - f → -∞ gives p(y=1) → 0
/// - f = 0 gives p(y=1) = 0.5
///
/// The GP models uncertainty in f, which translates to uncertainty in class probabilities.
///
/// Note: Bernoulli likelihood requires approximate inference (Laplace or EP).
/// </para>
/// </remarks>
public class BernoulliLikelihood<T> : ILikelihood<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Bernoulli likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Bernoulli likelihood for binary classification.
    /// Labels should be 0 or 1 (or can be -1/+1, which will be converted).
    /// </para>
    /// </remarks>
    public BernoulliLikelihood()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the name of this likelihood.
    /// </summary>
    public string Name => "Bernoulli";

    /// <summary>
    /// Computes the log-likelihood of observations given latent function values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For binary classification:
    ///
    /// log p(y|f) = Σᵢ [yᵢ×log(σ(fᵢ)) + (1-yᵢ)×log(1-σ(fᵢ))]
    ///            = Σᵢ [yᵢ×fᵢ - log(1 + exp(fᵢ))]  (numerically stable form)
    ///
    /// Labels y should be 0 or 1.
    /// </para>
    /// </remarks>
    public T LogLikelihood(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        double total = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = _numOps.ToDouble(f[i]);

            // Convert -1/+1 labels to 0/1 if needed
            if (yi < 0) yi = 0;

            // Numerically stable log-likelihood
            // log p(y|f) = y*f - log(1 + exp(f))
            total += yi * fi - LogOnePlusExp(fi);
        }

        return _numOps.FromDouble(total);
    }

    /// <summary>
    /// Computes log(1 + exp(x)) in a numerically stable way.
    /// </summary>
    private static double LogOnePlusExp(double x)
    {
        if (x > 35) return x;
        if (x < -35) return 0;
        return Math.Log(1 + Math.Exp(x));
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood with respect to f.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient for Bernoulli likelihood:
    ///
    /// ∂log p(y|f)/∂fᵢ = yᵢ - σ(fᵢ) = yᵢ - p(y=1|fᵢ)
    ///
    /// If y=1 and σ(f)&lt;1, gradient is positive (increase f to increase p(y=1))
    /// If y=0 and σ(f)>0, gradient is negative (decrease f to decrease p(y=1))
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodGradient(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        var grad = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = _numOps.ToDouble(f[i]);

            if (yi < 0) yi = 0; // Convert -1/+1 to 0/1

            double sigma = Sigmoid(fi);
            grad[i] = _numOps.FromDouble(yi - sigma);
        }

        return grad;
    }

    /// <summary>
    /// Computes the Hessian diagonal of the log-likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hessian for Bernoulli likelihood:
    ///
    /// ∂²log p(y|f)/∂fᵢ² = -σ(fᵢ)×(1-σ(fᵢ)) = -p×(1-p)
    ///
    /// This is always negative (log-likelihood is concave).
    /// Maximum magnitude at f=0 (most uncertain), decreases as |f| increases.
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodHessianDiag(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));

        var hess = new Vector<T>(f.Length);

        for (int i = 0; i < f.Length; i++)
        {
            double fi = _numOps.ToDouble(f[i]);
            double sigma = Sigmoid(fi);
            hess[i] = _numOps.FromDouble(-sigma * (1 - sigma));
        }

        return hess;
    }

    /// <summary>
    /// Transforms latent function value to class probability using sigmoid.
    /// </summary>
    public T TransformMean(T f)
    {
        double fVal = _numOps.ToDouble(f);
        return _numOps.FromDouble(Sigmoid(fVal));
    }

    /// <summary>
    /// Computes predictive variance for classification probabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For classification, the predictive variance accounts for
    /// uncertainty in both the latent function and the observation process.
    /// Uses the probit approximation for computational tractability.
    /// </para>
    /// </remarks>
    public T PredictiveVariance(T fMean, T fVariance)
    {
        double mu = _numOps.ToDouble(fMean);
        double v = _numOps.ToDouble(fVariance);

        // Probit approximation: scale mean by sqrt(1 + c²*variance)
        double c = Math.PI / 8.0;
        double scaleFactor = 1.0 / Math.Sqrt(1 + c * v);
        double p = Sigmoid(mu * scaleFactor);

        // Variance of Bernoulli is p*(1-p)
        return _numOps.FromDouble(p * (1 - p));
    }

    /// <summary>
    /// Numerically stable sigmoid function.
    /// </summary>
    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            double z = Math.Exp(-x);
            return 1.0 / (1.0 + z);
        }
        else
        {
            double z = Math.Exp(x);
            return z / (1.0 + z);
        }
    }
}

/// <summary>
/// Implements the Poisson likelihood for count data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Poisson likelihood is used for modeling count data
/// (number of events, customer arrivals, defects, etc.).
///
/// The latent function f is transformed to a rate using exp:
/// y ~ Poisson(λ), where λ = exp(f)
///
/// This means:
/// - f → +∞ gives high expected counts
/// - f → -∞ gives low expected counts (near 0)
/// - f = 0 gives expected count of 1
///
/// Properties of Poisson:
/// - E[y] = λ = exp(f)
/// - Var[y] = λ = exp(f) (variance equals mean)
///
/// Use when:
/// - Counting discrete events
/// - Events are independent
/// - Rate of events can vary smoothly over inputs
/// </para>
/// </remarks>
public class PoissonLikelihood<T> : ILikelihood<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Poisson likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Poisson likelihood for count data.
    /// Observations y should be non-negative integers (0, 1, 2, ...).
    /// </para>
    /// </remarks>
    public PoissonLikelihood()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the name of this likelihood.
    /// </summary>
    public string Name => "Poisson";

    /// <summary>
    /// Computes the log-likelihood of count observations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Poisson likelihood:
    ///
    /// log p(y|f) = Σᵢ [yᵢ×fᵢ - exp(fᵢ) - log(yᵢ!)]
    ///
    /// The log(y!) term is constant with respect to f so can be ignored for optimization.
    /// </para>
    /// </remarks>
    public T LogLikelihood(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        double total = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = _numOps.ToDouble(f[i]);

            // Clip f to prevent numerical overflow
            fi = Math.Min(fi, 20);

            double lambda = Math.Exp(fi);
            total += yi * fi - lambda - LogFactorial((int)Math.Round(yi));
        }

        return _numOps.FromDouble(total);
    }

    /// <summary>
    /// Computes log(n!) using Stirling's approximation for large n.
    /// </summary>
    private static double LogFactorial(int n)
    {
        if (n < 0) return 0;
        if (n <= 1) return 0;
        if (n < 20)
        {
            double result = 0;
            for (int i = 2; i <= n; i++)
            {
                result += Math.Log(i);
            }
            return result;
        }
        // Stirling's approximation
        return n * Math.Log(n) - n + 0.5 * Math.Log(2 * Math.PI * n);
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient for Poisson likelihood:
    ///
    /// ∂log p(y|f)/∂fᵢ = yᵢ - exp(fᵢ) = yᵢ - λᵢ
    ///
    /// If observed count > expected rate, gradient is positive (increase f).
    /// If observed count &lt; expected rate, gradient is negative (decrease f).
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodGradient(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        var grad = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            double yi = _numOps.ToDouble(y[i]);
            double fi = Math.Min(_numOps.ToDouble(f[i]), 20);
            double lambda = Math.Exp(fi);

            grad[i] = _numOps.FromDouble(yi - lambda);
        }

        return grad;
    }

    /// <summary>
    /// Computes the Hessian diagonal of the log-likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Hessian for Poisson likelihood:
    ///
    /// ∂²log p(y|f)/∂fᵢ² = -exp(fᵢ) = -λᵢ
    ///
    /// Always negative (concave), larger magnitude for higher rates.
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodHessianDiag(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));

        var hess = new Vector<T>(f.Length);

        for (int i = 0; i < f.Length; i++)
        {
            double fi = Math.Min(_numOps.ToDouble(f[i]), 20);
            double lambda = Math.Exp(fi);
            hess[i] = _numOps.FromDouble(-lambda);
        }

        return hess;
    }

    /// <summary>
    /// Transforms latent function value to expected count using exp.
    /// </summary>
    public T TransformMean(T f)
    {
        double fVal = Math.Min(_numOps.ToDouble(f), 20);
        return _numOps.FromDouble(Math.Exp(fVal));
    }

    /// <summary>
    /// Computes predictive variance for counts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Poisson, the variance equals the mean (λ).
    /// We also need to account for uncertainty in f.
    ///
    /// Using the delta method: Var[exp(f)] ≈ exp(2μ) × (exp(v) - 1)
    /// where μ and v are the mean and variance of f.
    /// </para>
    /// </remarks>
    public T PredictiveVariance(T fMean, T fVariance)
    {
        double mu = Math.Min(_numOps.ToDouble(fMean), 20);
        double v = _numOps.ToDouble(fVariance);

        // Expected lambda using log-normal moment
        double expectedLambda = Math.Exp(mu + 0.5 * v);

        // Variance of Poisson with uncertain rate
        // Var[Y] = E[λ] + Var[λ] = E[λ] + E[λ²] - E[λ]²
        double expectedLambdaSq = Math.Exp(2 * mu + 2 * v);
        double varLambda = expectedLambdaSq - expectedLambda * expectedLambda;

        return _numOps.FromDouble(expectedLambda + varLambda);
    }
}

/// <summary>
/// Implements the Student-t likelihood for robust regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Student-t likelihood provides robust regression that is
/// less sensitive to outliers than Gaussian likelihood.
///
/// y = f + ε, where ε ~ Student-t(0, σ², ν)
///
/// The degrees of freedom ν controls "heaviness" of tails:
/// - ν → ∞: Approaches Gaussian (not robust)
/// - ν = 4-5: Moderately robust
/// - ν = 1: Very robust (Cauchy distribution)
///
/// When to use:
/// - Data has outliers
/// - Errors might not be Gaussian
/// - You want predictions to not be pulled by extreme values
///
/// Note: Student-t likelihood requires approximate inference.
/// </para>
/// </remarks>
public class StudentTLikelihood<T> : ILikelihood<T>
{
    private readonly double _noiseScale;
    private readonly double _degreesOfFreedom;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Student-t likelihood.
    /// </summary>
    /// <param name="noiseScale">The noise scale parameter σ.</param>
    /// <param name="degreesOfFreedom">The degrees of freedom ν.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Student-t likelihood for robust regression.
    ///
    /// Choosing parameters:
    /// - noiseScale: Similar to standard deviation, start around 0.1-1.0
    /// - degreesOfFreedom: Start with 4-5 for moderate robustness
    ///   - Lower ν = more robust but harder to optimize
    ///   - Higher ν = approaches Gaussian
    ///
    /// Example:
    /// var likelihood = new StudentTLikelihood&lt;double&gt;(noiseScale: 0.5, degreesOfFreedom: 4);
    /// </para>
    /// </remarks>
    public StudentTLikelihood(double noiseScale = 0.5, double degreesOfFreedom = 4.0)
    {
        if (noiseScale <= 0)
            throw new ArgumentException("Noise scale must be positive.", nameof(noiseScale));
        if (degreesOfFreedom <= 0)
            throw new ArgumentException("Degrees of freedom must be positive.", nameof(degreesOfFreedom));

        _noiseScale = noiseScale;
        _degreesOfFreedom = degreesOfFreedom;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the name of this likelihood.
    /// </summary>
    public string Name => "StudentT";

    /// <summary>
    /// Gets the noise scale.
    /// </summary>
    public double NoiseScale => _noiseScale;

    /// <summary>
    /// Gets the degrees of freedom.
    /// </summary>
    public double DegreesOfFreedom => _degreesOfFreedom;

    /// <summary>
    /// Computes the log-likelihood of observations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Student-t likelihood:
    ///
    /// log p(y|f) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5×log(νπσ²)
    ///            - (ν+1)/2 × log(1 + (y-f)²/(νσ²))
    ///
    /// Compared to Gaussian, this penalizes large residuals less severely,
    /// making it more robust to outliers.
    /// </para>
    /// </remarks>
    public T LogLikelihood(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        double nu = _degreesOfFreedom;
        double sigma2 = _noiseScale * _noiseScale;

        // Log normalization constant
        double logNorm = LogGamma(0.5 * (nu + 1)) - LogGamma(0.5 * nu)
                       - 0.5 * Math.Log(nu * Math.PI * sigma2);

        double total = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double diff = _numOps.ToDouble(y[i]) - _numOps.ToDouble(f[i]);
            double z2 = diff * diff / (nu * sigma2);
            total += logNorm - 0.5 * (nu + 1) * Math.Log(1 + z2);
        }

        return _numOps.FromDouble(total);
    }

    /// <summary>
    /// Computes the gradient of the log-likelihood.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient for Student-t:
    ///
    /// ∂log p(y|f)/∂fᵢ = (ν+1) × (yᵢ-fᵢ) / (νσ² + (yᵢ-fᵢ)²)
    ///
    /// Unlike Gaussian, this is bounded for large residuals, which is
    /// what makes Student-t robust - outliers have limited influence.
    /// </para>
    /// </remarks>
    public Vector<T> LogLikelihoodGradient(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));
        if (y.Length != f.Length)
            throw new ArgumentException("y and f must have the same length.");

        double nu = _degreesOfFreedom;
        double sigma2 = _noiseScale * _noiseScale;
        var grad = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            double diff = _numOps.ToDouble(y[i]) - _numOps.ToDouble(f[i]);
            double denom = nu * sigma2 + diff * diff;
            grad[i] = _numOps.FromDouble((nu + 1) * diff / denom);
        }

        return grad;
    }

    /// <summary>
    /// Computes the Hessian diagonal of the log-likelihood.
    /// </summary>
    public Vector<T> LogLikelihoodHessianDiag(Vector<T> y, Vector<T> f)
    {
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (f is null) throw new ArgumentNullException(nameof(f));

        double nu = _degreesOfFreedom;
        double sigma2 = _noiseScale * _noiseScale;
        double nuSigma2 = nu * sigma2;
        var hess = new Vector<T>(f.Length);

        for (int i = 0; i < f.Length; i++)
        {
            double diff = _numOps.ToDouble(y[i]) - _numOps.ToDouble(f[i]);
            double diff2 = diff * diff;
            double denom = nuSigma2 + diff2;
            double denom2 = denom * denom;

            // ∂²/∂f² = -(ν+1) × (νσ² - (y-f)²) / (νσ² + (y-f)²)²
            hess[i] = _numOps.FromDouble(-(nu + 1) * (nuSigma2 - diff2) / denom2);
        }

        return hess;
    }

    /// <summary>
    /// Transforms latent function value (identity for Student-t).
    /// </summary>
    public T TransformMean(T f) => f;

    /// <summary>
    /// Computes predictive variance.
    /// </summary>
    public T PredictiveVariance(T fMean, T fVariance)
    {
        // Variance of Student-t is νσ²/(ν-2) for ν > 2
        double nu = _degreesOfFreedom;
        double sigma2 = _noiseScale * _noiseScale;

        double noiseVariance = nu > 2 ? nu * sigma2 / (nu - 2) : sigma2;
        return _numOps.Add(fVariance, _numOps.FromDouble(noiseVariance));
    }

    /// <summary>
    /// Computes the log of the gamma function using Stirling's approximation.
    /// </summary>
    private static double LogGamma(double x)
    {
        if (x <= 0) return 0;

        // Use Stirling's approximation for large x
        if (x > 10)
        {
            return (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2 * Math.PI)
                   + 1.0 / (12.0 * x);
        }

        // Use recurrence for small x
        double result = 0;
        while (x < 10)
        {
            result -= Math.Log(x);
            x += 1;
        }

        return result + (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2 * Math.PI)
               + 1.0 / (12.0 * x);
    }
}
