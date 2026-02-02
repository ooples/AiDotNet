namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Matérn family of kernels with configurable smoothness parameter.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Matérn kernel is one of the most important kernels in GP practice.
/// It provides a middle ground between the very smooth RBF kernel and rougher kernels.
///
/// The smoothness parameter ν controls how "wiggly" the functions can be:
/// - ν = 1/2: Exponential kernel (functions are continuous but not differentiable)
/// - ν = 3/2: Once differentiable (good for many real-world applications)
/// - ν = 5/2: Twice differentiable (often used as default)
/// - ν → ∞: RBF/Gaussian kernel (infinitely differentiable)
///
/// In mathematical terms:
/// k(r) = (2^(1-ν)/Γ(ν)) × (√(2ν)×r/l)^ν × K_ν(√(2ν)×r/l)
///
/// Where:
/// - r = |x - x'| is the distance
/// - l is the length scale
/// - ν is the smoothness parameter
/// - K_ν is the modified Bessel function
/// </para>
/// <para>
/// Why use Matérn over RBF?
///
/// 1. **More realistic**: Real-world functions are rarely infinitely smooth
/// 2. **Better extrapolation**: RBF can be overly smooth for extrapolation
/// 3. **Physical motivation**: Many physical processes have finite smoothness
/// 4. **Computational**: ν = 1/2, 3/2, 5/2 have simple closed forms
///
/// Rule of thumb:
/// - If you're unsure, start with ν = 5/2 (Matérn 5/2)
/// - For rough data, try ν = 3/2
/// - For very noisy data, ν = 1/2 might help
/// </para>
/// </remarks>
public class MaternKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The smoothness parameter (ν).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how "smooth" or "wiggly" the GP functions are.
    ///
    /// Common values:
    /// - 0.5: Ornstein-Uhlenbeck process (rough, only continuous)
    /// - 1.5: Once differentiable (moderate smoothness)
    /// - 2.5: Twice differentiable (fairly smooth, popular default)
    /// - Higher values approach RBF behavior
    /// </para>
    /// </remarks>
    private readonly double _nu;

    /// <summary>
    /// The length scale parameter.
    /// </summary>
    private readonly double _lengthScale;

    /// <summary>
    /// The signal variance (kernel scale).
    /// </summary>
    private readonly double _variance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Matérn kernel.
    /// </summary>
    /// <param name="nu">The smoothness parameter. Common values: 0.5, 1.5, 2.5.</param>
    /// <param name="lengthScale">The length scale parameter. Default is 1.0.</param>
    /// <param name="variance">The signal variance. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Matérn kernel with specified smoothness.
    ///
    /// Recommended starting points:
    /// - nu=2.5 is a good default for most problems
    /// - lengthScale: Set based on the scale of your input data
    /// - variance: Set based on the variance of your output data
    ///
    /// Example:
    /// var kernel = new MaternKernel&lt;double&gt;(nu: 2.5, lengthScale: 1.0);
    /// </para>
    /// </remarks>
    public MaternKernel(double nu = 2.5, double lengthScale = 1.0, double variance = 1.0)
    {
        if (nu <= 0)
            throw new ArgumentException("Smoothness parameter nu must be positive.", nameof(nu));
        if (lengthScale <= 0)
            throw new ArgumentException("Length scale must be positive.", nameof(lengthScale));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        _nu = nu;
        _lengthScale = lengthScale;
        _variance = variance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the smoothness parameter (ν).
    /// </summary>
    public double Nu => _nu;

    /// <summary>
    /// Gets the length scale.
    /// </summary>
    public double LengthScale => _lengthScale;

    /// <summary>
    /// Gets the signal variance.
    /// </summary>
    public double Variance => _variance;

    /// <summary>
    /// Calculates the Matérn kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the covariance between two points using the Matérn function.
    ///
    /// For common values of ν, we use closed-form expressions:
    /// - ν = 1/2: k(r) = σ² × exp(-r/l)
    /// - ν = 3/2: k(r) = σ² × (1 + √3×r/l) × exp(-√3×r/l)
    /// - ν = 5/2: k(r) = σ² × (1 + √5×r/l + 5r²/(3l²)) × exp(-√5×r/l)
    ///
    /// These closed forms are much faster than the general Bessel function formulation.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Compute Euclidean distance
        double r = 0;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            r += diff * diff;
        }
        r = Math.Sqrt(r);

        double result = ComputeMatern(r);
        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Computes the Matérn function value for a given distance.
    /// </summary>
    /// <param name="r">The Euclidean distance.</param>
    /// <returns>The kernel value.</returns>
    private double ComputeMatern(double r)
    {
        if (r < 1e-10)
        {
            return _variance;
        }

        // Use closed-form expressions for common values of nu
        double scaledR = r / _lengthScale;

        // Check for special cases (within tolerance)
        if (Math.Abs(_nu - 0.5) < 0.01)
        {
            // Matérn 1/2 (exponential)
            return _variance * Math.Exp(-scaledR);
        }
        else if (Math.Abs(_nu - 1.5) < 0.01)
        {
            // Matérn 3/2
            double sqrt3R = Math.Sqrt(3.0) * scaledR;
            return _variance * (1.0 + sqrt3R) * Math.Exp(-sqrt3R);
        }
        else if (Math.Abs(_nu - 2.5) < 0.01)
        {
            // Matérn 5/2
            double sqrt5R = Math.Sqrt(5.0) * scaledR;
            return _variance * (1.0 + sqrt5R + sqrt5R * sqrt5R / 3.0) * Math.Exp(-sqrt5R);
        }
        else
        {
            // General case using Bessel function approximation
            return ComputeGeneralMatern(scaledR);
        }
    }

    /// <summary>
    /// Computes the general Matérn function using Bessel function approximation.
    /// </summary>
    private double ComputeGeneralMatern(double scaledR)
    {
        double sqrt2NuR = Math.Sqrt(2.0 * _nu) * scaledR;

        if (sqrt2NuR < 1e-10)
        {
            return _variance;
        }

        // Use asymptotic expansion for modified Bessel function K_nu
        // For moderate arguments, use polynomial approximation
        double besselK = ApproximateBesselK(_nu, sqrt2NuR);
        double factor = Math.Pow(2.0, 1.0 - _nu) / GammaFunction(_nu);
        double powerTerm = Math.Pow(sqrt2NuR, _nu);

        return _variance * factor * powerTerm * besselK;
    }

    /// <summary>
    /// Approximates the modified Bessel function K_nu(x).
    /// </summary>
    private static double ApproximateBesselK(double nu, double x)
    {
        if (x < 0.01)
        {
            // Small x approximation
            if (Math.Abs(nu) < 0.01)
            {
                return -Math.Log(x / 2) - 0.5772156649; // Euler-Mascheroni
            }
            else
            {
                return 0.5 * GammaFunction(nu) * Math.Pow(x / 2, -nu);
            }
        }
        else if (x > 10)
        {
            // Large x asymptotic expansion
            return Math.Sqrt(Math.PI / (2 * x)) * Math.Exp(-x) *
                   (1 + (4 * nu * nu - 1) / (8 * x));
        }
        else
        {
            // Intermediate: use Miller's algorithm or series
            return BesselKMiller(nu, x);
        }
    }

    /// <summary>
    /// Miller's algorithm for modified Bessel function K_nu.
    /// </summary>
    private static double BesselKMiller(double nu, double x)
    {
        // For non-integer nu, use relation between K_nu and K_{-nu}
        // K_nu = K_{-nu} (symmetry)

        // Use recurrence relation and continued fraction
        // This is a simplified implementation

        // For half-integer nu, there are closed forms
        double halfNu = nu - Math.Floor(nu);
        if (Math.Abs(halfNu - 0.5) < 0.01)
        {
            int n = (int)Math.Floor(nu);
            return BesselKHalfInteger(n, x);
        }

        // General approximation using series
        double sum = 0;
        double term = 1;
        double gamma0 = GammaFunction(nu);
        double gamma1 = GammaFunction(-nu);
        double x2 = x * x / 4;

        for (int k = 0; k < 50; k++)
        {
            if (k > 0)
            {
                term *= x2 / (k * (k + nu)) * (k - 1 + nu > 0 ? 1 : -1);
            }
            sum += term;
            if (Math.Abs(term) < 1e-15 * Math.Abs(sum))
                break;
        }

        return Math.PI / (2 * Math.Sin(Math.PI * nu)) *
               (Math.Pow(x / 2, -nu) * gamma0 - Math.Pow(x / 2, nu) / gamma1) * sum;
    }

    /// <summary>
    /// Computes K_{n+1/2} using closed form for half-integer orders.
    /// </summary>
    private static double BesselKHalfInteger(int n, double x)
    {
        // K_{1/2}(x) = sqrt(pi/(2x)) * exp(-x)
        double k0 = Math.Sqrt(Math.PI / (2 * x)) * Math.Exp(-x);

        if (n == 0)
            return k0;

        // K_{3/2}(x) = sqrt(pi/(2x)) * exp(-x) * (1 + 1/x)
        double k1 = k0 * (1 + 1 / x);

        if (n == 1)
            return k1;

        // Use recurrence: K_{n+1}(x) = K_{n-1}(x) + (2n/x) * K_n(x)
        double kPrev = k0;
        double kCurr = k1;

        for (int i = 1; i < n; i++)
        {
            double kNext = kPrev + (2.0 * (i + 0.5) / x) * kCurr;
            kPrev = kCurr;
            kCurr = kNext;
        }

        return kCurr;
    }

    /// <summary>
    /// Approximates the Gamma function.
    /// </summary>
    private static double GammaFunction(double z)
    {
        if (z <= 0)
        {
            // Use reflection formula for negative values
            return Math.PI / (Math.Sin(Math.PI * z) * GammaFunction(1 - z));
        }

        // Use Stirling's approximation for large z
        if (z > 10)
        {
            return Math.Sqrt(2 * Math.PI / z) * Math.Pow(z / Math.E, z) *
                   (1 + 1 / (12 * z) + 1 / (288 * z * z));
        }

        // Use Lanczos approximation for small z
        double[] g = { 1.000000000000000174663, 5716.400188274341379136,
            -14815.30426768413909044, 14291.49277657478554025,
            -6348.160217641458813289, 1301.608286058321874105,
            -108.1767053514369634679, 2.605696505611755827729,
            -0.7423452510201416151527e-2, 0.5384136432509564062961e-7,
            -0.4023533141268236372067e-8 };

        if (z < 0.5)
        {
            return Math.PI / (Math.Sin(Math.PI * z) * GammaFunction(1 - z));
        }

        z -= 1;
        double x = g[0];
        for (int i = 1; i < 11; i++)
        {
            x += g[i] / (z + i);
        }

        double t = z + 10.5;
        return Math.Sqrt(2 * Math.PI) * Math.Pow(t, z + 0.5) * Math.Exp(-t) * x;
    }

    /// <summary>
    /// Creates a Matérn 1/2 (exponential) kernel.
    /// </summary>
    /// <param name="lengthScale">The length scale.</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A Matérn kernel with ν = 1/2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Matérn 1/2 is equivalent to the exponential kernel.
    /// It produces continuous but non-differentiable functions (rough/jagged).
    ///
    /// Use when: Your data has abrupt changes or you model an Ornstein-Uhlenbeck process.
    /// </para>
    /// </remarks>
    public static MaternKernel<T> Matern12(double lengthScale = 1.0, double variance = 1.0)
    {
        return new MaternKernel<T>(nu: 0.5, lengthScale: lengthScale, variance: variance);
    }

    /// <summary>
    /// Creates a Matérn 3/2 kernel.
    /// </summary>
    /// <param name="lengthScale">The length scale.</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A Matérn kernel with ν = 3/2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Matérn 3/2 produces once-differentiable functions.
    /// This is a good choice for moderately smooth data.
    ///
    /// Use when: Your data is fairly smooth but not extremely so.
    /// </para>
    /// </remarks>
    public static MaternKernel<T> Matern32(double lengthScale = 1.0, double variance = 1.0)
    {
        return new MaternKernel<T>(nu: 1.5, lengthScale: lengthScale, variance: variance);
    }

    /// <summary>
    /// Creates a Matérn 5/2 kernel.
    /// </summary>
    /// <param name="lengthScale">The length scale.</param>
    /// <param name="variance">The signal variance.</param>
    /// <returns>A Matérn kernel with ν = 5/2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Matérn 5/2 produces twice-differentiable functions.
    /// This is often the recommended default for GP regression.
    ///
    /// Use when: You're not sure which kernel to use - this is a good starting point.
    /// </para>
    /// </remarks>
    public static MaternKernel<T> Matern52(double lengthScale = 1.0, double variance = 1.0)
    {
        return new MaternKernel<T>(nu: 2.5, lengthScale: lengthScale, variance: variance);
    }
}
