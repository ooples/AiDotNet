namespace AiDotNet.Kernels;

/// <summary>
/// Implements the White Noise kernel, which adds independent noise to each observation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The White Noise kernel is a special kernel that models random measurement
/// noise in your data. It returns a non-zero value only when comparing a point to itself.
///
/// Think of it like this: every measurement you make has some random error. This error is
/// independent for each measurement - knowing the error of one measurement tells you nothing
/// about the error of another. The White Noise kernel captures this property.
///
/// In mathematical terms: k(x, x') = σ² if x = x', else 0
///
/// Where σ² is the noise variance (how much noise you expect in your measurements).
/// </para>
/// <para>
/// When to use the White Noise kernel:
/// - When your data has measurement noise that you want to model explicitly
/// - Combined with other kernels (like RBF + WhiteNoise) to separate signal from noise
/// - In Gaussian Process regression to account for observation noise
///
/// The White Noise kernel is rarely used alone - it's usually combined with other kernels
/// to create a more complete model of your data.
/// </para>
/// </remarks>
public class WhiteNoiseKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The noise variance (σ²), which controls the magnitude of the noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter controls how much noise you expect in your measurements.
    ///
    /// - A larger noise variance means you believe your measurements are noisy
    /// - A smaller noise variance means you believe your measurements are precise
    ///
    /// In practice:
    /// - If noiseVariance = 0.1, you expect measurement errors of about ±0.3 (√0.1 ≈ 0.316)
    /// - If noiseVariance = 1.0, you expect measurement errors of about ±1.0
    ///
    /// This helps the model distinguish between true patterns in the data and random noise.
    /// </para>
    /// </remarks>
    private readonly double _noiseVariance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Tolerance for comparing vectors for equality.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// Initializes a new instance of the White Noise kernel.
    /// </summary>
    /// <param name="noiseVariance">The noise variance parameter (σ²). Default is 1.0.</param>
    /// <param name="tolerance">Tolerance for comparing vectors for equality. Default is 1e-10.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the White Noise kernel with your chosen noise level.
    ///
    /// The noise variance should reflect how much random variation you expect in your measurements.
    /// If you don't know, start with 1.0 and adjust based on your model's performance.
    ///
    /// The tolerance is a technical parameter that determines when two vectors are considered
    /// "the same point." The default (1e-10) is usually fine unless you're working with
    /// very high precision or very small numbers.
    /// </para>
    /// </remarks>
    public WhiteNoiseKernel(double noiseVariance = 1.0, double tolerance = 1e-10)
    {
        if (noiseVariance < 0)
            throw new ArgumentException("Noise variance must be non-negative.", nameof(noiseVariance));
        if (tolerance < 0)
            throw new ArgumentException("Tolerance must be non-negative.", nameof(tolerance));

        _noiseVariance = noiseVariance;
        _tolerance = tolerance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the White Noise kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The noise variance if the vectors are identical, zero otherwise.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method returns the noise variance only when comparing
    /// a point to itself. For any two different points, it returns zero.
    ///
    /// Why does this make sense?
    /// - When you measure the same thing twice, the noise is the same (correlated)
    /// - When you measure different things, the noise is independent (uncorrelated)
    ///
    /// This behavior models the assumption that measurement errors are independent
    /// across different observations.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Check if the vectors are the same point (within tolerance)
        bool isSamePoint = true;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = Math.Abs(_numOps.ToDouble(_numOps.Subtract(x1[i], x2[i])));
            if (diff > _tolerance)
            {
                isSamePoint = false;
                break;
            }
        }

        return isSamePoint ? _numOps.FromDouble(_noiseVariance) : _numOps.Zero;
    }
}
