namespace AiDotNet.Kernels;

/// <summary>
/// Spectral Delta Kernel representing a single spectral component.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Spectral Delta Kernel is a single component of the Spectral Mixture Kernel.
/// While the SpectralMixtureKernel uses multiple frequency components, this kernel uses just one.
///
/// In mathematical terms:
/// k(τ) = σ² × exp(-2π²τ²σ_f²) × cos(2πμτ)
///
/// Where:
/// - τ = |x - x'| is the distance between points
/// - σ² is the output variance (weight)
/// - μ is the frequency (how fast the pattern repeats)
/// - σ_f is the frequency bandwidth (how "spread" the frequency is)
///
/// Think of it as a single "note" in the spectrum:
/// - μ determines the pitch (frequency)
/// - σ_f determines how pure the tone is (narrow = pure, wide = noisy)
/// - σ² determines the volume
///
/// This is useful when:
/// - You expect a single dominant periodicity
/// - You want a simple periodic pattern with decay
/// - As a building block for more complex spectral kernels
/// </para>
/// </remarks>
public class SpectralDeltaKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The mean frequency (μ).
    /// </summary>
    private readonly double _frequency;

    /// <summary>
    /// The frequency bandwidth (σ_f).
    /// </summary>
    private readonly double _bandwidth;

    /// <summary>
    /// The output variance (weight).
    /// </summary>
    private readonly double _variance;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Spectral Delta Kernel.
    /// </summary>
    /// <param name="frequency">The mean frequency (period = 1/frequency).</param>
    /// <param name="bandwidth">The frequency bandwidth (controls decay rate).</param>
    /// <param name="variance">The output variance (weight). Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a spectral kernel with a single frequency component.
    ///
    /// Choosing parameters:
    /// - frequency: Related to periodicity. period = 1/frequency
    ///   - frequency = 1.0 → period = 1 unit
    ///   - frequency = 0.1 → period = 10 units
    ///   - frequency = 7.0 → period ≈ 0.14 units (7 cycles per unit)
    ///
    /// - bandwidth: Controls how "localized" the pattern is
    ///   - Small bandwidth (0.01): Pattern persists over long distances
    ///   - Large bandwidth (1.0): Pattern decays quickly
    ///
    /// Example for daily periodicity in hourly data:
    /// var kernel = new SpectralDeltaKernel&lt;double&gt;(
    ///     frequency: 1.0/24,  // One cycle per 24 hours
    ///     bandwidth: 0.01     // Pattern persists
    /// );
    /// </para>
    /// </remarks>
    public SpectralDeltaKernel(double frequency, double bandwidth, double variance = 1.0)
    {
        if (bandwidth <= 0)
            throw new ArgumentException("Bandwidth must be positive.", nameof(bandwidth));
        if (variance <= 0)
            throw new ArgumentException("Variance must be positive.", nameof(variance));

        _frequency = frequency;
        _bandwidth = bandwidth;
        _variance = variance;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the mean frequency.
    /// </summary>
    public double Frequency => _frequency;

    /// <summary>
    /// Gets the period (1/frequency). Returns infinity if frequency is 0.
    /// </summary>
    public double Period => Math.Abs(_frequency) < 1e-10 ? double.PositiveInfinity : 1.0 / _frequency;

    /// <summary>
    /// Gets the frequency bandwidth.
    /// </summary>
    public double Bandwidth => _bandwidth;

    /// <summary>
    /// Gets the output variance.
    /// </summary>
    public double Variance => _variance;

    /// <summary>
    /// Calculates the spectral delta kernel value.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the kernel as:
    /// 1. Calculate distance τ = |x1 - x2|
    /// 2. Envelope: exp(-2π²τ²σ_f²) - controls decay
    /// 3. Oscillation: cos(2πτμ) - creates periodicity
    /// 4. Result: variance × envelope × oscillation
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        double result = 0;

        // For each dimension, compute spectral delta contribution
        for (int d = 0; d < x1.Length; d++)
        {
            double tau = _numOps.ToDouble(x1[d]) - _numOps.ToDouble(x2[d]);
            double tau2 = tau * tau;

            // Envelope: exp(-2π²τ²σ²)
            double envelope = Math.Exp(-2.0 * Math.PI * Math.PI * tau2 * _bandwidth * _bandwidth);

            // Oscillation: cos(2πτμ)
            double oscillation = Math.Cos(2.0 * Math.PI * tau * _frequency);

            result += _variance * envelope * oscillation;
        }

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Computes the power spectral density at a given frequency.
    /// </summary>
    /// <param name="f">The frequency to evaluate.</param>
    /// <returns>The power spectral density value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The power spectral density shows how much "power" is at each frequency.
    /// For a spectral delta kernel, it's a Gaussian centered at the mean frequency.
    ///
    /// Use this to visualize what frequencies the kernel represents.
    /// </para>
    /// </remarks>
    public double GetPowerSpectralDensity(double f)
    {
        // PSD is a Gaussian centered at the mean frequency
        double diff = f - _frequency;
        double exponent = -0.5 * diff * diff / (_bandwidth * _bandwidth);
        return _variance * Math.Exp(exponent) / (Math.Sqrt(2 * Math.PI) * _bandwidth);
    }

    /// <summary>
    /// Creates a Spectral Delta Kernel from a known period.
    /// </summary>
    /// <param name="period">The period of the pattern.</param>
    /// <param name="decayLength">Approximate distance over which the pattern decays.</param>
    /// <param name="variance">The output variance.</param>
    /// <returns>A new Spectral Delta Kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convenience factory when you know the period of your pattern.
    ///
    /// Example: For weekly patterns in daily data:
    /// var kernel = SpectralDeltaKernel&lt;double&gt;.FromPeriod(
    ///     period: 7,       // 7 days
    ///     decayLength: 30  // Pattern decays over ~30 days
    /// );
    /// </para>
    /// </remarks>
    public static SpectralDeltaKernel<T> FromPeriod(double period, double decayLength = 10.0, double variance = 1.0)
    {
        if (period <= 0)
            throw new ArgumentException("Period must be positive.", nameof(period));
        if (decayLength <= 0)
            throw new ArgumentException("Decay length must be positive.", nameof(decayLength));

        double frequency = 1.0 / period;
        // Bandwidth controls decay: larger bandwidth = faster decay
        // Setting so that envelope ≈ 0.01 at decayLength
        double bandwidth = Math.Sqrt(-Math.Log(0.01) / (2 * Math.PI * Math.PI * decayLength * decayLength));

        return new SpectralDeltaKernel<T>(frequency, bandwidth, variance);
    }
}
