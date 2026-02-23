namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Spectral Mixture (SM) kernel for discovering and exploiting patterns in data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Spectral Mixture kernel is a highly flexible kernel that can learn
/// complex patterns from data, including multiple periodicities at different scales.
///
/// Key idea: Any stationary kernel can be expressed as a mixture of cosine functions
/// (this is Bochner's theorem). The SM kernel makes this explicit by parameterizing
/// a mixture of Gaussians in the frequency domain.
///
/// In mathematical terms:
/// k(τ) = Σᵢ wᵢ × exp(-2π²τ²σᵢ²) × cos(2πτμᵢ)
///
/// Where:
/// - τ = |x - x'| is the distance between points
/// - wᵢ is the weight (importance) of component i
/// - μᵢ is the frequency (how fast the pattern repeats)
/// - σᵢ is the bandwidth (how wide the peak is in frequency domain)
/// </para>
/// <para>
/// Why use Spectral Mixture?
///
/// 1. **Pattern Discovery**: Automatically finds periodicities in data
/// 2. **Multiple Scales**: Can capture patterns at different frequencies
/// 3. **Interpretability**: Components correspond to different pattern types
/// 4. **Flexibility**: Can approximate any stationary kernel
///
/// Examples:
/// - Stock prices: Daily, weekly, monthly, yearly patterns
/// - Weather data: Daily and seasonal cycles
/// - Audio signals: Multiple frequency components
/// </para>
/// </remarks>
public class SpectralMixtureKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The weights for each mixture component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weights control how important each frequency component is.
    /// Higher weight = that pattern is more prominent in the data.
    /// Weights are typically initialized uniformly and learned during training.
    /// </para>
    /// </remarks>
    private readonly double[] _weights;

    /// <summary>
    /// The frequencies (means) for each mixture component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frequencies determine how fast each pattern repeats.
    ///
    /// Period = 1 / frequency:
    /// - μ = 1.0 → Pattern repeats every 1 unit of x
    /// - μ = 0.1 → Pattern repeats every 10 units of x
    /// - μ = 7.0 → Pattern repeats every ~0.14 units of x (7 cycles per unit)
    ///
    /// After training, inspect these to see what periodicities were discovered.
    /// </para>
    /// </remarks>
    private readonly double[] _frequencies;

    /// <summary>
    /// The bandwidths (length scales) for each mixture component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bandwidths control how "sharp" each frequency peak is.
    ///
    /// - Small bandwidth: Very specific frequency (narrow peak)
    /// - Large bandwidth: Broad range of frequencies (wide peak)
    ///
    /// In the time domain, this affects how quickly the pattern decays:
    /// - Large bandwidth → Fast decay → Pattern only matters locally
    /// - Small bandwidth → Slow decay → Pattern persists over long distances
    /// </para>
    /// </remarks>
    private readonly double[] _bandwidths;

    /// <summary>
    /// The number of mixture components.
    /// </summary>
    private readonly int _numComponents;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Spectral Mixture kernel with specified parameters.
    /// </summary>
    /// <param name="weights">The weights for each component.</param>
    /// <param name="frequencies">The frequencies (means) for each component.</param>
    /// <param name="bandwidths">The bandwidths (variances) for each component.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Spectral Mixture kernel with explicit parameters.
    ///
    /// The number of components determines model complexity:
    /// - 1 component: Equivalent to scaled RBF with periodic modulation
    /// - 2-3 components: Can capture a couple of distinct patterns
    /// - 5-10 components: Can model complex multi-scale patterns
    ///
    /// Example for daily + weekly patterns:
    /// var kernel = new SpectralMixtureKernel&lt;double&gt;(
    ///     weights: new[] { 1.0, 1.0 },
    ///     frequencies: new[] { 1.0/24, 1.0/168 }, // 24 hours, 168 hours
    ///     bandwidths: new[] { 0.1, 0.1 }
    /// );
    /// </para>
    /// </remarks>
    public SpectralMixtureKernel(double[] weights, double[] frequencies, double[] bandwidths)
    {
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (frequencies is null) throw new ArgumentNullException(nameof(frequencies));
        if (bandwidths is null) throw new ArgumentNullException(nameof(bandwidths));

        if (weights.Length == 0)
            throw new ArgumentException("Must have at least one component.", nameof(weights));
        if (weights.Length != frequencies.Length || weights.Length != bandwidths.Length)
            throw new ArgumentException("All parameter arrays must have the same length.");

        for (int i = 0; i < weights.Length; i++)
        {
            if (weights[i] <= 0)
                throw new ArgumentException("Weights must be positive.", nameof(weights));
            if (bandwidths[i] <= 0)
                throw new ArgumentException("Bandwidths must be positive.", nameof(bandwidths));
        }

        _weights = (double[])weights.Clone();
        _frequencies = (double[])frequencies.Clone();
        _bandwidths = (double[])bandwidths.Clone();
        _numComponents = weights.Length;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Initializes a new Spectral Mixture kernel with default initialization.
    /// </summary>
    /// <param name="numComponents">The number of mixture components.</param>
    /// <param name="maxFrequency">The maximum frequency to consider. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a Spectral Mixture kernel with random initialization.
    ///
    /// The frequencies are initialized uniformly across the range [0, maxFrequency].
    /// Use hyperparameter optimization to learn the best values from your data.
    ///
    /// Choosing numComponents:
    /// - Start with 5-10 for most problems
    /// - Increase if you expect many different patterns
    /// - Each component adds ~3 hyperparameters to optimize
    /// </para>
    /// </remarks>
    public SpectralMixtureKernel(int numComponents, double maxFrequency = 1.0)
    {
        if (numComponents < 1)
            throw new ArgumentException("Must have at least one component.", nameof(numComponents));
        if (maxFrequency <= 0)
            throw new ArgumentException("Max frequency must be positive.", nameof(maxFrequency));

        _numComponents = numComponents;
        _weights = new double[numComponents];
        _frequencies = new double[numComponents];
        _bandwidths = new double[numComponents];

        var rand = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < numComponents; i++)
        {
            _weights[i] = 1.0 / numComponents; // Uniform weights
            _frequencies[i] = (i + 0.5) * maxFrequency / numComponents; // Spread across range
            _bandwidths[i] = maxFrequency / (2.0 * numComponents) + rand.NextDouble() * 0.1;
        }

        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Spectral Mixture kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes similarity considering multiple frequency patterns.
    ///
    /// The calculation for each dimension d:
    /// For each component q:
    ///   - Compute τ = x1[d] - x2[d] (the distance)
    ///   - Envelope: exp(-2π²τ²σ_q²) - controls decay with distance
    ///   - Oscillation: cos(2πτμ_q) - creates the periodic pattern
    ///   - Component contribution: w_q × envelope × oscillation
    ///
    /// Sum over all components and dimensions.
    ///
    /// This creates rich patterns:
    /// - Multiple periodicities (different μ values)
    /// - Different decay rates (different σ values)
    /// - Different importance levels (different w values)
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        double result = 0.0;

        // Spectral Mixture kernel: k(x,x') = Σ_q w_q * Π_d [exp(-2π²τ_d²σ_q²) * cos(2πτ_d*μ_q)]
        // Sum over mixture components, product over dimensions (separable kernel)
        for (int q = 0; q < _numComponents; q++)
        {
            double weight = _weights[q];
            double mu = _frequencies[q];
            double sigma = _bandwidths[q];

            // Product over dimensions for this component
            double componentKernel = 1.0;
            for (int d = 0; d < x1.Length; d++)
            {
                double tau = _numOps.ToDouble(x1[d]) - _numOps.ToDouble(x2[d]);
                double tau2 = tau * tau;

                // Envelope: exp(-2π²τ²σ²)
                double envelope = Math.Exp(-2.0 * Math.PI * Math.PI * tau2 * sigma * sigma);

                // Oscillation: cos(2πτμ)
                double oscillation = Math.Cos(2.0 * Math.PI * tau * mu);

                componentKernel *= envelope * oscillation;
            }

            result += weight * componentKernel;
        }

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Gets the number of mixture components.
    /// </summary>
    public int NumComponents => _numComponents;

    /// <summary>
    /// Gets a copy of the component weights.
    /// </summary>
    /// <returns>The weights array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting, large weights indicate important patterns.
    /// Components with very small weights contribute little and could be removed.
    /// </para>
    /// </remarks>
    public double[] GetWeights() => (double[])_weights.Clone();

    /// <summary>
    /// Gets a copy of the component frequencies.
    /// </summary>
    /// <returns>The frequencies array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting, these show the periodicities found in your data.
    /// Period = 1/frequency tells you the length of each cycle.
    /// </para>
    /// </remarks>
    public double[] GetFrequencies() => (double[])_frequencies.Clone();

    /// <summary>
    /// Gets a copy of the component bandwidths.
    /// </summary>
    /// <returns>The bandwidths array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting, these show how localized each pattern is.
    /// Smaller bandwidths mean the pattern decays more slowly.
    /// </para>
    /// </remarks>
    public double[] GetBandwidths() => (double[])_bandwidths.Clone();

    /// <summary>
    /// Estimates initial frequencies from data using spectral analysis.
    /// </summary>
    /// <param name="data">Time series data to analyze.</param>
    /// <returns>Array of estimated dominant frequencies.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helps initialize frequencies to good starting values
    /// by looking at the data's frequency content.
    ///
    /// The method uses a simple periodogram (squared Fourier transform) to find
    /// which frequencies have the most power in your data.
    ///
    /// Usage:
    /// var freqs = SpectralMixtureKernel&lt;double&gt;.EstimateFrequencies(myData, 5);
    /// var kernel = new SpectralMixtureKernel&lt;double&gt;(weights, freqs, bandwidths);
    /// </para>
    /// </remarks>
    public static double[] EstimateFrequencies(double[] data, int numFrequencies)
    {
        if (data is null || data.Length == 0)
            throw new ArgumentException("Data cannot be null or empty.", nameof(data));
        if (numFrequencies < 1)
            throw new ArgumentException("Must request at least one frequency.", nameof(numFrequencies));

        int n = data.Length;

        // Center the data
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            mean += data[i];
        }
        mean /= n;

        var centered = new double[n];
        for (int i = 0; i < n; i++)
        {
            centered[i] = data[i] - mean;
        }

        // Compute periodogram using DFT
        int numFreqs = n / 2;
        var power = new double[numFreqs];
        var frequencies = new double[numFreqs];

        for (int k = 1; k < numFreqs; k++)
        {
            double freq = (double)k / n;
            frequencies[k] = freq;

            double realPart = 0;
            double imagPart = 0;

            for (int t = 0; t < n; t++)
            {
                double angle = 2.0 * Math.PI * k * t / n;
                realPart += centered[t] * Math.Cos(angle);
                imagPart -= centered[t] * Math.Sin(angle);
            }

            power[k] = (realPart * realPart + imagPart * imagPart) / n;
        }

        // Find top frequencies
        var indices = new int[numFreqs];
        for (int i = 0; i < numFreqs; i++)
        {
            indices[i] = i;
        }

        // Sort by power (descending)
        Array.Sort(power, indices);
        Array.Reverse(power);
        Array.Reverse(indices);

        // Return top frequencies (skip DC component at index 0)
        var result = new double[Math.Min(numFrequencies, numFreqs - 1)];
        int resultIdx = 0;
        for (int i = 0; i < numFreqs && resultIdx < result.Length; i++)
        {
            if (indices[i] > 0) // Skip DC
            {
                result[resultIdx++] = (double)indices[i] / n;
            }
        }

        return result;
    }
}
