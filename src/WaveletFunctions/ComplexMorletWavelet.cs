namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements a Complex Morlet wavelet, which is a complex exponential modulated by a Gaussian window,
/// making it well-suited for time-frequency analysis of signals.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Complex Morlet wavelet is one of the most widely used wavelets for time-frequency analysis.
/// It consists of a complex exponential (sine and cosine) modulated by a Gaussian envelope,
/// providing excellent time-frequency localization and the ability to analyze both amplitude
/// and phase information in signals.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Complex Morlet wavelet is like a mathematical magnifying glass that can see both
/// the "what" (frequency) and "when" (time) of patterns in your data simultaneously.
/// 
/// Key features of Complex Morlet wavelets:
/// - They combine a sine wave with a bell-shaped curve (Gaussian)
/// - They can detect oscillations in signals with great precision
/// - They work with complex numbers to capture both amplitude and phase
/// - They're excellent for analyzing rhythmic or oscillatory patterns
/// 
/// Think of them as special detectors that can find specific "musical notes" in your data
/// and tell you exactly when they occur.
/// 
/// These wavelets are particularly useful for:
/// - Audio processing and music analysis
/// - Brain wave (EEG) analysis
/// - Vibration analysis in mechanical systems
/// - Financial time series analysis
/// - Any application where finding oscillatory patterns is important
/// 
/// The parameters omega and sigma let you tune the wavelet to look for specific frequencies
/// and control how precise it is in time versus frequency.
/// </para>
/// </remarks>
public class ComplexMorletWavelet<T> : IWaveletFunction<Complex<T>>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;
    
    /// <summary>
    /// The central frequency of the wavelet.
    /// </summary>
    private readonly T _omega = default!;
    
    /// <summary>
    /// The bandwidth parameter controlling the width of the Gaussian window.
    /// </summary>
    private readonly T _sigma = default!;

    /// <summary>
    /// Initializes a new instance of the ComplexMorletWavelet class with the specified parameters.
    /// </summary>
    /// <param name="omega">The central frequency of the wavelet. Default is 5.0.</param>
    /// <param name="sigma">The bandwidth parameter controlling the width of the Gaussian window. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The two parameters control different aspects of the wavelet:
    /// 
    /// 1. Omega (ω): The central frequency of the wavelet
    ///    - Higher values look for higher-frequency oscillations in your data
    ///    - Lower values look for lower-frequency oscillations
    ///    - Think of this as tuning which "musical note" you're looking for
    ///    - Default value of 5.0 is a good general-purpose setting
    /// 
    /// 2. Sigma (σ): The bandwidth parameter
    ///    - Controls the width of the Gaussian window
    ///    - Affects the trade-off between time and frequency precision
    ///    - Smaller values: Better time localization, poorer frequency resolution
    ///    - Larger values: Better frequency resolution, poorer time localization
    ///    - Default value of 1.0 provides a balanced trade-off
    /// 
    /// The relationship between these parameters determines the wavelet's properties:
    /// - The product ω·σ should be > 5 to ensure admissibility (a mathematical requirement)
    /// - The default values (ω=5, σ=1) satisfy this condition
    /// 
    /// You might adjust these parameters when:
    /// - Looking for specific frequency components (adjust ω)
    /// - Needing better time precision or frequency precision (adjust σ)
    /// </para>
    /// </remarks>
    public ComplexMorletWavelet(double omega = 5.0, double sigma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = _numOps.FromDouble(omega);
        _sigma = _numOps.FromDouble(sigma);
    }

    /// <summary>
    /// Calculates the value of the Complex Morlet wavelet function at point z.
    /// </summary>
    /// <param name="z">The complex point at which to evaluate the wavelet function.</param>
    /// <returns>The complex value of the wavelet function at point z.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the Complex Morlet wavelet at a specific point.
    /// 
    /// The Complex Morlet wavelet is defined as:
    /// ψ(t) = (e^(iωt) · e^(-t²/(2σ²)))
    /// 
    /// Which can be broken down into:
    /// - e^(iωt) = cos(ωt) + i·sin(ωt): The complex exponential (oscillating part)
    /// - e^(-t²/(2σ²)): The Gaussian envelope (bell-shaped curve)
    /// 
    /// For a complex input z = x + iy, this function:
    /// 1. Calculates the Gaussian envelope based on the distance from the origin
    /// 2. Multiplies it by the cosine (for the real part) and sine (for the imaginary part)
    /// 3. Returns the resulting complex number
    /// 
    /// The result is a localized wave packet that oscillates at frequency ω within
    /// a Gaussian envelope of width controlled by σ.
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T expTerm = _numOps.Exp(_numOps.Negate(_numOps.Divide(_numOps.Add(_numOps.Multiply(x, x), _numOps.Multiply(y, y)), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(_numOps.Multiply(_omega, x));
        T sinTerm = MathHelper.Sin(_numOps.Multiply(_omega, x));

        T real = _numOps.Multiply(expTerm, cosTerm);
        T imag = _numOps.Multiply(expTerm, sinTerm);

        return new Complex<T>(real, imag);
    }

    /// <summary>
    /// Decomposes a complex input signal into approximation and detail coefficients using the Complex Morlet wavelet.
    /// </summary>
    /// <param name="input">The complex input signal vector to decompose.</param>
    /// <returns>A tuple containing the complex approximation coefficients and complex detail coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs one level of wavelet decomposition on your complex signal, splitting it into:
    /// 
    /// - Approximation coefficients: Represent the low-frequency components (the overall shape)
    /// - Detail coefficients: Represent the high-frequency components (the fine details)
    /// 
    /// The process works like this:
    /// 1. The input signal is convolved (filtered) with a scaling function to get the approximation
    /// 2. The input signal is convolved with the wavelet function to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes Complex Morlet wavelets special for this task is:
    /// - They can capture oscillatory patterns with great precision
    /// - They preserve both amplitude and phase information
    /// - They have excellent time-frequency localization
    /// 
    /// This is particularly useful for analyzing signals with rhythmic or oscillatory components,
    /// such as audio, EEG, vibration data, or financial time series.
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which makes wavelet decomposition efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public (Vector<Complex<T>> approximation, Vector<Complex<T>> detail) Decompose(Vector<Complex<T>> input)
    {
        var waveletCoeffs = GetWaveletCoefficients();
        var scalingCoeffs = GetScalingCoefficients();

        var approximation = Convolve(input, scalingCoeffs);
        var detail = Convolve(input, waveletCoeffs);

        // Downsample by 2
        approximation = Downsample(approximation, 2);
        detail = Downsample(detail, 2);

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Complex Morlet wavelet.
    /// </summary>
    /// <returns>A vector of complex scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    /// 
    /// For Complex Morlet wavelets, these coefficients are derived from a sinc function:
    /// 
    /// sinc(x) = sin(πx)/(πx)
    /// 
    /// The sinc function is the ideal low-pass filter in signal processing theory.
    /// It lets through all frequencies below a cutoff point and blocks all frequencies above it.
    /// 
    /// This method:
    /// 1. Creates a discretized sinc function of specified length
    /// 2. Normalizes the coefficients so they sum to 1
    /// 
    /// The resulting filter captures the low-frequency components of the signal,
    /// providing the "approximation" part of the wavelet decomposition.
    /// 
    /// Note that the sinc function has infinite support in theory, but we truncate it
    /// to a finite length for practical implementation.
    /// </para>
    /// </remarks>
    public Vector<Complex<T>> GetScalingCoefficients()
    {
        int length = 64; // Adjust as needed
        var coeffs = new Complex<T>[length];
        T sum = _numOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(i - length / 2), _numOps.FromDouble(length / 4));
            T value = _numOps.Equals(x, _numOps.Zero)
                ? _numOps.One
                : _numOps.Divide(MathHelper.Sin(_numOps.Divide(MathHelper.Pi<T>(), x)), _numOps.Multiply(MathHelper.Pi<T>(), x));
            coeffs[i] = new Complex<T>(value, _numOps.Zero);
            sum = _numOps.Add(sum, _numOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(_numOps.Divide(coeffs[i].Real, sum), _numOps.Zero);
        }

        return new Vector<Complex<T>>(coeffs);
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Complex Morlet wavelet.
    /// </summary>
    /// <returns>A vector of complex wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For Complex Morlet wavelets, these coefficients are a discretized version of the
    /// Complex Morlet wavelet function:
    /// 
    /// ψ(t) = e^(iωt) · e^(-t²/(2σ²))
    /// 
    /// This method:
    /// 1. Creates a discretized Complex Morlet wavelet of specified length
    /// 2. The real part is the Gaussian-modulated cosine: e^(-t²/(2σ²)) · cos(ωt)
    /// 3. The imaginary part is the Gaussian-modulated sine: e^(-t²/(2σ²)) · sin(ωt)
    /// 4. Normalizes the coefficients to ensure energy preservation
    /// 
    /// The resulting filter is sensitive to oscillations at frequency ω, making it
    /// ideal for detecting specific frequency components in the signal.
    /// 
    /// The complex nature of the filter allows it to capture both amplitude and phase
    /// information, which is crucial for many applications like audio processing,
    /// vibration analysis, and EEG analysis.
    /// </para>
    /// </remarks>
    public Vector<Complex<T>> GetWaveletCoefficients()
    {
        int length = 256; // Adjust based on desired precision
        var coeffs = new Complex<T>[length];
        T sum = _numOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T t = _numOps.Divide(_numOps.FromDouble(i - length / 2), _numOps.FromDouble(length / 4));
            T realPart = MathHelper.Cos(_numOps.Multiply(_omega, t));
            T imagPart = MathHelper.Sin(_numOps.Multiply(_omega, t));
            T envelope = _numOps.Exp(_numOps.Divide(_numOps.Negate(_numOps.Multiply(t, t)), _numOps.Multiply(_numOps.FromDouble(2), _numOps.Multiply(_sigma, _sigma))));

            coeffs[i] = new Complex<T>(
                _numOps.Multiply(envelope, realPart),
                _numOps.Multiply(envelope, imagPart)
            );
            sum = _numOps.Add(sum, _numOps.Abs(coeffs[i].Real));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(
                _numOps.Divide(coeffs[i].Real, sum),
                _numOps.Divide(coeffs[i].Imaginary, sum)
            );
        }

        return new Vector<Complex<T>>(coeffs);
    }

    /// <summary>
    /// Performs convolution of a complex input signal with a complex filter.
    /// </summary>
    /// <param name="input">The complex input signal vector.</param>
    /// <param name="kernel">The complex filter vector.</param>
    /// <returns>The convolved complex signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Convolution is a mathematical operation that combines two functions to produce a third function.
    /// In signal processing, it's like sliding a filter over a signal and calculating a weighted sum
    /// at each position.
    /// 
    /// For complex signals, convolution works similarly but involves complex multiplication:
    /// 
    /// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    /// 
    /// The process works like this:
    /// 1. For each position in the output:
    ///    a. Center the filter at that position
    ///    b. Multiply each filter coefficient by the corresponding signal value (using complex multiplication)
    ///    c. Sum these products to get the output value at that position
    /// 
    /// This implementation produces a result with length equal to input.length + kernel.length - 1,
    /// which is the full convolution without truncation. This ensures that no information is lost
    /// at the boundaries.
    /// 
    /// Complex convolution is essential for processing signals that contain phase information,
    /// such as in communications, radar, audio processing, and many other fields.
    /// </para>
    /// </remarks>
    private Vector<Complex<T>> Convolve(Vector<Complex<T>> input, Vector<Complex<T>> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new Complex<T>[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            Complex<T> sum = new(_numOps.Zero, _numOps.Zero);
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum += input[i - j] * kernel[j];
                }
            }

            result[i] = sum;
        }

        return new Vector<Complex<T>>(result);
    }

    /// <summary>
    /// Downsamples a complex signal by keeping only every nth sample.
    /// </summary>
    /// <param name="input">The complex input signal vector.</param>
    /// <param name="factor">The downsampling factor.</param>
    /// <returns>The downsampled complex signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Downsampling reduces the sampling rate of a signal by keeping only a subset of the samples.
    /// 
    /// For example, with a factor of 2:
    /// - Original signal: [10+2i, 20+3i, 30+4i, 40+5i, 50+6i, 60+7i]
    /// - Downsampled signal: [10+2i, 30+4i, 50+6i]
    /// 
    /// This method keeps every nth sample (where n is the factor) and discards the rest.
    /// In wavelet decomposition, downsampling by 2 is standard after filtering, because:
    /// 
    /// 1. The filters have already removed frequency components that would cause aliasing
    /// 2. It reduces the data size by half at each decomposition level
    /// 3. It ensures that the total size of approximation and detail coefficients equals the input size
    /// 
    /// For complex signals, both the real and imaginary parts are downsampled together,
    /// preserving the phase relationship between them.
    /// 
    /// Downsampling is crucial for the efficiency of wavelet transforms, especially for
    /// multi-level decomposition and compression applications.
    /// </para>
    /// </remarks>
    private Vector<Complex<T>> Downsample(Vector<Complex<T>> input, int factor)
    {
        int newLength = input.Length / factor;
        var result = new Complex<T>[newLength];

        for (int i = 0; i < newLength; i++)
        {
            result[i] = input[i * factor];
        }

        return new Vector<Complex<T>>(result);
    }
}