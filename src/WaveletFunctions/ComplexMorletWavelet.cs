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
public class ComplexMorletWavelet<T> : ComplexWaveletFunctionBase<T>
{
    /// <summary>
    /// The central frequency of the wavelet.
    /// </summary>
    private readonly T _omega;

    /// <summary>
    /// The bandwidth parameter controlling the width of the Gaussian window.
    /// </summary>
    private readonly T _sigma;

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
    /// 1. Omega (?): The central frequency of the wavelet
    ///    - Higher values look for higher-frequency oscillations in your data
    ///    - Lower values look for lower-frequency oscillations
    ///    - Think of this as tuning which "musical note" you're looking for
    ///    - Default value of 5.0 is a good general-purpose setting
    /// 
    /// 2. Sigma (s): The bandwidth parameter
    ///    - Controls the width of the Gaussian window
    ///    - Affects the trade-off between time and frequency precision
    ///    - Smaller values: Better time localization, poorer frequency resolution
    ///    - Larger values: Better frequency resolution, poorer time localization
    ///    - Default value of 1.0 provides a balanced trade-off
    /// 
    /// The relationship between these parameters determines the wavelet's properties:
    /// - The product ?·s should be > 5 to ensure admissibility (a mathematical requirement)
    /// - The default values (?=5, s=1) satisfy this condition
    /// 
    /// You might adjust these parameters when:
    /// - Looking for specific frequency components (adjust ?)
    /// - Needing better time precision or frequency precision (adjust s)
    /// </para>
    /// </remarks>
    public ComplexMorletWavelet(double omega = 5.0, double sigma = 1.0)
    {
        _omega = NumOps.FromDouble(omega);
        _sigma = NumOps.FromDouble(sigma);
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
    /// ?(t) = (e^(i?t) · e^(-t²/(2s²)))
    /// 
    /// Which can be broken down into:
    /// - e^(i?t) = cos(?t) + i·sin(?t): The complex exponential (oscillating part)
    /// - e^(-t²/(2s²)): The Gaussian envelope (bell-shaped curve)
    /// 
    /// For a complex input z = x + iy, this function:
    /// 1. Calculates the Gaussian envelope based on the distance from the origin
    /// 2. Multiplies it by the cosine (for the real part) and sine (for the imaginary part)
    /// 3. Returns the resulting complex number
    /// 
    /// The result is a localized wave packet that oscillates at frequency ? within
    /// a Gaussian envelope of width controlled by s.
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T expTerm = NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.Add(NumOps.Multiply(x, x), NumOps.Multiply(y, y)), NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Multiply(_sigma, _sigma)))));
        T cosTerm = MathHelper.Cos(NumOps.Multiply(_omega, x));
        T sinTerm = MathHelper.Sin(NumOps.Multiply(_omega, x));

        T real = NumOps.Multiply(expTerm, cosTerm);
        T imag = NumOps.Multiply(expTerm, sinTerm);

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
    public override (Vector<Complex<T>> approximation, Vector<Complex<T>> detail) Decompose(Vector<Complex<T>> input)
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
    /// sinc(x) = sin(px)/(px)
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
    public override Vector<Complex<T>> GetScalingCoefficients()
    {
        int length = 64; // Adjust as needed
        var coeffs = new Complex<T>[length];
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(i - length / 2), NumOps.FromDouble(length / 4));
            T value = NumOps.Equals(x, NumOps.Zero)
                ? NumOps.One
                : NumOps.Divide(MathHelper.Sin(NumOps.Divide(MathHelper.Pi<T>(), x)), NumOps.Multiply(MathHelper.Pi<T>(), x));
            coeffs[i] = new Complex<T>(value, NumOps.Zero);
            sum = NumOps.Add(sum, NumOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(NumOps.Divide(coeffs[i].Real, sum), NumOps.Zero);
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
    /// ?(t) = e^(i?t) · e^(-t²/(2s²))
    /// 
    /// This method:
    /// 1. Creates a discretized Complex Morlet wavelet of specified length
    /// 2. The real part is the Gaussian-modulated cosine: e^(-t²/(2s²)) · cos(?t)
    /// 3. The imaginary part is the Gaussian-modulated sine: e^(-t²/(2s²)) · sin(?t)
    /// 4. Normalizes the coefficients to ensure energy preservation
    /// 
    /// The resulting filter is sensitive to oscillations at frequency ?, making it
    /// ideal for detecting specific frequency components in the signal.
    /// 
    /// The complex nature of the filter allows it to capture both amplitude and phase
    /// information, which is crucial for many applications like audio processing,
    /// vibration analysis, and EEG analysis.
    /// </para>
    /// </remarks>
    public override Vector<Complex<T>> GetWaveletCoefficients()
    {
        int length = 256; // Adjust based on desired precision
        var coeffs = new Complex<T>[length];
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T t = NumOps.Divide(NumOps.FromDouble(i - length / 2), NumOps.FromDouble(length / 4));
            T realPart = MathHelper.Cos(NumOps.Multiply(_omega, t));
            T imagPart = MathHelper.Sin(NumOps.Multiply(_omega, t));
            T envelope = NumOps.Exp(NumOps.Divide(NumOps.Negate(NumOps.Multiply(t, t)), NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(_sigma, _sigma))));

            coeffs[i] = new Complex<T>(
                NumOps.Multiply(envelope, realPart),
                NumOps.Multiply(envelope, imagPart)
            );
            sum = NumOps.Add(sum, NumOps.Abs(coeffs[i].Real));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(
                NumOps.Divide(coeffs[i].Real, sum),
                NumOps.Divide(coeffs[i].Imaginary, sum)
            );
        }

        return new Vector<Complex<T>>(coeffs);
    }
}
