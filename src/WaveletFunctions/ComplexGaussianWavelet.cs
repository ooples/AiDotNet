namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements a Complex Gaussian wavelet, which is based on the derivative of a Gaussian function
/// and can handle complex-valued signals.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Complex Gaussian wavelets are derived from the derivatives of the Gaussian function and extended
/// to the complex domain. They offer excellent time-frequency localization and are particularly useful
/// for analyzing signals with both amplitude and phase information.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Complex Gaussian wavelets are mathematical tools that can analyze both the "what" (amplitude)
/// and the "when" (phase) of signals simultaneously.
/// 
/// Key features of Complex Gaussian wavelets:
/// - They're based on the Gaussian function (bell curve) and its derivatives
/// - They can handle complex numbers (numbers with real and imaginary parts)
/// - They provide excellent time-frequency localization
/// - They're smooth and have good mathematical properties
/// 
/// Think of these wavelets as special lenses that can see both the size of a signal's components
/// (through the real part) and their timing or phase (through the imaginary part).
/// 
/// These wavelets are particularly useful for:
/// - Analyzing signals with phase information (like radar or sonar)
/// - Detecting oscillatory behavior in signals
/// - Applications where both magnitude and phase are important
/// - Signal processing tasks requiring complex analysis
/// 
/// The order parameter controls how many derivatives are taken of the Gaussian function,
/// affecting the wavelet's frequency selectivity.
/// </para>
/// </remarks>
public class ComplexGaussianWavelet<T> : ComplexWaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the Complex Gaussian wavelet (number of derivatives).
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the ComplexGaussianWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the wavelet (number of derivatives). Default is 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines how many times we differentiate the Gaussian function
    /// to create our wavelet.
    /// 
    /// Different orders create wavelets with different properties:
    /// 
    /// - Order 1: First derivative of Gaussian, simplest form
    ///   - Good general-purpose wavelet
    ///   - Balanced time-frequency localization
    /// 
    /// - Order 2: Second derivative of Gaussian (Mexican hat wavelet in real domain)
    ///   - Better frequency selectivity
    ///   - Slightly wider support in time
    /// 
    /// - Higher orders: More oscillations
    ///   - Better frequency selectivity
    ///   - Poorer time localization
    /// 
    /// Higher orders create wavelets that are more selective in frequency but less localized in time.
    /// The default order of 1 provides a good balance for most applications.
    /// </para>
    /// </remarks>
    public ComplexGaussianWavelet(int order = 1)
    {
        _order = order;
    }

    /// <summary>
    /// Calculates the value of the Complex Gaussian wavelet function at point z.
    /// </summary>
    /// <param name="z">The complex point at which to evaluate the wavelet function.</param>
    /// <returns>The complex value of the wavelet function at point z.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the Complex Gaussian wavelet at a specific point.
    /// 
    /// The Complex Gaussian wavelet is constructed as:
    /// ?(x) = C * H_n(x) * e^(-x²)
    /// 
    /// Where:
    /// - H_n(x) is the Hermite polynomial of order n
    /// - e^(-x²) is the Gaussian function
    /// - C is a normalization constant
    /// 
    /// For complex input z = x + iy, we evaluate this function at the real part x,
    /// and set the imaginary part of the result to zero for real input.
    /// 
    /// The Hermite polynomials add oscillations to the Gaussian function, creating
    /// a wavelet that can detect specific frequency components in a signal.
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override Complex<T> Calculate(Complex<T> z)
    {
        T x = z.Real;
        T y = z.Imaginary;

        T gaussianTerm = NumOps.Exp(NumOps.Negate(NumOps.Multiply(x, x)));
        T polynomialTerm = HermitePolynomial(x, _order);

        T real = NumOps.Multiply(gaussianTerm, polynomialTerm);
        T imag = NumOps.Zero; // The imaginary part is zero for real input

        return new Complex<T>(real, imag);
    }

    /// <summary>
    /// Decomposes a complex input signal into approximation and detail coefficients using the Complex Gaussian wavelet.
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
    /// 1. The input signal is convolved (filtered) with a low-pass filter to get the approximation
    /// 2. The input signal is convolved with a high-pass filter to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes Complex Gaussian wavelets special is that they can process complex signals,
    /// preserving both magnitude and phase information throughout the decomposition.
    /// 
    /// This is particularly useful for:
    /// - Radar and sonar signal processing
    /// - Communications systems
    /// - Any application where phase information is important
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which makes wavelet decomposition efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public override (Vector<Complex<T>> approximation, Vector<Complex<T>> detail) Decompose(Vector<Complex<T>> input)
    {
        var lowPass = GetScalingCoefficients();
        var highPass = GetWaveletCoefficients();

        var approximation = Convolve(input, lowPass);
        var detail = Convolve(input, highPass);

        // Downsample by 2
        approximation = ComplexGaussianWavelet<T>.Downsample(approximation, 2);
        detail = ComplexGaussianWavelet<T>.Downsample(detail, 2);

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Complex Gaussian wavelet.
    /// </summary>
    /// <returns>A vector of complex scaling function coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when sigma or error tolerance is not positive.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    /// 
    /// For Complex Gaussian wavelets, these coefficients are derived from a Gaussian function:
    /// 
    /// g(x) = e^(-x²/2)
    /// 
    /// The method:
    /// 1. Determines an appropriate length for the filter based on the desired accuracy
    /// 2. Samples the Gaussian function at discrete points
    /// 3. Normalizes the coefficients so they sum to 1
    /// 
    /// The width of the Gaussian (sigma) is set based on the wavelet order, with higher
    /// orders using wider Gaussians to capture their more oscillatory nature.
    /// 
    /// These coefficients create a low-pass filter that smooths the signal, capturing
    /// its overall shape while removing high-frequency details.
    /// </para>
    /// </remarks>
    public override Vector<Complex<T>> GetScalingCoefficients()
    {
        var errorTolerance = NumOps.FromDouble(1e-6);
        var sigma = NumOps.FromDouble(Math.Sqrt(_order));

        if (NumOps.LessThanOrEquals(sigma, NumOps.Zero))
            throw new ArgumentException("Sigma must be positive", nameof(sigma));

        if (NumOps.LessThanOrEquals(errorTolerance, NumOps.Zero))
            throw new ArgumentException("Error tolerance must be positive", nameof(errorTolerance));

        int length = DetermineAdaptiveLength(sigma, errorTolerance);
        var coeffs = new Complex<T>[length];

        T sum = NumOps.Zero;
        T centerIndex = NumOps.FromDouble((length - 1) / 2.0);

        for (int i = 0; i < length; i++)
        {
            T x = NumOps.Divide(NumOps.Subtract(NumOps.FromDouble(i), centerIndex), sigma);
            T gaussianValue = CalculateGaussianValue(x);
            coeffs[i] = new Complex<T>(gaussianValue, NumOps.Zero);
            sum = NumOps.Add(sum, gaussianValue);
        }

        // Normalize coefficients
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = new Complex<T>(NumOps.Divide(coeffs[i].Real, sum), NumOps.Zero);
        }

        return new Vector<Complex<T>>(coeffs);
    }

    /// <summary>
    /// Determines the appropriate length for the filter based on the desired accuracy.
    /// </summary>
    /// <param name="sigma">The standard deviation of the Gaussian function.</param>
    /// <param name="errorTolerance">The error tolerance that determines when to truncate the filter.</param>
    /// <returns>The appropriate length for the filter.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This helper method determines how long our filter needs to be to achieve the desired accuracy.
    /// 
    /// Since the Gaussian function extends to infinity but gets very small as you move away from
    /// the center, we need to decide where to cut it off. This method:
    /// 
    /// 1. Starts with a small filter length
    /// 2. Calculates the value of the Gaussian at the edge of the filter
    /// 3. If this value is larger than our error tolerance, increases the length
    /// 4. Repeats until the Gaussian value at the edge is smaller than our tolerance
    /// 
    /// This adaptive approach ensures that:
    /// - The filter is long enough to capture the significant parts of the Gaussian
    /// - We don't waste computation on coefficients that are effectively zero
    /// - The filter length automatically adjusts based on the wavelet order
    /// 
    /// The method ensures an odd length for symmetry, which is important for maintaining
    /// the phase properties of the wavelet transform.
    /// </para>
    /// </remarks>
    private int DetermineAdaptiveLength(T sigma, T errorTolerance)
    {
        int length = 1;
        T x = NumOps.Zero;
        T gaussianValue;

        do
        {
            length += 2; // Ensure odd length for symmetry
            x = NumOps.Divide(NumOps.FromDouble(length / 2), sigma);
            gaussianValue = CalculateGaussianValue(x);
        } while (NumOps.GreaterThan(gaussianValue, errorTolerance));

        return length;
    }

    /// <summary>
    /// Calculates the value of the Gaussian function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the Gaussian function.</param>
    /// <returns>The value of the Gaussian function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This helper method calculates the value of a Gaussian function (bell curve) at a specific point.
    /// 
    /// The Gaussian function is defined as:
    /// g(x) = e^(-x²/2)
    /// 
    /// This function has several important properties:
    /// - It's symmetric around x=0 (bell-shaped)
    /// - It reaches its maximum value of 1 at x=0
    /// - It approaches zero as x moves away from zero in either direction
    /// - It's infinitely differentiable (very smooth)
    /// 
    /// The Gaussian function is fundamental to the Complex Gaussian wavelet and provides
    /// the smooth envelope that gives the wavelet its good time-frequency localization properties.
    /// </para>
    /// </remarks>
    private T CalculateGaussianValue(T x)
    {
        return NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.Multiply(x, x), NumOps.FromDouble(2))));
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Complex Gaussian wavelet.
    /// </summary>
    /// <returns>A vector of complex wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For Complex Gaussian wavelets, these coefficients combine a Gaussian envelope
    /// with sine and cosine functions to create a complex-valued filter:
    /// 
    /// ?(x) = e^(-x²/2) * (cos(x) + i*sin(x))
    /// 
    /// Where:
    /// - e^(-x²/2) is the Gaussian envelope
    /// - cos(x) becomes the real part of the coefficient
    /// - sin(x) becomes the imaginary part of the coefficient
    /// - i is the imaginary unit
    /// 
    /// This creates a filter that:
    /// - Is sensitive to oscillations in the signal
    /// - Can detect both magnitude and phase information
    /// - Has good localization in both time and frequency
    /// 
    /// The length of the filter is proportional to the order, ensuring that higher-order
    /// wavelets (which have more oscillations) have sufficient support.
    /// </para>
    /// </remarks>
    public override Vector<Complex<T>> GetWaveletCoefficients()
    {
        int length = 10 * _order;
        var coeffs = new Complex<T>[length];
        T sigma = NumOps.FromDouble(Math.Sqrt(_order));

        for (int i = 0; i < length; i++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(i - length / 2), sigma);
            T gaussianValue = NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.Multiply(x, x), NumOps.FromDouble(2))));
            T sinValue = MathHelper.Sin(x);
            T cosValue = MathHelper.Cos(x);

            coeffs[i] = new Complex<T>(
                NumOps.Multiply(gaussianValue, cosValue),
                NumOps.Multiply(gaussianValue, sinValue)
            );
        }

        return new Vector<Complex<T>>(coeffs);
    }

    /// <summary>
    /// Calculates the Hermite polynomial of order n at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the Hermite polynomial.</param>
    /// <param name="n">The order of the Hermite polynomial.</param>
    /// <returns>The value of the Hermite polynomial at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Hermite polynomials are a sequence of orthogonal polynomials that play an important role
    /// in probability, physics, and wavelet theory.
    /// 
    /// The first few Hermite polynomials are:
    /// - H0(x) = 1
    /// - H1(x) = 2x
    /// - H2(x) = 4x² - 2
    /// - H3(x) = 8x³ - 12x
    /// 
    /// This method calculates these polynomials using a recurrence relation:
    /// H_{n+1}(x) = 2x·H_n(x) - 2n·H_{n-1}(x)
    /// 
    /// Starting with the known values for H0 and H1, it iteratively builds up to the
    /// desired order n.
    /// 
    /// In the context of Complex Gaussian wavelets, Hermite polynomials add oscillations
    /// to the Gaussian envelope, creating wavelets with different frequency characteristics.
    /// Higher-order polynomials create wavelets with more oscillations, suitable for
    /// detecting higher-frequency components in signals.
    /// </para>
    /// </remarks>
    private T HermitePolynomial(T x, int n)
    {
        if (n == 0) return NumOps.One;
        if (n == 1) return NumOps.Multiply(NumOps.FromDouble(2), x);

        T h0 = NumOps.One;
        T h1 = NumOps.Multiply(NumOps.FromDouble(2), x);

        for (int i = 2; i <= n; i++)
        {
            T hi = NumOps.Subtract(NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(2), x), h1), NumOps.Multiply(NumOps.FromDouble(2 * (i - 1)), h0));
            h0 = h1;
            h1 = hi;
        }

        return h1;
    }
}
