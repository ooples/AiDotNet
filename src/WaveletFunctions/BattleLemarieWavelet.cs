namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements the Battle-Lemarie wavelet function, which is a smooth, orthogonal wavelet based on B-splines.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Battle-Lemarie wavelet is a family of orthogonal wavelets constructed from B-spline functions.
/// It offers good frequency localization and smoothness properties, making it useful for signal analysis
/// where both time and frequency resolution are important.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// A wavelet is a mathematical function used to divide a signal into different frequency components.
/// Think of it as a special kind of lens that lets you examine different details of a signal.
/// 
/// The Battle-Lemarie wavelet is particularly useful because:
/// - It's smooth, which helps avoid artifacts in signal processing
/// - It has good localization in both time and frequency domains
/// - It can be adjusted (via the order parameter) to balance between time and frequency resolution
/// 
/// You might use this wavelet for:
/// - Image compression
/// - Noise reduction
/// - Feature extraction from signals
/// - Analyzing signals with different scales of detail
/// 
/// The higher the order, the smoother the wavelet, but also the wider its support (meaning it
/// considers more neighboring points when analyzing a signal).
/// </para>
/// </remarks>
public class BattleLemarieWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the B-spline used to construct the wavelet.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the BattleLemarieWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the B-spline used to construct the wavelet. Default is 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines the smoothness and support width of the wavelet.
    /// 
    /// - Order 1: Creates the simplest Battle-Lemarie wavelet, which is less smooth but has narrower support
    /// - Higher orders: Create smoother wavelets with wider support
    /// 
    /// Think of the order like the "softness" of the lens you're using to analyze your signal:
    /// - Lower order: Sharper but more sensitive to noise (like a high-contrast lens)
    /// - Higher order: Smoother but less precise in localization (like a soft-focus lens)
    /// 
    /// For most applications, orders between 1 and 4 work well, balancing smoothness and computational efficiency.
    /// </para>
    /// </remarks>
    public BattleLemarieWavelet(int order = 1)
    {
        _order = order;
    }

    /// <summary>
    /// Calculates the value of the Battle-Lemarie wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the wavelet function at a specific point.
    /// 
    /// The Battle-Lemarie wavelet is constructed as a linear combination of shifted B-spline functions.
    /// For each point x, it:
    /// 1. Evaluates the B-spline function at x shifted by different integer values
    /// 2. Multiplies each shifted B-spline by an alternating sign (+1, -1, +1, ...)
    /// 3. Sums these values to get the final result
    /// 
    /// This creates a wavelet function that oscillates (has positive and negative parts)
    /// and has good localization properties.
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        T result = NumOps.Zero;
        for (int k = -_order; k <= _order; k++)
        {
            T term = BSpline(NumOps.Add(x, NumOps.FromDouble(k)));
            result = NumOps.Add(result, NumOps.Multiply(NumOps.FromDouble(Math.Pow(-1, k)), term));
        }

        return result;
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the Battle-Lemarie wavelet.
    /// </summary>
    /// <param name="input">The input signal vector to decompose.</param>
    /// <returns>A tuple containing the approximation coefficients and detail coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the input length is not even.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs one level of wavelet decomposition on your signal, splitting it into:
    /// 
    /// - Approximation coefficients: Represent the low-frequency components (the overall shape)
    /// - Detail coefficients: Represent the high-frequency components (the fine details)
    /// 
    /// The process works like this:
    /// 1. The input signal is convolved (filtered) with scaling coefficients to get approximation coefficients
    /// 2. The input signal is convolved with wavelet coefficients to get detail coefficients
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// This is similar to passing your signal through two different filters:
    /// - A low-pass filter to get the approximation (like blurring an image)
    /// - A high-pass filter to get the details (like edge detection in an image)
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which is why the input length must be even.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Battle-Lemarie wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        var scalingCoeffs = GetScalingCoefficients();
        var waveletCoeffs = GetWaveletCoefficients();

        for (int i = 0; i < halfLength; i++)
        {
            T approx = NumOps.Zero;
            T det = NumOps.Zero;

            for (int j = 0; j < scalingCoeffs.Length; j++)
            {
                int index = (2 * i + j) % input.Length;
                approx = NumOps.Add(approx, NumOps.Multiply(scalingCoeffs[j], input[index]));
                det = NumOps.Add(det, NumOps.Multiply(waveletCoeffs[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Battle-Lemarie wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    /// 
    /// This method calculates these coefficients using a sophisticated approach:
    /// 1. It works in the frequency domain using Fourier transforms
    /// 2. It constructs the B-spline function in the frequency domain
    /// 3. It ensures orthogonality (a mathematical property that makes wavelets useful)
    /// 4. It transforms back to the time domain using an inverse Fourier transform
    /// 5. It normalizes the coefficients to ensure proper scaling
    /// 
    /// The result is a set of coefficients that, when used as a filter, extract the
    /// low-frequency components of a signal while preserving important mathematical properties.
    /// 
    /// These coefficients are critical for the wavelet decomposition process and determine
    /// how the signal's energy is distributed between approximation and detail coefficients.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int order = _order;
        int numCoeffs = 4 * order + 1; // Increased support for better accuracy
        Vector<T> coeffs = new Vector<T>(numCoeffs);

        // Calculate Battle-Lemarie coefficients using Fourier transform
        int numSamples = 1024; // Number of samples for FFT, adjust as needed
        Vector<Complex<T>> fftInput = new Vector<Complex<T>>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            T omega = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * i), NumOps.FromDouble(1.0 / numSamples));
            Complex<T> bSplineFourier = BSplineFourier(omega, order);
            T denominator = NumOps.Sqrt(SumSquaredBSplineFourier(omega, order));
            fftInput[i] = new Complex<T>(NumOps.Divide(bSplineFourier.Real, denominator), NumOps.Divide(bSplineFourier.Imaginary, denominator));
        }

        // Perform inverse FFT
        Vector<Complex<T>> fftOutput = InverseFFT(fftInput);

        // Extract and normalize coefficients
        int center = numSamples / 2;
        for (int i = 0; i < numCoeffs; i++)
        {
            coeffs[i] = fftOutput[(center - numCoeffs / 2 + i + numSamples) % numSamples].Real;
        }

        // Normalize the coefficients
        T sum = coeffs.Sum();
        for (int i = 0; i < coeffs.Length; i++)
        {
            coeffs[i] = NumOps.Divide(coeffs[i], sum);
        }

        return coeffs;
    }

    /// <summary>
    /// Calculates the Fourier transform of the B-spline function at a given frequency.
    /// </summary>
    /// <param name="omega">The frequency at which to evaluate the Fourier transform.</param>
    /// <param name="order">The order of the B-spline.</param>
    /// <returns>The complex value of the Fourier transform at the specified frequency.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates how the B-spline function (the building block of the Battle-Lemarie wavelet)
    /// behaves in the frequency domain.
    /// 
    /// The Fourier transform converts a function from the time domain to the frequency domain,
    /// showing how much of each frequency is present in the original function.
    /// 
    /// For B-splines, the Fourier transform has a simple form: it's the product of sinc functions
    /// (sin(x)/x) raised to the power of the spline order.
    /// 
    /// This frequency-domain representation is used to ensure the wavelet has good mathematical
    /// properties like orthogonality, which makes it useful for signal analysis.
    /// </para>
    /// </remarks>
    private Complex<T> BSplineFourier(T omega, int order)
    {
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        Complex<T> result = new Complex<T>(NumOps.One, NumOps.Zero);
        for (int i = 0; i < order; i++)
        {
            T sinc = NumOps.Divide(MathHelper.Sin(NumOps.Divide(omega, NumOps.FromDouble(2))), NumOps.Divide(omega, NumOps.FromDouble(2)));
            result = complexOps.Multiply(result, new Complex<T>(sinc, NumOps.Zero));
        }

        return result;
    }

    /// <summary>
    /// Calculates the sum of squared Fourier transforms of shifted B-splines.
    /// </summary>
    /// <param name="omega">The frequency at which to evaluate.</param>
    /// <param name="order">The order of the B-spline.</param>
    /// <returns>The sum of squared Fourier transforms.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method is part of the orthogonalization process for creating the Battle-Lemarie wavelet.
    /// 
    /// To make a wavelet orthogonal (a desirable mathematical property), we need to normalize
    /// the B-spline function in the frequency domain. This requires calculating the sum of
    /// squared Fourier transforms of the B-spline shifted by different integer values.
    /// 
    /// The method:
    /// 1. Shifts the frequency by multiples of 2p
    /// 2. Calculates the B-spline Fourier transform at each shifted frequency
    /// 3. Squares the magnitude of each transform
    /// 4. Sums these squared magnitudes
    /// 
    /// This sum is then used as a normalization factor to ensure the wavelet has the
    /// orthogonality property, which makes it useful for decomposing signals into
    /// independent components.
    /// </para>
    /// </remarks>
    private T SumSquaredBSplineFourier(T omega, int order)
    {
        T sum = NumOps.Zero;
        for (int k = -order; k <= order; k++)
        {
            T shiftedOmega = NumOps.Add(omega, NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), NumOps.FromDouble(k)));
            Complex<T> bSpline = BSplineFourier(shiftedOmega, order);
            sum = NumOps.Add(sum, NumOps.Add(NumOps.Multiply(bSpline.Real, bSpline.Real), NumOps.Multiply(bSpline.Imaginary, bSpline.Imaginary)));
        }

        return sum;
    }

    /// <summary>
    /// Performs an inverse Fast Fourier Transform (FFT) on the input complex vector.
    /// </summary>
    /// <param name="input">The complex vector in the frequency domain.</param>
    /// <returns>The complex vector in the time domain.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The inverse Fast Fourier Transform (FFT) converts data from the frequency domain back to the time domain.
    /// 
    /// In the context of wavelet construction:
    /// - We design the wavelet in the frequency domain for certain mathematical properties
    /// - We then use the inverse FFT to convert it back to the time domain for practical use
    /// 
    /// This method implements a direct (non-optimized) version of the inverse FFT:
    /// 1. For each output point, it calculates a weighted sum of all input points
    /// 2. The weights are complex exponentials (cosine and sine functions)
    /// 3. The result is divided by the number of points to normalize
    /// 
    /// While not as efficient as specialized FFT algorithms, this implementation is
    /// straightforward and works well for the wavelet coefficient calculation.
    /// 
    /// The result is the time-domain representation of the wavelet's scaling function,
    /// which is used to extract the approximation coefficients during decomposition.
    /// </para>
    /// </remarks>
    private Vector<Complex<T>> InverseFFT(Vector<Complex<T>> input)
    {
        int n = input.Length;
        Vector<Complex<T>> output = new Vector<Complex<T>>(n);

        for (int k = 0; k < n; k++)
        {
            var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
            Complex<T> sum = new Complex<T>(NumOps.Zero, NumOps.Zero);
            for (int t = 0; t < n; t++)
            {
                T angle = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * t * k), NumOps.FromDouble(1.0 / n));
                Complex<T> exp = new Complex<T>(MathHelper.Cos(angle), MathHelper.Sin(angle));
                sum = complexOps.Add(sum, complexOps.Multiply(input[t], exp));
            }

            output[k] = new Complex<T>(NumOps.Divide(sum.Real, NumOps.FromDouble(n)), NumOps.Divide(sum.Imaginary, NumOps.FromDouble(n)));
        }

        return output;
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Battle-Lemarie wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// In the Battle-Lemarie wavelet, these coefficients are derived from the scaling coefficients
    /// using a simple relationship:
    /// 
    /// wavelet_coeff[i] = (-1)^i * scaling_coeff[L-1-i]
    /// 
    /// Where:
    /// - L is the length of the scaling coefficients
    /// - (-1)^i alternates between +1 and -1 as i changes
    /// - The order of the coefficients is reversed
    /// 
    /// This relationship, known as the "quadrature mirror filter" property, ensures that:
    /// - The wavelet filter captures frequencies that the scaling filter misses
    /// - Together, they cover the entire frequency spectrum without gaps
    /// - The decomposition preserves the energy of the signal
    /// 
    /// These wavelet coefficients are used in the decomposition process to extract
    /// the detail coefficients, which represent the fine structure of the signal.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        var scalingCoeffs = GetScalingCoefficients();
        int L = scalingCoeffs.Length;
        var waveletCoeffs = new T[L];

        for (int i = 0; i < L; i++)
        {
            waveletCoeffs[i] = NumOps.Multiply(NumOps.FromDouble(Math.Pow(-1, i)), scalingCoeffs[L - 1 - i]);
        }

        return new Vector<T>(waveletCoeffs);
    }

    /// <summary>
    /// Calculates the value of the B-spline function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the B-spline function.</param>
    /// <returns>The value of the B-spline function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// B-splines are the building blocks used to construct the Battle-Lemarie wavelet.
    /// They are smooth, bell-shaped functions with compact support (they're zero outside a finite interval).
    /// 
    /// This method implements a simple B-spline of order 2, which has the following properties:
    /// - It equals 1 when |x| < 0.5 (a flat top in the middle)
    /// - It transitions smoothly to 0 as |x| approaches 1.5
    /// - It equals 0 when |x| = 1.5
    /// 
    /// The transition region (0.5 < |x| < 1.5) follows a quadratic curve: 0.5 * (1.5 - |x|)²
    /// 
    /// This particular B-spline is chosen for its balance of smoothness and computational simplicity.
    /// Higher-order B-splines would be smoother but more complex to calculate.
    /// 
    /// The B-spline function is used in the Calculate method to construct the actual wavelet function.
    /// </para>
    /// </remarks>
    private T BSpline(T x)
    {
        T absX = NumOps.Abs(x);
        if (NumOps.LessThan(absX, NumOps.FromDouble(0.5)))
        {
            return NumOps.One;
        }
        else if (NumOps.LessThanOrEquals(absX, NumOps.FromDouble(1.5)))
        {
            T temp = NumOps.Subtract(NumOps.FromDouble(1.5), absX);
            return NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(temp, temp));
        }

        return NumOps.Zero;
    }
}
