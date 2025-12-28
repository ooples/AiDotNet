namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements a B-spline wavelet, which is a smooth wavelet constructed from B-spline functions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// B-spline wavelets are constructed from B-spline functions, which are piecewise polynomial functions
/// with compact support and maximum smoothness for a given support width. These wavelets offer excellent
/// smoothness properties and are particularly useful for applications requiring high regularity.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// B-spline wavelets are like smooth, bell-shaped curves that can be used to analyze signals at
/// different levels of detail.
/// 
/// Key features of B-spline wavelets:
/// - They're very smooth (no sharp corners or discontinuities)
/// - They have compact support (affect only a limited region)
/// - Their smoothness increases with the order
/// 
/// Think of B-splines as building blocks that can be combined to create smooth curves.
/// Higher-order B-splines are smoother but have wider support (affect more neighboring points).
/// 
/// These wavelets are particularly useful for:
/// - Signal smoothing and denoising
/// - Feature extraction where smoothness is important
/// - Applications where derivatives of the signal are analyzed
/// - Image processing tasks requiring high regularity
/// 
/// The order parameter lets you control the trade-off between smoothness and localization.
/// </para>
/// </remarks>
public class BSplineWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the B-spline used to construct the wavelet.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the BSplineWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the B-spline. Default is 3.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines the smoothness and support width of the B-spline wavelet.
    /// 
    /// - Order 1: Creates the simplest B-spline (a box function), which is not smooth
    /// - Order 2: Creates a triangular function (continuous but not smooth at the joints)
    /// - Order 3: Creates a quadratic B-spline (smooth with continuous first derivative)
    /// - Order 4: Creates a cubic B-spline (smooth with continuous second derivative)
    /// 
    /// Higher orders create smoother wavelets but with wider support, meaning they affect
    /// more neighboring points. This creates a trade-off:
    /// 
    /// - Lower orders: Better time localization but poorer frequency localization
    /// - Higher orders: Better frequency localization but poorer time localization
    /// 
    /// The default order of 3 provides a good balance for most applications, offering
    /// sufficient smoothness while maintaining reasonable localization.
    /// </para>
    /// </remarks>
    public BSplineWavelet(int order = 3)
    {
        _order = order;
    }

    /// <summary>
    /// Calculates the value of the B-spline wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the B-spline function at a specific point.
    /// 
    /// The B-spline function is defined recursively:
    /// - For order 0, it's a simple box function (1 between 0 and 1, 0 elsewhere)
    /// - For higher orders, it's a weighted average of two shifted lower-order B-splines
    /// 
    /// This recursive definition creates increasingly smooth functions as the order increases.
    /// For example:
    /// - Order 1: A box function (discontinuous)
    /// - Order 2: A triangle function (continuous but not smooth)
    /// - Order 3: A quadratic curve (smooth with continuous first derivative)
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        return BSpline(x, _order);
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the B-spline wavelet.
    /// </summary>
    /// <param name="input">The input signal vector to decompose.</param>
    /// <returns>A tuple containing the approximation coefficients and detail coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs one level of wavelet decomposition on your signal, splitting it into:
    /// 
    /// - Approximation coefficients: Represent the low-frequency components (the overall shape)
    /// - Detail coefficients: Represent the high-frequency components (the fine details)
    /// 
    /// The process works like this:
    /// 1. The input signal is convolved (filtered) with a low-pass filter to get the approximation
    /// 2. The input signal is convolved with a high-pass filter to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// This is similar to passing your signal through two different lenses:
    /// - A blurry lens to get the approximation (overall shape)
    /// - A sharpening lens to get the details (fine structure)
    /// 
    /// The downsampling step reduces the data size by half, which is efficient for
    /// multi-level decomposition and compression applications.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        var lowPass = GetDecompositionLowPassFilter();
        var highPass = GetDecompositionHighPassFilter();

        var approximation = Convolve(input, lowPass);
        var detail = Convolve(input, highPass);

        // Downsample by 2
        approximation = BSplineWavelet<T>.Downsample(approximation, 2);
        detail = BSplineWavelet<T>.Downsample(detail, 2);

        return (approximation, detail);
    }

    /// <summary>
    /// Reconstructs the original signal from approximation and detail coefficients.
    /// </summary>
    /// <param name="approximation">The approximation coefficients from decomposition.</param>
    /// <param name="detail">The detail coefficients from decomposition.</param>
    /// <returns>The reconstructed signal.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method reverses the decomposition process to get back the original signal.
    ///
    /// The reconstruction process for B-spline wavelets:
    /// 1. Upsample the approximation and detail coefficients by inserting zeros
    /// 2. Convolve with the time-reversed reconstruction filters
    /// 3. Add the results together
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        var lowPass = GetDecompositionLowPassFilter();
        var highPass = GetDecompositionHighPassFilter();

        // Upsample by 2
        var upsampledApprox = Upsample(approximation, 2);
        var upsampledDetail = Upsample(detail, 2);

        // Convolve with time-reversed filters
        var reconLow = ConvolveReversed(upsampledApprox, lowPass);
        var reconHigh = ConvolveReversed(upsampledDetail, highPass);

        // Add the results - ensure same length
        int outputLength = Math.Min(reconLow.Length, reconHigh.Length);
        var reconstructed = new Vector<T>(outputLength);
        for (int i = 0; i < outputLength; i++)
        {
            reconstructed[i] = NumOps.Add(reconLow[i], reconHigh[i]);
        }

        return reconstructed;
    }

    /// <summary>
    /// Upsamples a signal by inserting zeros.
    /// </summary>
    private static Vector<T> Upsample(Vector<T> input, int factor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[input.Length * factor];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = numOps.Zero;
        }
        for (int i = 0; i < input.Length; i++)
        {
            result[i * factor] = input[i];
        }
        return new Vector<T>(result);
    }

    /// <summary>
    /// Convolves with a time-reversed filter.
    /// </summary>
    private Vector<T> ConvolveReversed(Vector<T> input, Vector<T> filter)
    {
        int filterLen = filter.Length;
        var reversed = new T[filterLen];
        for (int i = 0; i < filterLen; i++)
        {
            reversed[i] = filter[filterLen - 1 - i];
        }
        return Convolve(input, new Vector<T>(reversed));
    }

    /// <summary>
    /// Gets the scaling function coefficients for the B-spline wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    ///
    /// For B-spline wavelets, these coefficients are derived directly from the B-spline
    /// function of the specified order. They create a low-pass filter that:
    /// - Lets through the low-frequency components of the signal
    /// - Blocks the high-frequency components
    /// - Has the smoothness properties of the B-spline
    /// 
    /// These coefficients are normalized to ensure that the energy of the signal is preserved
    /// during the wavelet transform.
    /// 
    /// The scaling coefficients determine how the wavelet will smooth the signal and
    /// capture its overall shape.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        return GetDecompositionLowPassFilter();
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the B-spline wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For B-spline wavelets, these coefficients are derived from the B-spline function
    /// of one order higher than the scaling function, with alternating signs. They create
    /// a high-pass filter that:
    /// - Lets through the high-frequency components of the signal
    /// - Blocks the low-frequency components
    /// - Complements the scaling function to cover the entire frequency spectrum
    /// 
    /// The alternating signs (positive, negative, positive, ...) create the oscillating
    /// nature that is characteristic of wavelets.
    /// 
    /// These coefficients determine how the wavelet will capture the details and
    /// fine structure of the signal.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        return GetDecompositionHighPassFilter();
    }

    /// <summary>
    /// Gets the low-pass filter coefficients used for decomposition.
    /// </summary>
    /// <returns>A vector of low-pass filter coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the coefficients for the low-pass filter used during decomposition.
    /// 
    /// The low-pass filter:
    /// - Lets through the low-frequency components of the signal
    /// - Blocks the high-frequency components
    /// - Produces the approximation coefficients
    /// 
    /// For B-spline wavelets, these coefficients are derived directly from the B-spline
    /// function of the specified order. The B-spline naturally acts as a smoothing function,
    /// making it an excellent basis for a low-pass filter.
    /// 
    /// The coefficients are normalized to ensure proper energy preservation during
    /// the decomposition process.
    /// </para>
    /// </remarks>
    private Vector<T> GetDecompositionLowPassFilter()
    {
        var coeffs = BSplineWavelet<T>.GetBSplineCoefficients(_order);
        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Gets the high-pass filter coefficients used for decomposition.
    /// </summary>
    /// <returns>A vector of high-pass filter coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the coefficients for the high-pass filter used during decomposition.
    /// 
    /// The high-pass filter:
    /// - Lets through the high-frequency components of the signal
    /// - Blocks the low-frequency components
    /// - Produces the detail coefficients
    /// 
    /// For B-spline wavelets, these coefficients are derived from the B-spline function
    /// of one order higher than the scaling function, with alternating signs. This creates
    /// a filter that responds to rapid changes in the signal.
    /// 
    /// The alternating signs (multiplying every other coefficient by -1) create the
    /// oscillating nature that allows the filter to detect high-frequency components.
    /// 
    /// The coefficients are normalized to ensure proper energy preservation during
    /// the decomposition process.
    /// </para>
    /// </remarks>
    private Vector<T> GetDecompositionHighPassFilter()
    {
        var coeffs = BSplineWavelet<T>.GetBSplineCoefficients(_order + 1);
        for (int i = 1; i < coeffs.Length; i += 2)
        {
            coeffs[i] = -coeffs[i];
        }

        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Generates B-spline coefficients for the specified order.
    /// </summary>
    /// <param name="order">The order of the B-spline.</param>
    /// <returns>An array of B-spline coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates the coefficients that define a B-spline of the specified order.
    /// 
    /// B-splines are defined recursively:
    /// - Order 1: Just two coefficients [1, 1] (representing a box function)
    /// - Higher orders: Calculated by a weighted average of the previous order's coefficients
    /// 
    /// The recursive calculation creates increasingly smooth functions as the order increases.
    /// Each higher order adds one more coefficient to the result.
    /// 
    /// For example:
    /// - Order 1: [1, 1]
    /// - Order 2: [1, 2, 1]
    /// - Order 3: [1, 3, 3, 1]
    /// - Order 4: [1, 4, 6, 4, 1]
    /// 
    /// These coefficients follow the pattern of binomial coefficients (Pascal's triangle),
    /// which is a mathematical property of B-splines.
    /// </para>
    /// </remarks>
    private static double[] GetBSplineCoefficients(int order)
    {
        if (order == 1)
        {
            return [1, 1];
        }

        var prev = BSplineWavelet<T>.GetBSplineCoefficients(order - 1);
        var result = new double[prev.Length + 1];

        for (int i = 0; i < result.Length; i++)
        {
            double t = (double)i / (order - 1);
            result[i] = (1 - t) * (i > 0 ? prev[i - 1] : 0) + t * (i < prev.Length ? prev[i] : 0);
        }

        return result;
    }

    /// <summary>
    /// Normalizes and converts an array of double coefficients to type T.
    /// </summary>
    /// <param name="coeffs">The array of double coefficients to normalize and convert.</param>
    /// <returns>A vector of normalized coefficients of type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This helper method performs two important tasks:
    /// 
    /// 1. Normalization: It adjusts the coefficients so that their energy (sum of squares)
    ///    equals 1. This ensures that the wavelet transform preserves the energy of the signal.
    /// 
    /// 2. Type conversion: It converts the double-precision coefficients to the generic
    ///    numeric type T used by the class.
    /// 
    /// Normalization is important because it ensures that the wavelet transform doesn't
    /// artificially amplify or reduce the signal's energy. It's a standard practice in
    /// wavelet design to ensure consistent results regardless of the specific wavelet used.
    /// 
    /// The type conversion allows the class to work with different numeric types (float, double, etc.)
    /// while maintaining the same coefficient values.
    /// </para>
    /// </remarks>
    private Vector<T> NormalizeAndConvert(double[] coeffs)
    {
        double normFactor = Math.Sqrt(coeffs.Sum(c => c * c));
        return new Vector<T>(coeffs.Select(c => NumOps.FromDouble(c / normFactor)).ToArray());
    }

    /// <summary>
    /// Performs convolution of an input signal with a filter.
    /// </summary>
    /// <param name="input">The input signal vector.</param>
    /// <param name="filter">The filter vector.</param>
    /// <returns>The convolved signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Convolution is a mathematical operation that combines two functions to produce a third function.
    /// In signal processing, it's like sliding a filter over a signal and calculating a weighted sum
    /// at each position.
    /// 
    /// The process works like this:
    /// 1. For each position in the input signal:
    ///    a. Center the filter at that position
    ///    b. Multiply each filter coefficient by the corresponding signal value
    ///    c. Sum these products to get the output value at that position
    /// 
    /// Convolution is fundamental to wavelet transforms and many other signal processing operations.
    /// It's how filters are applied to signals to extract specific frequency components.
    /// 
    /// This implementation handles boundary conditions by only including filter coefficients
    /// that overlap with the input signal, effectively using zero-padding at the boundaries.
    /// </para>
    /// </remarks>
    private Vector<T> Convolve(Vector<T> input, Vector<T> filter)
    {
        var result = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.Zero;
            for (int j = 0; j < filter.Length; j++)
            {
                int k = i - j + filter.Length / 2;
                if (k >= 0 && k < input.Length)
                {
                    result[i] = NumOps.Add(result[i], NumOps.Multiply(input[k], filter[j]));
                }
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Downsamples a signal by keeping only every nth sample.
    /// </summary>
    /// <param name="input">The input signal vector.</param>
    /// <param name="factor">The downsampling factor.</param>
    /// <returns>The downsampled signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Downsampling reduces the sampling rate of a signal by keeping only a subset of the samples.
    /// 
    /// For example, with a factor of 2:
    /// - Original signal: [10, 20, 30, 40, 50, 60]
    /// - Downsampled signal: [10, 30, 50]
    /// 
    /// This method keeps every nth sample (where n is the factor) and discards the rest.
    /// In wavelet decomposition, downsampling by 2 is standard after filtering, because:
    /// 
    /// 1. The filters have already removed frequency components that would cause aliasing
    /// 2. It reduces the data size by half at each decomposition level
    /// 3. It ensures that the total size of approximation and detail coefficients equals the input size
    /// 
    /// Downsampling is crucial for the efficiency of wavelet transforms, especially for
    /// multi-level decomposition and compression applications.
    /// </para>
    /// </remarks>
    private static Vector<T> Downsample(Vector<T> input, int factor)
    {
        return new Vector<T>([.. input.Where((_, i) => i % factor == 0)]);
    }

    /// <summary>
    /// Calculates the value of the B-spline function of order n at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the B-spline.</param>
    /// <param name="n">The order of the B-spline.</param>
    /// <returns>The value of the B-spline at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method implements the recursive definition of the B-spline function.
    /// 
    /// B-splines are defined recursively:
    /// - Order 0: A box function (1 between 0 and 1, 0 elsewhere)
    /// - Higher orders: A weighted average of two shifted lower-order B-splines
    /// 
    /// The recursive formula is:
    /// B_n(x) = (x/n) * B_{n-1}(x-1) + ((n+1-x)/n) * B_{n-1}(x)
    /// 
    /// This creates increasingly smooth functions as the order increases:
    /// - Order 0: Box function (discontinuous)
    /// - Order 1: Triangle function (continuous but not smooth)
    /// - Order 2: Quadratic curve (smooth with continuous first derivative)
    /// - Order 3: Cubic curve (smooth with continuous second derivative)
    /// 
    /// Each higher order adds one more degree of smoothness to the function.
    /// The support (non-zero region) of a B-spline of order n is [0, n+1].
    /// </para>
    /// </remarks>
    private T BSpline(T x, int n)
    {
        if (n == 0)
        {
            return NumOps.GreaterThanOrEquals(x, NumOps.Zero) && NumOps.LessThan(x, NumOps.One) ? NumOps.One : NumOps.Zero;
        }

        T term1 = NumOps.Multiply(NumOps.Divide(x, NumOps.FromDouble(n)), BSpline(NumOps.Subtract(x, NumOps.One), n - 1));
        T term2 = NumOps.Multiply(NumOps.Divide(NumOps.Subtract(NumOps.FromDouble(n + 1), x), NumOps.FromDouble(n)), BSpline(x, n - 1));

        return NumOps.Add(term1, term2);
    }
}
