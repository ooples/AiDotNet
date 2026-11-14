namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements Coiflet wavelets, which are compactly supported wavelets with a high number of vanishing moments
/// for both the wavelet and scaling functions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Coiflet wavelets were designed by Ingrid Daubechies at the request of Ronald Coifman. They are distinguished
/// by having vanishing moments for both the wavelet and scaling functions, which makes them more symmetric
/// than Daubechies wavelets and better suited for certain applications.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Coiflet wavelets are special mathematical tools designed to analyze signals with particular properties.
/// 
/// Key features of Coiflet wavelets:
/// - They're nearly symmetric (more symmetric than Daubechies wavelets)
/// - They have vanishing moments for both the wavelet and scaling functions
/// - They have compact support (affect only a limited region)
/// 
/// "Vanishing moments" means the wavelet can ignore certain polynomial trends in the data.
/// For example, a wavelet with 3 vanishing moments will be "blind" to constant, linear, and
/// quadratic trends, allowing it to focus on more complex patterns.
/// 
/// These properties make Coiflet wavelets particularly useful for:
/// - Signal compression
/// - Feature extraction
/// - Numerical analysis
/// - Applications where symmetry is important
/// 
/// The order parameter (1-5) controls how many vanishing moments the wavelet has,
/// with higher orders providing more vanishing moments but wider support.
/// </para>
/// </remarks>
public class CoifletWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The coefficients of the Coiflet wavelet.
    /// </summary>
    private readonly T[] _coefficients;

    /// <summary>
    /// The order of the Coiflet wavelet.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the CoifletWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the Coiflet wavelet. Default is 2.</param>
    /// <exception cref="ArgumentException">Thrown when the order is less than 1 or greater than 5.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines the properties of the Coiflet wavelet.
    /// 
    /// For Coiflet wavelets, the order N (1-5) means:
    /// - The wavelet has 2N vanishing moments
    /// - The scaling function has 2N-1 vanishing moments
    /// - The support width is 6N-1
    /// 
    /// In practical terms:
    /// - Order 1: Simplest, with 2 vanishing moments, support width of 5
    /// - Order 2: More vanishing moments (4), support width of 11
    /// - Order 3: Even more vanishing moments (6), support width of 17
    /// - And so on...
    /// 
    /// Higher orders create wavelets that:
    /// - Can ignore more complex polynomial trends
    /// - Are smoother
    /// - Have wider support (affect more neighboring points)
    /// 
    /// The default order of 2 provides a good balance for most applications.
    /// </para>
    /// </remarks>
    public CoifletWavelet(int order = 2)
    {
        if (order < 1 || order > 5)
            throw new ArgumentException("Order must be between 1 and 5.", nameof(order));

        _order = order;
        double[] doubleCoefficients = CoifletWavelet<T>.GetCoifletCoefficients(order);
        _coefficients = [.. doubleCoefficients.Select(c => NumOps.FromDouble(c))];
    }

    /// <summary>
    /// Calculates the value of the Coiflet wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the Coiflet wavelet function at a specific point.
    /// 
    /// The Coiflet wavelet is constructed as a linear combination of shifted scaling functions:
    /// 1. It checks if the point is within the support of the wavelet (0 to 6*order-1)
    /// 2. For each coefficient, it applies the scaling function shifted to different positions
    /// 3. It multiplies each shifted function by the corresponding coefficient
    /// 4. It sums these weighted functions to get the final result
    /// 
    /// The scaling function itself is defined recursively through a two-scale relation,
    /// which makes exact calculation challenging. This implementation uses an approximation
    /// approach.
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        if (t < 0 || t > 6 * _order - 1)
            return NumOps.Zero;

        // Only the integer shift k = floor(t) puts t - k in [0, 1), so the sum over k has a single term.
        int k = (int)Math.Floor(t);
        if (k >= 6 * _order)
            return NumOps.Zero;

        return NumOps.Multiply(_coefficients[k], NumOps.FromDouble(ScalingFunction(t - k)));
    }

    /// <summary>
    /// Evaluates the scaling function at point t using a recursive approximation.
    /// </summary>
    /// <param name="t">The point at which to evaluate the scaling function.</param>
    /// <returns>The value of the scaling function at point t.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling function is the basic building block used to construct the wavelet.
    ///
    /// For Coiflet wavelets, the scaling function satisfies a two-scale relation:
    /// f(t) = S c_k f(2t-k)
    ///
    /// This is a recursive definition, which makes exact calculation challenging.
    /// This method implements a simple recursive approximation that:
    /// 1. Checks if the point is within the support [0,1]
    /// 2. If so, calculates the function value using the two-scale relation
    ///
    /// In practice, this recursive approach has limitations and would typically be
    /// replaced by more sophisticated numerical methods for accurate calculation.
    /// However, it illustrates the fundamental recursive nature of wavelet scaling functions.
    /// </para>
    /// </remarks>
    private double ScalingFunction(double t)
    {
        return ScalingFunctionRecursive(t, 0);
    }

    /// <summary>
    /// Recursive helper for scaling function with depth limit to prevent stack overflow.
    /// </summary>
    private double ScalingFunctionRecursive(double t, int depth)
    {
        // Base case: outside support
        if (t < 0 || t > 1)
            return 0;

        // Base case: max recursion depth reached - return approximation
        const int MaxDepth = 10;
        if (depth >= MaxDepth)
            return 1.0; // Return constant approximation at max depth

        double result = 0;
        for (int k = 0; k < 6 * _order; k++)
        {
            double shiftedT = 2 * t - k;
            // Only recurse if the shifted value is within support
            if (shiftedT >= 0 && shiftedT <= 1)
            {
                result += Convert.ToDouble(_coefficients[k]) * ScalingFunctionRecursive(shiftedT, depth + 1);
            }
        }

        return result;
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the Coiflet wavelet.
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
    /// What makes Coiflet wavelets special for this task is:
    /// - Their near symmetry reduces phase distortion
    /// - Their vanishing moments allow them to ignore polynomial trends
    /// - Their compact support makes computation efficient
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which makes wavelet decomposition efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        var lowPass = GetScalingCoefficients();
        var highPass = GetWaveletCoefficients();

        var approximation = Convolve(input, lowPass);
        var detail = Convolve(input, highPass);

        // Downsample by 2
        approximation = Downsample(approximation, 2);
        detail = Downsample(detail, 2);

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
    /// The reconstruction process for Coiflet wavelets:
    /// 1. Upsample the approximation and detail coefficients by inserting zeros
    /// 2. Convolve with the time-reversed reconstruction filters
    /// 3. Add the results together
    ///
    /// For orthogonal wavelets like Coiflets, perfect reconstruction is guaranteed when:
    /// - The filters satisfy the orthogonality conditions
    /// - The signal length is compatible with the filter length
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        var lowPass = GetScalingCoefficients();
        var highPass = GetWaveletCoefficients();

        // Upsample by 2
        var upsampledApprox = Upsample(approximation, 2);
        var upsampledDetail = Upsample(detail, 2);

        // Convolve with time-reversed filters
        var reconLow = ConvolveReversed(upsampledApprox, lowPass);
        var reconHigh = ConvolveReversed(upsampledDetail, highPass);

        // Add the results - ensure same length
        return WaveletEngineOps<T>.AddTruncated(reconLow, reconHigh);
    }

    /// <summary>
    /// Upsamples a signal by inserting zeros.
    /// </summary>
    private static Vector<T> Upsample(Vector<T> input, int factor)
    {
        return WaveletEngineOps<T>.Upsample(input, factor);
    }

    /// <summary>
    /// Convolves with a time-reversed filter.
    /// </summary>
    private Vector<T> ConvolveReversed(Vector<T> input, Vector<T> filter)
    {
        return Convolve(input, WaveletEngineOps<T>.Reverse(filter));
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Coiflet wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    ///
    /// For Coiflet wavelets, these coefficients:
    /// - Are carefully designed to have specific vanishing moments
    /// - Are nearly symmetric (more symmetric than Daubechies wavelets)
    /// - Create a low-pass filter that captures the overall shape of the signal
    /// 
    /// These coefficients are pre-calculated for each order and have been designed to
    /// satisfy specific mathematical properties that make Coiflet wavelets useful for
    /// signal analysis.
    /// 
    /// The coefficients are normalized to ensure proper energy preservation during
    /// the decomposition process.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        double[] coeffs = CoifletWavelet<T>.GetCoifletCoefficients(_order);
        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Coiflet wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For Coiflet wavelets, these coefficients are derived from the scaling coefficients using
    /// the quadrature mirror filter relationship:
    /// 
    /// g[n] = (-1)^n * h[L-1-n]
    /// 
    /// Where:
    /// - g[n] are the wavelet coefficients
    /// - h[n] are the scaling coefficients
    /// - L is the length of the filter
    /// - n is the index
    /// 
    /// This relationship ensures that:
    /// - The wavelet filter captures frequencies that the scaling filter misses
    /// - Together, they cover the entire frequency spectrum without gaps
    /// - The decomposition preserves the energy of the signal
    /// 
    /// The alternating signs ((-1)^n) create the oscillating nature that is characteristic of wavelets.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        double[] coeffs = CoifletWavelet<T>.GetCoifletCoefficients(_order);
        int n = coeffs.Length;
        double[] highPass = new double[n];
        for (int i = 0; i < n; i++)
        {
            highPass[i] = Math.Pow(-1, i) * coeffs[n - 1 - i];
        }

        return NormalizeAndConvert(highPass);
    }

    /// <summary>
    /// Gets the pre-calculated coefficients for the Coiflet wavelet of the specified order.
    /// </summary>
    /// <param name="order">The order of the Coiflet wavelet (1-5).</param>
    /// <returns>An array of Coiflet coefficients.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the order is not supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the pre-calculated coefficients for Coiflet wavelets of different orders.
    /// 
    /// These coefficients were derived by Ingrid Daubechies to satisfy specific mathematical
    /// properties, particularly having vanishing moments for both the wavelet and scaling functions.
    /// 
    /// The number of coefficients depends on the order:
    /// - Order 1: 6 coefficients
    /// - Order 2: 12 coefficients
    /// - Order 3: 18 coefficients
    /// - Order 4: 24 coefficients
    /// - Order 5: 30 coefficients
    /// 
    /// The general pattern is 6N coefficients for order N.
    /// 
    /// These coefficients are the foundation of the Coiflet wavelet transform and determine
    /// the wavelet's properties like smoothness, symmetry, and frequency response.
    /// </para>
    /// </remarks>
    private static double[] GetCoifletCoefficients(int order)
    {
        return order switch
        {
            1 => [
                    -0.015655728135791993, -0.07273261951252645, 0.3848648468648578,
                    0.8525720202116004, 0.3378976624574818, -0.07273261951252645
                ],
            2 => [
                    -0.000720549445520347, -0.0018232088709110323, 0.005611434819368834,
                    0.02368017194684777, -0.05943441864643109, -0.07648859907828076,
                    0.4170051844232391, 0.8127236354494135, 0.3861100668227629,
                    -0.0673725547237256, -0.04146493678687178, 0.01638733646320364
                ],
            3 => [
                    -3.459977319727278e-05, -7.0983302506379e-05, 0.0004662169598204029,
                    0.0011175187708306303, -0.0025745176881367972, -0.009007976136730624,
                    0.015880544863669452, 0.03455502757329774, -0.08230192710629983,
                    -0.07179982161915484, 0.42848347637737, 0.7937772226260872,
                    0.40517690240911824, -0.06112339000297255, -0.06577191128146936,
                    0.023452696142077168, 0.007782596425672746, -0.003793512864380802
                ],
            4 => [
                    -1.7849909144933469e-06, -3.259647940030751e-06, 3.1229861599195265e-05,
                    6.233885431278719e-05, -0.0002599743371222568, -0.0005890202246332165,
                    0.0012665610789256603, 0.0037514346971460866, -0.0056582838001308835,
                    -0.015211728187697211, 0.02508225333794961, 0.03933442260558915,
                    -0.09622042453595264, -0.06662747236681717, 0.43438603311435653,
                    0.7822389344242826, 0.41530842700068227, -0.05607731960356926,
                    -0.08126671024919373, 0.02668230466960483, 0.01606894713157503,
                    -0.007346167936268051, -0.001629492425226786, 0.000892313902537003
                ],
            5 => [
                    -9.604010112767894e-08, -1.6237995172048338e-07, 2.0612203985788783e-06,
                    3.7007277113394796e-06, -2.1270221672515614e-05, -4.12198619242655e-05,
                    0.00014035632812373243, 0.0003018579416682448, -0.0006375589261258812,
                    -0.0016616273039298788, 0.0024315754425382886, 0.006761520220620417,
                    -0.009159507338676163, -0.019758391600965465, 0.032674799467057355,
                    0.041287530472117834, -0.10556315130733723, -0.06203775157498196,
                    0.4379823066591634, 0.7742936228603274, 0.42157126673075435,
                    -0.052046670253554764, -0.09192158806008609, 0.028169744270532353,
                    0.023408322118927783, -0.010131584846900276, -0.00415931262757864,
                    0.0021782943778456947, 0.0003585777411617577, -0.000212081862067494
                ],
            _ => throw new ArgumentOutOfRangeException(nameof(order), "Unsupported Coiflet order."),
        };
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
        return new Vector<T>([.. coeffs.Select(c => NumOps.FromDouble(c / normFactor))]);
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
        // result[i] = sum_j input[i - j + L / 2] * filter[j] with zero padding
        return WaveletEngineOps<T>.CenteredConvolve(input, filter);
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
        // Keeps indices 0, factor, 2 * factor, ... (ceil(n / factor) samples).
        return WaveletEngineOps<T>.Decimate(input, factor, (input.Length + factor - 1) / factor);
    }
}
