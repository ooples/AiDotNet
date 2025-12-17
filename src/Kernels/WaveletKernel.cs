global using AiDotNet.WaveletFunctions;

namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Wavelet kernel for measuring similarity between data points using wavelet functions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Wavelet kernel uses wavelet functions to measure similarity between data points. Wavelets are
/// wave-like oscillations that start at zero, increase, and then decrease back to zero. They are
/// particularly useful for analyzing signals at different scales.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Wavelet kernel is special because it uses wavelet functions, which are like little waves that can
/// detect patterns at different scales or resolutions in your data.
/// </para>
/// <para>
/// Think of it like this: If you're looking at a beach from far away, you might see big waves, but as you
/// zoom in, you'll see smaller ripples too. Wavelets can help analyze both the big waves and small ripples
/// in your data. The Wavelet kernel uses this property to measure similarity between data points.
/// </para>
/// <para>
/// The formula for the Wavelet kernel is:
/// k(x, y) = ? h((x_i - y_i)/a) * vc
/// where:
/// - h is the wavelet function (like the Mexican Hat wavelet)
/// - a is a dilation parameter that controls the width of the wavelet
/// - c is a scaling parameter
/// - ? means multiply all the results together for each dimension i
/// </para>
/// <para>
/// Common uses include:
/// - Signal processing and time series analysis
/// - Image processing and computer vision
/// - Data with multi-scale patterns
/// - Feature extraction at different resolutions
/// </para>
/// </remarks>
public class WaveletKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The dilation parameter that controls the width of the wavelet.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This parameter controls how "stretched" or "compressed" the wavelet is.
    /// A larger value makes the wavelet wider, which means it can detect broader patterns.
    /// A smaller value makes the wavelet narrower, which means it can detect finer details.
    /// </remarks>
    private readonly T _a;

    /// <summary>
    /// The scaling parameter that affects the overall magnitude of the kernel value.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This parameter controls the overall "strength" or "importance" of the
    /// kernel value. A larger value will increase the magnitude of the similarity measure.
    /// </remarks>
    private readonly T _c;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that allows the kernel to perform mathematical
    /// operations regardless of what numeric type (like double, float, decimal) you're using.
    /// You don't need to interact with this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The wavelet function used to calculate similarity.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the specific wavelet function that will be used to measure similarity.
    /// By default, it's the Mexican Hat wavelet, which looks like a small wave with a peak in the
    /// middle and two smaller dips on either side. Different wavelet functions can capture different
    /// types of patterns in your data.
    /// </remarks>
    private readonly IWaveletFunction<T> _waveletFunction;

    /// <summary>
    /// Initializes a new instance of the Wavelet kernel with the specified parameters.
    /// </summary>
    /// <param name="waveletFunction">
    /// The wavelet function to use. If not specified, defaults to the Mexican Hat wavelet.
    /// </param>
    /// <param name="a">
    /// The dilation parameter that controls the width of the wavelet. If not specified, defaults to 1.0.
    /// </param>
    /// <param name="c">
    /// The scaling parameter that affects the overall magnitude. If not specified, defaults to 1.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Wavelet kernel for use. You can optionally
    /// provide a specific wavelet function and values for the parameters a and c.
    /// </para>
    /// <para>
    /// If you don't specify these values, the kernel will use:
    /// - The Mexican Hat wavelet (which looks like a small wave with a peak in the middle)
    /// - a = 1.0 (standard width)
    /// - c = 1.0 (standard scaling)
    /// </para>
    /// <para>
    /// When might you want to adjust these parameters?
    /// - Change the wavelet function when you want to detect different types of patterns
    /// - Use a smaller 'a' value when you want to detect finer details in your data
    /// - Use a larger 'a' value when you want to detect broader patterns
    /// - Adjust 'c' to control the overall magnitude of the similarity values
    /// </para>
    /// <para>
    /// The Wavelet kernel is particularly useful for:
    /// - Data with patterns at different scales (like images or signals)
    /// - Time series data with both short-term and long-term patterns
    /// - Feature extraction at multiple resolutions
    /// </para>
    /// </remarks>
    public WaveletKernel(IWaveletFunction<T>? waveletFunction = null, T? a = default, T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _a = a ?? _numOps.One;
        _c = c ?? _numOps.One;
        _waveletFunction = waveletFunction ?? new MexicanHatWavelet<T>();
    }

    /// <summary>
    /// Calculates the Wavelet kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Wavelet kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. For each dimension (feature) in the vectors:
    ///    a. Finding the difference between the corresponding values
    ///    b. Scaling this difference by dividing by parameter 'a'
    ///    c. Applying the wavelet function to this scaled difference
    ///    d. Multiplying by the square root of parameter 'c'
    /// 2. Multiplying all these individual results together
    /// </para>
    /// <para>
    /// The result is a similarity measure that captures patterns at the scale determined by
    /// parameter 'a'. The closer the value is to 1, the more similar the vectors are.
    /// </para>
    /// <para>
    /// What makes this kernel special is its ability to detect patterns at different scales,
    /// which is particularly useful for data with multi-scale features like images, signals,
    /// or time series.
    /// </para>
    /// <para>
    /// For example, if you're analyzing stock market data, the Wavelet kernel could help
    /// identify both short-term fluctuations and long-term trends by using different values
    /// of parameter 'a'.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T product = _numOps.One;
        for (int i = 0; i < x1.Length; i++)
        {
            T diff = _numOps.Subtract(x1[i], x2[i]);
            T scaledDiff = _numOps.Divide(diff, _a);
            product = _numOps.Multiply(product, _numOps.Multiply(_waveletFunction.Calculate(scaledDiff), _numOps.Sqrt(_c)));
        }

        return product;
    }
}
