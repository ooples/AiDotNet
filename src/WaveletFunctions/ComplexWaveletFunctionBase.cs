namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Base class for all complex wavelet function implementations providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for the real and imaginary parts (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides shared infrastructure for complex wavelet function implementations,
/// including numeric operations support for both the underlying type T and complex operations.
/// All complex wavelet functions in the library should inherit from this base class to ensure
/// consistent behavior and reduce code duplication.
/// </para>
/// <para><b>For Beginners:</b> This is a foundation class that all complex wavelets build upon.
///
/// Think of this base class like a blueprint that ensures all complex wavelets have:
/// - Access to mathematical operations for both real and complex numbers
/// - A consistent structure that makes them work together
/// - Shared utilities that every complex wavelet needs
///
/// Complex wavelets work with complex numbers (numbers with real and imaginary parts),
/// which allows them to capture both amplitude and phase information in signals.
/// This is especially useful for analyzing oscillatory patterns.
/// </para>
/// </remarks>
public abstract class ComplexWaveletFunctionBase<T> : IWaveletFunction<Complex<T>>
{
    /// <summary>
    /// Provides mathematical operations for the underlying numeric type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of numeric operations that can work with the generic type T.
    /// It provides methods for basic arithmetic operations, comparisons, and conversions that are used
    /// throughout the wavelet calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that lets us do math with different number types.
    ///
    /// Because wavelet classes can work with different types of numbers (like float, double, or decimal),
    /// we need a special helper that knows how to:
    /// - Add, subtract, multiply, and divide these numbers
    /// - Compare them (greater than, less than, etc.)
    /// - Convert between different number formats
    ///
    /// This allows the wavelet code to work with whatever number type you choose,
    /// without having to write separate code for each number type.
    /// </para>
    /// </remarks>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Calculates the complex wavelet function value at the specified complex point.
    /// </summary>
    /// <param name="z">The complex input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated complex wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the complex wavelet function at the given complex input point.
    /// Each complex wavelet type has its own unique shape and mathematical definition.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what complex value the wavelet has at a specific point.
    ///
    /// Different complex wavelet functions have different shapes. When you ask for the value
    /// at a specific complex position, this method returns the complex value of the
    /// wavelet function at that point, containing both real and imaginary components.
    /// </para>
    /// </remarks>
    public abstract Complex<T> Calculate(Complex<T> z);

    /// <summary>
    /// Decomposes a complex input signal using the wavelet transform.
    /// </summary>
    /// <param name="input">The complex input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the wavelet transform for complex signals, decomposing the input into
    /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your complex data into two parts.
    ///
    /// When decomposing a complex signal:
    /// - The approximation captures the overall trend or "big picture"
    /// - The detail captures the sharp changes and fine features
    ///
    /// For complex signals, both the amplitude and phase information are preserved
    /// throughout the decomposition, which is important for applications like
    /// radar, communications, and audio processing.
    /// </para>
    /// </remarks>
    public abstract (Vector<Complex<T>> approximation, Vector<Complex<T>> detail) Decompose(Vector<Complex<T>> input);

    /// <summary>
    /// Gets the complex scaling coefficients used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the complex scaling coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients (also called low-pass filter coefficients)
    /// used to calculate the approximation (low-frequency) components of the signal during decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> These coefficients define how to calculate averages in the transform.
    ///
    /// The scaling coefficients:
    /// - Are used to create the "approximation" part when decomposing a signal
    /// - Act like an averaging or smoothing filter
    /// - Are specific to each type of wavelet
    /// - For complex wavelets, these are complex-valued coefficients
    /// </para>
    /// </remarks>
    public abstract Vector<Complex<T>> GetScalingCoefficients();

    /// <summary>
    /// Gets the complex wavelet coefficients used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the complex wavelet coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients (also called high-pass filter coefficients)
    /// used to calculate the detail (high-frequency) components of the signal during decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> These coefficients define how to detect details in the transform.
    ///
    /// The wavelet coefficients:
    /// - Are used to create the "detail" part when decomposing a signal
    /// - Act like a difference or edge-detection filter
    /// - Are specific to each type of wavelet
    /// - For complex wavelets, these capture both amplitude and phase variations
    /// </para>
    /// </remarks>
    public abstract Vector<Complex<T>> GetWaveletCoefficients();

    /// <summary>
    /// Performs convolution of a complex input signal with a complex filter.
    /// </summary>
    /// <param name="input">The complex input signal vector.</param>
    /// <param name="kernel">The complex filter vector.</param>
    /// <returns>The convolved complex signal vector.</returns>
    /// <remarks>
    /// <para>
    /// This protected method provides convolution functionality for derived complex wavelet classes.
    /// Convolution is a fundamental operation in wavelet transforms.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method slides a filter over a signal and computes weighted sums.
    ///
    /// For complex signals, convolution involves complex multiplication:
    /// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    ///
    /// The result captures how well the filter matches the signal at each position.
    /// </para>
    /// </remarks>
    protected Vector<Complex<T>> Convolve(Vector<Complex<T>> input, Vector<Complex<T>> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new Complex<T>[resultLength];
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        for (int i = 0; i < resultLength; i++)
        {
            Complex<T> sum = new Complex<T>(NumOps.Zero, NumOps.Zero);
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum = complexOps.Add(sum, complexOps.Multiply(input[i - j], kernel[j]));
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
    /// This protected method provides downsampling functionality for derived complex wavelet classes.
    /// Downsampling is used in wavelet decomposition to reduce the data size at each level.
    /// </para>
    /// <para><b>For Beginners:</b> This method reduces the sampling rate by keeping only every nth sample.
    ///
    /// For example, with factor 2:
    /// - Original: [a, b, c, d, e, f]
    /// - Result: [a, c, e]
    ///
    /// For complex signals, both real and imaginary parts are kept together.
    /// </para>
    /// </remarks>
    protected static Vector<Complex<T>> Downsample(Vector<Complex<T>> input, int factor)
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
