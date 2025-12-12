namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Base class for all wavelet function implementations providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides shared infrastructure for wavelet function implementations,
/// including numeric operations support. All wavelet functions in the library should inherit
/// from this base class to ensure consistent behavior and reduce code duplication.
/// </para>
/// <para><b>For Beginners:</b> This is a foundation class that all wavelet types build upon.
///
/// Think of this base class like a blueprint that ensures all wavelets have:
/// - Access to mathematical operations (addition, multiplication, etc.)
/// - A consistent structure that makes them work together
/// - Shared utilities that every wavelet needs
///
/// When you create a new wavelet type, you inherit from this class to get all
/// these common features automatically, then just implement what makes your
/// wavelet unique.
/// </para>
/// </remarks>
public abstract class WaveletFunctionBase<T> : IWaveletFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
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
    /// Calculates the wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the wavelet function at the given input point.
    /// Each wavelet type has its own unique shape and mathematical definition.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what value the wavelet has at a specific point.
    ///
    /// Different wavelet functions have different shapes. When you ask for the value
    /// at a specific x position, this method returns the y-value (height) of the
    /// wavelet function at that point.
    /// </para>
    /// </remarks>
    public abstract T Calculate(T x);

    /// <summary>
    /// Decomposes an input signal using the wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the wavelet transform, which decomposes the input signal into
    /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into two parts.
    ///
    /// When decomposing a signal:
    /// - The approximation captures the overall trend or "big picture"
    /// - The detail captures the sharp changes and fine features
    ///
    /// This separation is useful for noise removal, compression, and feature detection.
    /// </para>
    /// </remarks>
    public abstract (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input);

    /// <summary>
    /// Gets the scaling coefficients used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients.</returns>
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
    /// </para>
    /// </remarks>
    public abstract Vector<T> GetScalingCoefficients();

    /// <summary>
    /// Gets the wavelet coefficients used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients.</returns>
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
    /// </para>
    /// </remarks>
    public abstract Vector<T> GetWaveletCoefficients();
}
