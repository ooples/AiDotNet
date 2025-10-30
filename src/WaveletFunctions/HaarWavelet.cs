namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Haar wavelet function implementation for signal processing and analysis.
/// </summary>
/// <remarks>
/// <para>
/// The Haar wavelet is the simplest possible wavelet, resembling a step function. It is
/// discontinuous and represents the same wavelet as the Daubechies wavelet with one vanishing moment.
/// This implementation provides methods for calculating wavelet values and decomposing signals
/// using the Haar wavelet transform.
/// </para>
/// <para><b>For Beginners:</b> The Haar wavelet is like a digital "on/off" switch for analyzing data.
/// 
/// Think of the Haar wavelet as the simplest possible pattern-matching tool:
/// - It's a square wave that is +1 for half its width and -1 for the other half
/// - It's excellent at detecting sudden changes or edges in your data
/// - It's the oldest and simplest type of wavelet, discovered in 1909
/// 
/// The Haar wavelet is particularly good at finding abrupt transitions in your data,
/// like the edge between a light and dark area in an image or a sudden change in a sound wave.
/// It's widely used in image compression, feature detection, and as a teaching tool for wavelet concepts.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HaarWavelet<T> : IWaveletFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of numeric operations that can work with the generic type T.
    /// It provides methods for basic arithmetic operations, comparisons, and conversions that are used
    /// throughout the wavelet calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that lets us do math with different number types.
    /// 
    /// Because this class can work with different types of numbers (like float, double, or decimal),
    /// we need a special helper that knows how to:
    /// - Add, subtract, multiply, and divide these numbers
    /// - Compare them (greater than, less than, etc.)
    /// - Convert between different number formats
    /// 
    /// This allows the wavelet code to work with whatever number type you choose,
    /// without having to write separate code for each number type.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="HaarWavelet{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Haar wavelet and sets up the numeric operations helper
    /// for the specified numeric type T. Unlike other wavelets, the Haar wavelet doesn't require
    /// any parameters as it has a fixed, simple definition.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Haar wavelet for use.
    /// 
    /// When creating a Haar wavelet:
    /// - No parameters are needed because the Haar wavelet has a fixed shape
    /// - It's the simplest wavelet, defined by just a few values
    /// - It's ready to use immediately after creation
    /// 
    /// The simplicity of the Haar wavelet makes it a good starting point for understanding
    /// wavelets in general, before moving on to more complex wavelet types.
    /// </para>
    /// </remarks>
    public HaarWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Haar wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated Haar wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Haar wavelet function at the given input point.
    /// The Haar wavelet is defined as +1 for x in [0,0.5), -1 for x in [0.5,1), and 0 elsewhere.
    /// This step-like function is useful for detecting abrupt changes in signals.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what value the Haar wavelet has at a specific point.
    /// 
    /// The Haar wavelet has a very simple pattern:
    /// - If x is between 0 (inclusive) and 0.5 (exclusive), it returns +1
    /// - If x is between 0.5 (inclusive) and 1 (exclusive), it returns -1
    /// - For all other values of x, it returns 0
    /// 
    /// This creates a simple step function that goes up, then down, and is zero everywhere else.
    /// This shape is particularly good at detecting sudden changes or edges in your data.
    /// </para>
    /// </remarks>
    public T Calculate(T x)
    {
        if (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.FromDouble(0.5)))
            return _numOps.One;
        if (_numOps.GreaterThanOrEquals(x, _numOps.FromDouble(0.5)) && _numOps.LessThan(x, _numOps.One))
            return _numOps.FromDouble(-1);
        return _numOps.Zero;
    }

    /// <summary>
    /// Decomposes an input signal using the Haar wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <exception cref="ArgumentException">Thrown when the input vector length is not even.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the Haar wavelet transform, which decomposes the input signal into
    /// approximation coefficients (averages) and detail coefficients (differences). For each pair of
    /// adjacent values in the input, it computes the scaled average and difference. The approximation
    /// coefficients represent the low-frequency components of the signal, while the detail coefficients
    /// represent the high-frequency components.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into "averages" and "differences".
    /// 
    /// When decomposing a signal with the Haar wavelet:
    /// - The method takes every pair of adjacent values in your data
    /// - For each pair, it calculates their average (for the approximation)
    /// - For each pair, it also calculates their difference (for the detail)
    /// - These values are scaled by a factor (1/v2) to preserve energy
    /// 
    /// The approximation tells you about the overall trend of your data (like a blurry version),
    /// while the detail captures the sharp changes and edges (like the fine details).
    /// This process reduces your data to half its original size while preserving its essential information.
    /// </para>
    /// </remarks>
    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Haar wavelet decomposition.");
        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);
        for (int i = 0; i < halfLength; i++)
        {
            T sum = _numOps.Add(input[2 * i], input[2 * i + 1]);
            T diff = _numOps.Subtract(input[2 * i], input[2 * i + 1]);
            approximation[i] = _numOps.Multiply(_numOps.FromDouble(1.0 / Math.Sqrt(2)), sum);
            detail[i] = _numOps.Multiply(_numOps.FromDouble(1.0 / Math.Sqrt(2)), diff);
        }
        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients used in the Haar wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients [1/v2, 1/v2].</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients used in the Haar wavelet transform, which are
    /// [1/v2, 1/v2]. These coefficients are used to calculate the approximation (low-frequency)
    /// components of the signal during decomposition. The factor 1/v2 ensures energy conservation
    /// during the transform.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the values used to calculate averages in the transform.
    /// 
    /// The scaling coefficients for the Haar wavelet:
    /// - Are simply [1/v2, 1/v2]
    /// - Act like a simple averaging filter (both values are the same)
    /// - Are used to create the "approximation" part when decomposing a signal
    /// 
    /// The factor 1/v2 (approximately 0.7071) is important for mathematical reasons:
    /// it ensures that the energy of the signal is preserved during transformation,
    /// which makes the transform reversible and maintains correct signal properties.
    /// </para>
    /// </remarks>
    public Vector<T> GetScalingCoefficients()
    {
        return new Vector<T>(new T[]
        {
            _numOps.FromDouble(1.0 / Math.Sqrt(2)),
            _numOps.FromDouble(1.0 / Math.Sqrt(2))
        });
    }

    /// <summary>
    /// Gets the wavelet coefficients used in the Haar wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients [1/v2, -1/v2].</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients used in the Haar wavelet transform, which are
    /// [1/v2, -1/v2]. These coefficients are used to calculate the detail (high-frequency)
    /// components of the signal during decomposition. The factor 1/v2 ensures energy conservation
    /// during the transform, and the opposite signs detect differences between adjacent values.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the values used to calculate differences in the transform.
    /// 
    /// The wavelet coefficients for the Haar wavelet:
    /// - Are simply [1/v2, -1/v2]
    /// - Act like a difference detector (one positive, one negative)
    /// - Are used to create the "detail" part when decomposing a signal
    /// 
    /// The opposite signs mean this filter finds differences between adjacent values,
    /// which is why it's so good at detecting edges and sudden changes in your data.
    /// The factor 1/v2 (approximately 0.7071) ensures mathematical consistency 
    /// with the scaling coefficients.
    /// </para>
    /// </remarks>
    public Vector<T> GetWaveletCoefficients()
    {
        return new Vector<T>(new T[]
        {
            _numOps.FromDouble(1.0 / Math.Sqrt(2)),
            _numOps.FromDouble(-1.0 / Math.Sqrt(2))
        });
    }
}