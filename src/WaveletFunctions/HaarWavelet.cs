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
public class HaarWavelet<T> : WaveletFunctionBase<T>
{

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
    public override T Calculate(T x)
    {
        if (NumOps.GreaterThanOrEquals(x, NumOps.Zero) && NumOps.LessThan(x, NumOps.FromDouble(0.5)))
            return NumOps.One;
        if (NumOps.GreaterThanOrEquals(x, NumOps.FromDouble(0.5)) && NumOps.LessThan(x, NumOps.One))
            return NumOps.FromDouble(-1);
        return NumOps.Zero;
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
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Haar wavelet decomposition.");
        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);
        for (int i = 0; i < halfLength; i++)
        {
            T sum = NumOps.Add(input[2 * i], input[2 * i + 1]);
            T diff = NumOps.Subtract(input[2 * i], input[2 * i + 1]);
            approximation[i] = NumOps.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(2)), sum);
            detail[i] = NumOps.Multiply(NumOps.FromDouble(1.0 / Math.Sqrt(2)), diff);
        }
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
    /// For the Haar wavelet, reconstruction is the inverse of decomposition:
    /// - For each pair of coefficients (approx[i], detail[i]):
    ///   - reconstructed[2*i] = (approx[i] + detail[i]) / sqrt(2)
    ///   - reconstructed[2*i+1] = (approx[i] - detail[i]) / sqrt(2)
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        int outputLength = approximation.Length * 2;
        var reconstructed = new Vector<T>(outputLength);
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(2));

        for (int i = 0; i < approximation.Length; i++)
        {
            T sum = NumOps.Add(approximation[i], detail[i]);
            T diff = NumOps.Subtract(approximation[i], detail[i]);
            reconstructed[2 * i] = NumOps.Multiply(scale, sum);
            reconstructed[2 * i + 1] = NumOps.Multiply(scale, diff);
        }

        return reconstructed;
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
    public override Vector<T> GetScalingCoefficients()
    {
        return new Vector<T>(new T[]
        {
            NumOps.FromDouble(1.0 / Math.Sqrt(2)),
            NumOps.FromDouble(1.0 / Math.Sqrt(2))
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
    public override Vector<T> GetWaveletCoefficients()
    {
        return new Vector<T>(new T[]
        {
            NumOps.FromDouble(1.0 / Math.Sqrt(2)),
            NumOps.FromDouble(-1.0 / Math.Sqrt(2))
        });
    }
}
