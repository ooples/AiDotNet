namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Gaussian wavelet function implementation for signal processing and analysis.
/// </summary>
/// <remarks>
/// <para>
/// The Gaussian wavelet is based on the Gaussian function and its derivatives. It provides
/// excellent localization in both time and frequency domains, making it useful for detecting
/// smooth changes and transitions in signals. This implementation uses the Gaussian function
/// for approximation coefficients and its first derivative for detail coefficients.
/// </para>
/// <para><b>For Beginners:</b> A Gaussian wavelet is a smooth bell-shaped curve that helps analyze data.
/// 
/// Think of the Gaussian wavelet like a special magnifying glass that:
/// - Can detect smooth transitions and gradual changes in your data
/// - Has a perfect balance between time and frequency precision
/// - Resembles the familiar "bell curve" shape used in statistics
/// 
/// This type of wavelet is especially good at finding gradual changes and smooth features
/// in signals like audio recordings, temperature measurements, or image brightness variations.
/// It's named after Carl Friedrich Gauss, a mathematician who first described the bell-shaped
/// curve that forms the foundation of this wavelet.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GaussianWavelet<T> : IWaveletFunction<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds an implementation of numeric operations that can work with the generic type T.
    /// It provides methods for basic arithmetic operations, exponentials, comparisons, and conversions
    /// that are used throughout the wavelet calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that lets us do math with different number types.
    /// 
    /// Because this class can work with different types of numbers (like float, double, or decimal),
    /// we need a special helper that knows how to:
    /// - Perform addition, subtraction, multiplication, and division
    /// - Calculate exponential functions (like those used in the bell curve)
    /// - Convert between different number formats
    /// 
    /// This allows the wavelet code to work with whatever number type you choose,
    /// without having to write separate code for each number type.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The standard deviation parameter that controls the width of the Gaussian wavelet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the width of the Gaussian function, affecting how localized
    /// the wavelet is in both time and frequency domains. A smaller sigma creates a narrower
    /// wavelet that provides better time localization but poorer frequency resolution, while
    /// a larger sigma does the opposite.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how wide or narrow the bell-shaped curve is.
    /// 
    /// Think of sigma like the width setting on a flashlight beam:
    /// - Smaller values create a narrower, more focused beam (good for pinpointing exact locations)
    /// - Larger values create a wider, more spread-out beam (good for seeing broader patterns)
    /// 
    /// If you're looking for precise locations of features in your data, use a smaller sigma.
    /// If you're more interested in the general frequency content, use a larger sigma.
    /// This represents the fundamental trade-off in signal analysis: you can't simultaneously have
    /// perfect precision in both where and what frequency.
    /// </para>
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Initializes a new instance of the <see cref="GaussianWavelet{T}"/> class with the specified sigma.
    /// </summary>
    /// <param name="sigma">The standard deviation that controls the width of the Gaussian wavelet. Defaults to 1.0.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the Gaussian wavelet with the specified sigma parameter,
    /// which controls the width of the Gaussian function and its derivative. The default value
    /// of 1.0 provides a balanced wavelet suitable for general-purpose analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Gaussian wavelet with your chosen width.
    /// 
    /// When creating a Gaussian wavelet:
    /// - The sigma parameter controls how wide the bell curve will be
    /// - The default value (1.0) works well for many common applications
    /// - You can experiment with different values to match your specific needs
    /// 
    /// For example, if you're analyzing an audio signal and want to detect short, quick changes
    /// in the sound, you might use a smaller sigma. If you're more interested in the overall
    /// frequency content of the sound, you might use a larger sigma.
    /// </para>
    /// </remarks>
    public GaussianWavelet(double sigma = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = _numOps.FromDouble(sigma);
    }

    /// <summary>
    /// Calculates the Gaussian function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the Gaussian function value.</param>
    /// <returns>The calculated Gaussian function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the Gaussian function (bell curve) at the given input point.
    /// The Gaussian function is defined as e^(-x�/2s�), where s is the standard deviation parameter.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the height of the bell curve at a specific point.
    /// 
    /// When you use this method:
    /// - You provide a point (x) on the horizontal axis
    /// - The method returns the height of the bell curve at that point
    /// - Points closer to the center (x=0) have higher values
    /// - Points far from the center have values close to zero
    /// 
    /// The Gaussian function creates the familiar bell-shaped curve used in statistics.
    /// It's highest at the center (x=0) and gradually decreases as you move away from the center.
    /// </para>
    /// </remarks>
    public T Calculate(T x)
    {
        T exponent = _numOps.Divide(_numOps.Multiply(x, x), _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(_sigma, _sigma)));
        return _numOps.Exp(_numOps.Negate(exponent));
    }

    /// <summary>
    /// Decomposes an input signal using the Gaussian wavelet and its derivative.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <remarks>
    /// <para>
    /// This method decomposes the input signal by applying the Gaussian function for approximation
    /// coefficients and its first derivative for detail coefficients. The approximation coefficients
    /// capture the smooth components of the signal, while the detail coefficients highlight transitions
    /// and changes.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into two complementary parts.
    /// 
    /// When decomposing a signal:
    /// - The approximation coefficients (using the Gaussian function) capture the smooth, slowly changing parts
    /// - The detail coefficients (using the derivative) capture the edges and rapid changes
    /// - The process centers the wavelet around each point in your data
    /// 
    /// This is like analyzing a landscape photo with two different filters:
    /// one that shows the general shapes and contours (approximation),
    /// and another that highlights the edges and boundaries (detail).
    /// </para>
    /// </remarks>
    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int size = input.Length;
        var approximation = new Vector<T>(size);
        var detail = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T x = _numOps.FromDouble(i - size / 2);
            T waveletValue = Calculate(x);
            T derivativeValue = CalculateDerivative(x);
            approximation[i] = _numOps.Multiply(waveletValue, input[i]);
            detail[i] = _numOps.Multiply(derivativeValue, input[i]);
        }

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients (Gaussian function) used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the Gaussian function sampled at 101 points centered around zero.
    /// These coefficients represent the smoothing function used to calculate the approximation
    /// coefficients during signal decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the bell curve values used in the transform.
    /// 
    /// The scaling coefficients:
    /// - Form a bell-shaped curve centered at the middle point
    /// - Are used to detect smooth, slowly varying features in your data
    /// - Act like a low-pass filter that keeps the general shape while removing details
    /// 
    /// These coefficients are sampled at 101 points to provide a good approximation of the
    /// continuous Gaussian function. The odd number ensures there's a center point exactly at zero.
    /// </para>
    /// </remarks>
    public Vector<T> GetScalingCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T x = _numOps.FromDouble(i - size / 2);
            coefficients[i] = Calculate(x);
        }

        return coefficients;
    }

    /// <summary>
    /// Gets the wavelet coefficients (derivative of Gaussian function) used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the first derivative of the Gaussian function sampled at 101 points
    /// centered around zero. These coefficients represent the detail-detecting function used to
    /// calculate the detail coefficients during signal decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the edge detection values used in the transform.
    /// 
    /// The wavelet coefficients:
    /// - Form an S-shaped curve that crosses zero at the middle point
    /// - Are strongest at the points of steepest slope on the Gaussian curve
    /// - Act like an edge detector that highlights changes and transitions
    /// 
    /// These coefficients are the first derivative (rate of change) of the Gaussian function.
    /// They're excellent at detecting points where your data is changing rapidly, such as edges,
    /// boundaries, or transitions between different states.
    /// </para>
    /// </remarks>
    public Vector<T> GetWaveletCoefficients()
    {
        int size = 101; // Odd number to have a center point
        var coefficients = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            T x = _numOps.FromDouble(i - size / 2);
            coefficients[i] = CalculateDerivative(x);
        }

        return coefficients;
    }

    /// <summary>
    /// Calculates the first derivative of the Gaussian function at the specified point.
    /// </summary>
    /// <param name="x">The position at which to calculate the derivative.</param>
    /// <returns>The calculated derivative value.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the first derivative of the Gaussian function at the given input point.
    /// The derivative is proportional to -x/s� multiplied by the Gaussian function itself. This
    /// derivative highlights points where the signal changes rapidly, making it useful for edge detection.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method calculates how quickly the Gaussian curve is changing at a specific point.
    /// 
    /// The derivative calculation:
    /// - Measures the slope of the Gaussian bell curve at point x
    /// - Is zero at the center (x=0) where the bell curve is flat at its peak
    /// - Is positive on one side of center and negative on the other
    /// - Has its maximum magnitude where the bell curve is steepest
    /// 
    /// This is why the derivative is excellent at detecting edges: it gives the strongest
    /// response exactly where your data is changing most rapidly, and the sign (positive or negative)
    /// tells you the direction of the change.
    /// </para>
    /// </remarks>
    private T CalculateDerivative(T x)
    {
        T gaussianValue = Calculate(x);
        T factor = _numOps.Divide(x, _numOps.Multiply(_sigma, _sigma));

        return _numOps.Multiply(_numOps.Negate(factor), gaussianValue);
    }
}