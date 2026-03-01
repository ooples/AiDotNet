namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Hamming window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Hamming window is a widely used window function that provides good frequency resolution
/// and reduced spectral leakage. It is defined by the equation:
/// w(n) = 0.54 - 0.46 * cos(2pn/(N-1))
/// where n is the sample index and N is the window size. The Hamming window is optimized to
/// minimize the maximum sidelobe amplitude, making it particularly useful for spectral analysis.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Hamming window:
/// - Creates a bell-shaped curve that tapers to near-zero (but not exactly zero) at the edges
/// - Is specifically designed to reduce "spectral leakage" (unwanted frequency artifacts)
/// - Offers a good balance of spectral resolution and amplitude accuracy
/// 
/// Think of it like a pair of well-designed sunglasses that reduces glare while still providing
/// a clear view. When analyzing frequencies in a signal (like audio), the Hamming window helps
/// you see the true frequencies more clearly by reducing unwanted artifacts. It's named after
/// Richard Hamming, who developed it for telecommunications applications, and is one of the most
/// commonly used window functions because of its good all-around performance.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HammingWindow<T> : IWindowFunction<T>
{
    /// <summary>
    /// The numeric operations provider for performing calculations with type T.
    /// </summary>
    /// <remarks>
    /// This field stores an object that provides operations like addition, multiplication, and
    /// trigonometric functions for the generic numeric type T. It's essential for implementing
    /// the window function's mathematical formula across different numeric types.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="HammingWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Hamming window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public HammingWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Hamming window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Hamming window function formula:
    /// w(n) = 0.54 - 0.46 * cos(2pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Hamming formula
    /// - The values form a smooth curve that's highest in the middle (around 1.0)
    /// - The edges taper down to approximately 0.08, not quite reaching zero
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps you analyze frequencies more accurately. This is particularly useful
    /// in audio processing, communications, radar, and other applications where you need to identify
    /// frequency components in a signal with good precision.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);

        if (windowSize == 1)
        {
            window[0] = _numOps.One;
            return window;
        }

        for (int n = 0; n < windowSize; n++)
        {
            window[n] = _numOps.Add(_numOps.FromDouble(0.54), _numOps.Multiply(_numOps.FromDouble(-0.46),
                MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1))))));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Hamming window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Hamming;
}
