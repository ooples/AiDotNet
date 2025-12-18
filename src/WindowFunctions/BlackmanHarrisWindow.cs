namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Blackman-Harris window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Blackman-Harris window is an advanced window function that provides excellent frequency 
/// resolution and spectral leakage suppression. It uses a weighted cosine series with four terms:
/// w(n) = 0.35875 - 0.48829 * cos(2pn/(N-1)) + 0.14128 * cos(4pn/(N-1)) - 0.01168 * cos(6pn/(N-1))
/// where n is the sample index and N is the window size.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter shape applied to your data.
/// 
/// The Blackman-Harris window:
/// - Creates a bell-shaped curve that's smoother than simpler windows
/// - Has very little "leakage" (unwanted frequencies) compared to other windows
/// - Is excellent for detecting signals that are close together in frequency
/// 
/// Think of it like a high-quality camera lens that gives you a clearer, more accurate picture
/// of the frequencies in your signal. It's especially useful in applications where you need to
/// distinguish between frequencies that are very close to each other, such as in audio analysis,
/// radar signal processing, or spectrum analysis.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BlackmanHarrisWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="BlackmanHarrisWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Blackman-Harris window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public BlackmanHarrisWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Blackman-Harris window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Blackman-Harris window function formula:
    /// w(n) = 0.35875 - 0.48829 * cos(2pn/(N-1)) + 0.14128 * cos(4pn/(N-1)) - 0.01168 * cos(6pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Blackman-Harris formula
    /// - The values form a bell-shaped curve (near 0 at the edges, higher in the middle)
    /// - The formula uses cosine functions with different frequencies to create a precise shape
    /// 
    /// The resulting window values can be applied to your signal (like audio or radar data)
    /// by multiplying each value in your signal by the corresponding window value.
    /// This helps analyze the frequencies in your signal with high precision.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.Multiply(_numOps.FromDouble(0.35875), _numOps.One);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.48829), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.14128), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term4 = _numOps.Multiply(_numOps.FromDouble(0.01168), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(6 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            window[n] = _numOps.Subtract(_numOps.Add(_numOps.Subtract(term1, term2), term3), term4);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Blackman-Harris window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.BlackmanHarris;
}
