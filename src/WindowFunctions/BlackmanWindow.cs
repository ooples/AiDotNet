namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Blackman window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Blackman window is a commonly used window function that provides good frequency resolution 
/// and reduced spectral leakage. It uses a weighted cosine series with three terms:
/// w(n) = 0.42 - 0.5 * cos(2pn/(N-1)) + 0.08 * cos(4pn/(N-1))
/// where n is the sample index and N is the window size.
/// </para>
/// <para><b>For Beginners:</b> A window function is a mathematical tool that helps analyze signals more accurately.
/// 
/// The Blackman window:
/// - Creates a smooth bell-shaped curve that's zero at both ends
/// - Provides a good balance between time and frequency resolution
/// - Reduces unwanted "spectral leakage" (false frequency readings)
/// 
/// Think of it like a magnifying glass with special properties - when you look at a signal through 
/// this "Blackman lens," you can more clearly see its true frequency components without as much 
/// distortion. It's commonly used in audio processing, vibration analysis, and other signal 
/// processing applications.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BlackmanWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="BlackmanWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Blackman window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public BlackmanWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Blackman window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Blackman window function formula:
    /// w(n) = 0.42 - 0.5 * cos(2pn/(N-1)) + 0.08 * cos(4pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Blackman formula
    /// - The values form a bell-shaped curve (zero at the edges, higher in the middle)
    /// - These values will later be multiplied by your signal data
    /// 
    /// When you apply this window to your data, it helps focus your analysis on the important 
    /// parts of your signal while smoothly reducing the influence of the edges, which can 
    /// cause measurement errors in frequency analysis.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.Multiply(_numOps.FromDouble(0.42), _numOps.One);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.5), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.08), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            window[n] = _numOps.Add(_numOps.Subtract(term1, term2), term3);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Blackman window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Blackman;
}
