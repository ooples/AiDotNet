namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Hanning window function (also known as Hann window) for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Hanning window is a popular window function that provides good frequency resolution and 
/// reduced spectral leakage. It is defined by the equation:
/// w(n) = 0.5 * (1 - cos(2pn/(N-1)))
/// where n is the sample index and N is the window size. The Hanning window reaches exactly zero 
/// at both ends, which makes it particularly useful for analyzing periodic signals.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Hanning window:
/// - Creates a smooth bell-shaped curve that starts and ends exactly at zero
/// - Has a simple mathematical form that makes it easy to implement
/// - Provides a good balance between frequency resolution and spectral leakage
/// 
/// Think of it like looking through a window with edges that gradually fade to black. When analyzing 
/// frequencies in a signal (like audio), the Hanning window helps reduce the artifacts that occur 
/// when you're only looking at a portion of a continuous signal. It's particularly good for analyzing 
/// sounds like musical notes or any other signals that repeat over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HanningWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="HanningWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Hanning window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public HanningWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Hanning window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Hanning window function formula:
    /// w(n) = 0.5 * (1 - cos(2pn/(N-1)))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Hanning formula
    /// - The values start at 0, rise smoothly to 1 in the middle, and fall back to 0 at the end
    /// - This creates a gentle tapering effect that minimizes distortion in frequency analysis
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps you analyze frequencies with less artificial noise. This is particularly 
    /// useful in audio processing, vibration analysis, and other applications where you need to extract 
    /// frequency information from a segment of a longer signal.
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
            window[n] = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Subtract(_numOps.One,
                MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1))))));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Hanning window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Hanning;
}
