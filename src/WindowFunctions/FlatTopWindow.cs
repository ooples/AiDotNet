namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Flat Top window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Flat Top window is a specialized window function designed for amplitude accuracy in spectral analysis.
/// It uses a weighted sum of cosine terms:
/// w(n) = 1.0 - 1.93 * cos(2pn/(N-1)) + 1.29 * cos(4pn/(N-1)) - 0.388 * cos(6pn/(N-1)) + 0.028 * cos(8pn/(N-1))
/// where n is the sample index and N is the window size.
/// The Flat Top window has superior amplitude accuracy but poorer frequency resolution compared to other windows.
/// </para>
/// <para><b>For Beginners:</b> A window function is a mathematical tool that helps analyze signals more accurately.
/// 
/// The Flat Top window:
/// - Has a unique shape with a flat middle section and steep sides
/// - Is specifically designed for accurate measurement of signal amplitudes
/// - Sacrifices frequency precision for amplitude precision
/// 
/// Imagine you're trying to measure the height of people in a crowd. Some window functions might 
/// give you a good idea of how many people are of each height (frequency resolution), but the 
/// Flat Top window specializes in telling you exactly how tall each person is (amplitude accuracy). 
/// This makes it ideal for calibration, testing, and situations where you need to know the exact 
/// strength of a signal component rather than just detecting its presence.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class FlatTopWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="FlatTopWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Flat Top window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public FlatTopWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Flat Top window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Flat Top window function formula:
    /// w(n) = 1.0 - 1.93 * cos(2pn/(N-1)) + 1.29 * cos(4pn/(N-1)) - 0.388 * cos(6pn/(N-1)) + 0.028 * cos(8pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Flat Top formula
    /// - The resulting values form a specialized shape with a flat central region
    /// - The calculation uses several cosine terms to create the precise shape needed
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps you measure the true strength of frequency components in your signal.
    /// This is especially important in applications like instrument calibration, sound level measurement,
    /// or any situation where you need to know exactly how strong a signal is at different frequencies.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.Multiply(_numOps.FromDouble(1.0), _numOps.One);
            T term2 = _numOps.Multiply(_numOps.FromDouble(1.93), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(1.29), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term4 = _numOps.Multiply(_numOps.FromDouble(0.388), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(6 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term5 = _numOps.Multiply(_numOps.FromDouble(0.028), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(8 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            window[n] = _numOps.Subtract(_numOps.Subtract(_numOps.Add(_numOps.Subtract(term1, term2), term3), term4), term5);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Flat Top window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.FlatTop;
}
