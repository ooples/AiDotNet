namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Triangular window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Triangular window is a simple window function that creates a triangular shape. It is defined by the equation:
/// w(n) = 1 - |2n - L|/L
/// where n is the sample index and L is (windowSize - 1). The Triangular window provides moderate spectral leakage
/// reduction compared to the Rectangular window.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Triangular window:
/// - Creates a simple triangle shape that peaks in the middle
/// - Starts at 0, rises linearly to 1 at the center, then decreases linearly back to 0
/// - Provides a basic improvement over the Rectangular window for reducing spectral leakage
/// 
/// Think of it like a ramp that gradually increases and then decreases - like a mountain peak.
/// When analyzing frequencies in a signal, this gradual transition helps reduce some of the
/// unwanted artifacts that occur with the Rectangular window's abrupt edges. It's like turning
/// a volume knob smoothly up and down instead of suddenly flipping a switch, which creates
/// fewer distortions in the analysis.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TriangularWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="TriangularWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Triangular window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public TriangularWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Triangular window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Triangular window function formula:
    /// w(n) = 1 - |2n - L|/L
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Triangular window formula
    /// - The values start at 0, increase linearly to 1 at the center, then decrease linearly back to 0
    /// - This creates a simple triangle shape
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps reduce some of the unwanted artifacts in frequency analysis. This window
    /// is particularly useful when you need a simple window function that provides better spectral 
    /// characteristics than the Rectangular window but with less computational complexity than more 
    /// sophisticated windows.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T L = _numOps.FromDouble(windowSize - 1);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = _numOps.Subtract(_numOps.One, _numOps.Abs(_numOps.Divide(_numOps.Subtract(_numOps.Multiply(nT, _numOps.FromDouble(2)), L), L)));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Triangular window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Triangular;
}
