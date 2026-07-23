namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Bartlett window function (triangular window) for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Bartlett window is a triangular window function that provides a simple approach to smoothing signals.
/// It is defined by the equation: w(n) = 1 - |2(n - (N-1)/2)/(N-1)|
/// where n is the sample index and N is the window size.
/// The Bartlett window has a value of 0 at both endpoints and a value of 1 at the center.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that smooths out a signal.
/// 
/// The Bartlett window specifically creates a triangle shape:
/// - It starts at 0 at the beginning
/// - Increases to 1 at the middle
/// - Decreases back to 0 at the end
/// 
/// This window is useful when you need to:
/// - Analyze frequencies in audio or other signals
/// - Reduce sharp transitions at the edges of your data
/// - Apply a simple, computationally efficient smoothing effect
/// 
/// For example, if you're analyzing sound and want to focus on a specific time segment,
/// the Bartlett window helps blend the edges smoothly instead of creating an abrupt cutoff.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BartlettWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="BartlettWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Bartlett window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public BartlettWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Bartlett window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Bartlett window function formula:
    /// w(n) = 1 - |2(n - (N-1)/2)/(N-1)|
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the actual window values based on the size you need.
    /// 
    /// For each position in the window:
    /// - A value is calculated using the Bartlett formula
    /// - The values form a triangle shape (0 at the edges, 1 in the middle)
    /// - These values will be multiplied by your data to apply the window effect
    /// 
    /// For example, a Bartlett window of size 5 would create values like [0, 0.5, 1, 0.5, 0]
    /// which create a triangular shape when plotted.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T N = _numOps.FromDouble(windowSize - 1);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = _numOps.Subtract(_numOps.One, _numOps.Abs(_numOps.Multiply(_numOps.FromDouble(2), _numOps.Divide(_numOps.Subtract(nT, _numOps.Divide(N, _numOps.FromDouble(2))), N))));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Bartlett window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Bartlett;
}
