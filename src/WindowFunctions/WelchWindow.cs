namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Welch window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Welch window is a parabolic window function defined by the equation:
/// w(n) = 1 - ((n - N/2)/(N/2))²
/// where n is the sample index and N is (windowSize - 1). The Welch window has a parabolic shape
/// that reaches 1 at the center and 0 at both ends.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Welch window:
/// - Creates a smooth parabolic (bowl-shaped) curve
/// - Reaches a maximum value of 1 at the center and tapers to 0 at both ends
/// - Provides good frequency resolution with moderate side lobe suppression
/// 
/// Think of the Welch window like a smooth hill that gradually rises from the edges to the center.
/// Named after Peter Welch, who developed it for power spectrum estimation, this window is particularly 
/// useful when analyzing signals for their frequency content. The parabolic shape provides a good 
/// balance between preserving the main frequency components while reducing unwanted artifacts in the analysis.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class WelchWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="WelchWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Welch window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public WelchWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Welch window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Welch window function formula:
    /// w(n) = 1 - ((n - N/2)/(N/2))²
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates how far the position is from the center
    /// - It then uses this distance to calculate a parabolic value
    /// - The values form a smooth curve, with 1 at the center and 0 at the edges
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it emphasizes the central portion of your data while smoothly reducing the
    /// contribution from the edges. This helps reduce the artificial effects that can occur when
    /// analyzing a limited segment of a longer signal, especially in applications like power spectrum
    /// estimation where the Welch window was originally developed.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T N = _numOps.FromDouble(windowSize - 1);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term = _numOps.Divide(_numOps.Subtract(nT, _numOps.Divide(N, _numOps.FromDouble(2))), _numOps.Divide(N, _numOps.FromDouble(2)));
            window[n] = _numOps.Subtract(_numOps.One, _numOps.Multiply(term, term));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Welch window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Welch;
}
