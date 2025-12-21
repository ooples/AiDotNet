namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Rectangular window function (also known as the boxcar or Dirichlet window) for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Rectangular window is the simplest window function, defined by the equation:
/// w(n) = 1 for all n
/// where n is the sample index. Unlike other window functions, the Rectangular window does not
/// taper or modify the signal at the edges, which can lead to spectral leakage in frequency analysis.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Rectangular window:
/// - Is the simplest window function - it has a value of 1 for all points
/// - Does not taper the signal at the edges like other windows do
/// - Preserves the original amplitude of the signal at all points
/// 
/// Think of it like looking at your data through a simple rectangular frame with sharp edges.
/// While this window preserves the most signal energy, the abrupt transitions at the edges can
/// create artifacts in frequency analysis. It's like suddenly turning a speaker on and off instead
/// of gradually adjusting the volume - the sudden change creates additional frequency components
/// that weren't in the original signal.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RectangularWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="RectangularWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Rectangular window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public RectangularWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Rectangular window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Rectangular window function by creating a vector filled with 1s.
    /// Since the Rectangular window has a constant value of 1 for all points, the implementation
    /// simply creates a vector of the specified size and fills it with the value 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For the Rectangular window:
    /// - Every position gets the same value: 1.0
    /// - This means the window doesn't change your data at all when applied
    /// - It's the equivalent of not using a window function
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value, which is always 1), the data remains unchanged. This preserves all the original
    /// information but can lead to spectral leakage - unwanted frequency artifacts that appear when
    /// analyzing a limited segment of a longer signal. The Rectangular window is often used as a baseline
    /// for comparing other window functions or when processing needs to preserve all signal energy.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        return Vector<T>.CreateDefault(windowSize, _numOps.One);
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Rectangular window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Rectangular;
}
