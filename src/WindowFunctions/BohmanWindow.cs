namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Bohman window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Bohman window is a specialized window function that provides excellent spectral characteristics
/// with very low sidelobe levels. It's defined by a more complex formula compared to simpler windows:
/// w(n) = (1 - |x|) * cos(π|x|) + (1/π) * sin(π|x|)
/// where x = 2n/(N-1) - 1 and n is the sample index and N is the window size.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Bohman window:
/// - Creates a smooth bell-shaped curve that touches zero at both ends
/// - Has excellent performance in reducing "spectral leakage" (unwanted frequency artifacts)
/// - Uses a more complex mathematical formula than simpler windows
/// 
/// Think of signal processing like trying to hear one conversation in a crowded room. 
/// The Bohman window is like a sophisticated noise-canceling headphone that helps you 
/// focus on just the conversation you want to hear, filtering out background noise very 
/// effectively. It's particularly useful in applications where you need to distinguish 
/// between frequencies that are very close to each other, such as radar systems, sonar, 
/// or detailed audio analysis.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BohmanWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="BohmanWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Bohman window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public BohmanWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Bohman window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Bohman window function formula by calculating the window
    /// function value for each point from 0 to windowSize-1. The Bohman window is implemented
    /// using a combination of trigonometric functions to achieve its superior spectral characteristics.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Bohman formula
    /// - These values form a specialized bell-shaped curve
    /// - The calculations use cosine functions to create the precise shape needed
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps you analyze frequencies with high precision, reducing unwanted 
    /// artifacts that can occur when analyzing a limited segment of data. This is especially 
    /// important when you need to detect faint signals amid noise or separate signals that 
    /// are very close in frequency.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);

        // Handle edge case: window of size 1 has a single element with value 1.0
        if (windowSize == 1)
        {
            window[0] = _numOps.One;
            return window;
        }

        for (int n = 0; n < windowSize; n++)
        {
            // Bohman formula: w(n) = (1 - |x|) * cos(π|x|) + (1/π) * sin(π|x|)
            // where x = 2n/(N-1) - 1, so |x| ranges from 0 at center to 1 at edges
            double x = (2.0 * n / (windowSize - 1)) - 1.0;
            double absX = Math.Abs(x);
            double piAbsX = Math.PI * absX;

            // w(n) = (1 - |x|) * cos(π|x|) + (1/π) * sin(π|x|)
            double value = (1.0 - absX) * Math.Cos(piAbsX) + (1.0 / Math.PI) * Math.Sin(piAbsX);
            window[n] = _numOps.FromDouble(value);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Bohman window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Bohman;
}
