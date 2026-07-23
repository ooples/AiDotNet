namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Nuttall window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Nuttall window is a high-performance window function that provides excellent side lobe 
/// suppression. It uses a weighted sum of cosine terms:
/// w(n) = 0.355768 - 0.487396 * cos(2pn/(N-1)) + 0.144232 * cos(4pn/(N-1)) - 0.012604 * cos(6pn/(N-1))
/// where n is the sample index and N is the window size. The Nuttall window was designed to provide
/// very low side lobe levels with a continuous first derivative.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Nuttall window:
/// - Creates a smooth bell-shaped curve that gradually tapers to zero at the edges
/// - Has extremely low "side lobes" (unwanted ripples in frequency analysis)
/// - Works well when you need to detect weak signals near strong ones
/// 
/// Think of it like a high-quality telescope that allows you to see faint stars that are close to very 
/// bright ones. When analyzing frequencies in a signal, the Nuttall window helps you detect weaker 
/// frequencies that might be close to dominant ones. It's particularly useful in applications like radar, 
/// sonar, and spectrum analysis where you need to distinguish between closely spaced frequency components 
/// with widely different amplitudes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NuttallWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="NuttallWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Nuttall window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public NuttallWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Nuttall window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Nuttall window function formula:
    /// w(n) = 0.355768 - 0.487396 * cos(2pn/(N-1)) + 0.144232 * cos(4pn/(N-1)) - 0.012604 * cos(6pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Nuttall formula
    /// - The calculation combines four cosine waves with different frequencies and amplitudes
    /// - These precise coefficients were carefully chosen to create optimal spectral properties
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps you analyze frequencies with minimal interference between different
    /// frequency components. This is especially valuable when you need to detect weak signals in
    /// the presence of strong ones, such as in radar systems, medical imaging, or audio analysis.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.Multiply(_numOps.FromDouble(0.355768), _numOps.One);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.487396), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.144232), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(4 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            T term4 = _numOps.Multiply(_numOps.FromDouble(0.012604), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(6 * Math.PI * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)))));
            window[n] = _numOps.Subtract(_numOps.Subtract(_numOps.Add(term1, term2), term3), term4);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Nuttall window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Nuttall;
}
