namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Blackman-Nuttall window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Blackman-Nuttall window is a high-performance window function that provides excellent side lobe 
/// suppression, making it ideal for spectral analysis. It uses a weighted cosine series with four terms:
/// w(n) = 0.3635819 - 0.4891775 * cos(2pn/(N-1)) + 0.1365995 * cos(4pn/(N-1)) - 0.0106411 * cos(6pn/(N-1))
/// where n is the sample index and N is the window size.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special lens that helps focus on specific parts of your data.
/// 
/// The Blackman-Nuttall window:
/// - Creates a smooth, bell-shaped curve that gradually tapers to zero at the edges
/// - Has very good "side lobe suppression" (reduces unwanted artifacts in frequency analysis)
/// - Helps you see the true frequencies in your data with minimal distortion
/// 
/// Imagine looking at stars through a telescope - a regular window might show some glare around bright stars,
/// while the Blackman-Nuttall window would help reduce that glare, giving you a clearer view of fainter stars nearby.
/// This is particularly useful in audio processing, radar systems, or any application where you need to
/// distinguish between frequencies that are close to each other.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BlackmanNuttallWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="BlackmanNuttallWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Blackman-Nuttall window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public BlackmanNuttallWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Blackman-Nuttall window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Blackman-Nuttall window function formula:
    /// w(n) = 0.3635819 - 0.4891775 * cos(2pn/(N-1)) + 0.1365995 * cos(4pn/(N-1)) - 0.0106411 * cos(6pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Blackman-Nuttall formula
    /// - The values form a bell-shaped curve (near 0 at the edges, higher in the middle)
    /// - The precise mathematical formula uses cosine waves at different frequencies
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps focus your analysis on the true frequency content of your signal,
    /// reducing unwanted artifacts that can occur when analyzing a finite segment of data.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            // Blackman-Nuttall formula: w(n) = a0 - a1*cos(2πn/(N-1)) + a2*cos(4πn/(N-1)) - a3*cos(6πn/(N-1))
            T term1 = _numOps.FromDouble(0.3635819);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.4891775), MathHelper.Cos(_numOps.FromDouble(2 * Math.PI * n / (windowSize - 1))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.1365995), MathHelper.Cos(_numOps.FromDouble(4 * Math.PI * n / (windowSize - 1))));
            T term4 = _numOps.Multiply(_numOps.FromDouble(0.0106411), MathHelper.Cos(_numOps.FromDouble(6 * Math.PI * n / (windowSize - 1))));
            // Formula: term1 - term2 + term3 - term4
            window[n] = _numOps.Subtract(_numOps.Add(_numOps.Subtract(term1, term2), term3), term4);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Blackman-Nuttall window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.BlackmanNuttall;
}
