namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Gaussian window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Gaussian window is based on the Gaussian (normal) distribution and provides excellent 
/// time-frequency localization. It is defined by the equation:
/// w(n) = exp(-(n-N/2)²/(2s²))
/// where n is the sample index, N is the window size, and s (sigma) controls the width of the window.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Gaussian window:
/// - Creates a smooth bell-shaped curve (like the famous "bell curve" in statistics)
/// - Has a shape that can be adjusted using the sigma parameter
/// - Provides a good balance between time and frequency resolution
/// 
/// Think of it like a spotlight with adjustable focus. A narrow spotlight (small sigma) gives you 
/// very precise location information but less overall visibility. A wide spotlight (large sigma) 
/// shows you more of the scene but with less precision about exact locations. The Gaussian window 
/// works the same way when analyzing frequencies in signals - you can adjust it to balance between 
/// precise frequency measurement and detecting the presence of multiple frequencies.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GaussianWindow<T> : IWindowFunction<T>
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
    /// The standard deviation (sigma) parameter that controls the width of the Gaussian window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the width of the Gaussian window. A larger value creates a wider window
    /// with better frequency resolution but poorer time resolution. A smaller value creates a narrower
    /// window with better time resolution but poorer frequency resolution.
    /// </para>
    /// <para><b>For Beginners:</b> The sigma parameter adjusts how wide or narrow the bell curve is.
    /// 
    /// Think of sigma like the focus control on a camera:
    /// - A smaller sigma (e.g., 0.3) creates a narrower peak, focusing on a specific time segment
    /// - A larger sigma (e.g., 0.7) creates a wider peak, covering more of the data at once
    /// - The default value (0.5) provides a balanced setting for many applications
    /// 
    /// Adjusting sigma helps you fine-tune the window for your specific needs, whether you need
    /// precise timing information or better frequency detection.
    /// </para>
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Initializes a new instance of the <see cref="GaussianWindow{T}"/> class with the specified sigma value.
    /// </summary>
    /// <param name="sigma">
    /// The standard deviation parameter that controls the width of the window. 
    /// Default value is 0.5.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Gaussian window function with the specified sigma parameter
    /// and initializes the numeric operations provider for the specified type T. The sigma parameter
    /// controls the width of the Gaussian curve, with higher values creating a wider window.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the window with your chosen width setting.
    /// 
    /// When creating a Gaussian window:
    /// - You can optionally specify how wide you want the window to be
    /// - The default (0.5) works well for many general purposes
    /// - Lower values create a narrower window that focuses on precise time locations
    /// - Higher values create a wider window that's better for detecting frequency components
    /// 
    /// This is like choosing a camera lens before taking a photo - different lenses (sigma values)
    /// are better for different types of analysis.
    /// </para>
    /// </remarks>
    public GaussianWindow(double sigma = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = _numOps.FromDouble(sigma);
    }

    /// <summary>
    /// Creates a Gaussian window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Gaussian window function formula:
    /// w(n) = exp(-(n-N/2)²/(2s²))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates the distance from the center
    /// - It applies the Gaussian formula (similar to the bell curve in statistics)
    /// - The result is highest in the middle and gradually decreases toward the edges
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps you analyze your signal with a balanced approach to time and frequency
    /// precision. This is particularly useful in audio processing, image processing, and other 
    /// applications where you need to analyze patterns in complex data.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T center = _numOps.Divide(_numOps.FromDouble(windowSize - 1), _numOps.FromDouble(2.0));
        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.Subtract(_numOps.FromDouble(n), center), _numOps.Multiply(_sigma, _numOps.FromDouble(2.0)));
            window[n] = _numOps.Exp(_numOps.Negate(_numOps.Multiply(x, x)));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Gaussian window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Gaussian;
}
