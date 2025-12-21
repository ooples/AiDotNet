namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Lanczos window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Lanczos window is based on the Lanczos kernel (sinc function) and is defined by the equation:
/// w(n) = sinc(2n/(N-1) - 1)
/// where n is the sample index, N is the window size, and sinc(x) = sin(px)/(px) for x ? 0 and sinc(0) = 1.
/// The Lanczos window provides good frequency resolution while reducing side lobe amplitude.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Lanczos window:
/// - Creates a smooth curve based on the sinc function (sin(x)/x)
/// - Provides good preservation of the main signal features
/// - Is particularly useful for resampling and interpolation tasks
/// 
/// Think of the Lanczos window like a special camera filter that helps capture more detail without 
/// distortion. The sinc function it's based on is fundamental in signal processing because it has 
/// ideal frequency characteristics - it can perfectly reconstruct bandwidth-limited signals. In practice, 
/// the Lanczos window is commonly used when resizing images, resampling audio, or analyzing data 
/// where preserving the main features of the signal is important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LanczosWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="LanczosWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Lanczos window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public LanczosWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Lanczos window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Lanczos window function formula:
    /// w(n) = sinc(2n/(N-1) - 1)
    /// It calculates the window function value for each point from 0 to windowSize-1 using the sinc function.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the sinc function (sin(px)/(px))
    /// - The values form a curve with a main lobe in the middle and smaller oscillations on the sides
    /// - These specific mathematical properties make it ideal for certain signal processing tasks
    /// 
    /// The sinc function has a special property: its frequency response is a perfect rectangular shape.
    /// This makes the Lanczos window particularly good at tasks like interpolation (estimating values
    /// between known points) and resampling (changing the sampling rate of a signal). When you apply
    /// this window to your data, it helps preserve important signal characteristics while minimizing
    /// unwanted artifacts.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T x = _numOps.Multiply(_numOps.FromDouble(2 * n), _numOps.Divide(_numOps.One, _numOps.FromDouble(windowSize - 1)));
            x = _numOps.Subtract(x, _numOps.One);
            window[n] = MathHelper.Sinc(x);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Lanczos window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Lanczos;
}
