namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Cosine window function (sometimes called Sine window) for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Cosine window function is a simple yet effective window defined by the sine function:
/// w(n) = sin(pn/(N-1))
/// where n is the sample index and N is the window size.
/// Despite its name, this window actually uses the sine function mathematically, but it's called
/// the Cosine window due to historical convention in signal processing.
/// </para>
/// <para><b>For Beginners:</b> A window function is a mathematical tool that helps analyze signals more accurately.
/// 
/// The Cosine window:
/// - Creates a smooth half-sine wave shape from 0 to 1 and back to 0
/// - Has a simple mathematical form compared to other windows
/// - Provides a gentle tapering effect at the edges of your data
/// 
/// Think of it like gradually turning the volume up and then down when listening to audio.
/// Instead of an abrupt start and stop (which can cause distortion in analysis), the Cosine 
/// window smoothly increases from zero, reaches its maximum in the middle, and then smoothly 
/// decreases back to zero at the end. This helps reduce unwanted artifacts when analyzing 
/// frequencies in your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class CosineWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="CosineWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Cosine window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public CosineWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Cosine window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Cosine window function formula:
    /// w(n) = sin(pn/(N-1))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the sine function
    /// - The values start at 0, increase to 1 in the middle, and return to 0 at the end
    /// - The shape looks like a half-cycle of a sine wave
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), you smooth the transitions at the beginning and end of your data segment.
    /// This is particularly useful in audio processing, spectral analysis, and other applications
    /// where you need to analyze a portion of a longer signal.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            window[n] = MathHelper.Sin(_numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.Divide(nT, _numOps.FromDouble(windowSize - 1))));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Cosine window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Cosine;
}
