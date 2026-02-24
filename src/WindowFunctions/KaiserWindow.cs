namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Kaiser window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Kaiser window is a flexible window function based on the modified Bessel function of the first kind.
/// It is defined by the equation:
/// w(n) = I0(ßv(1-(2n/(N-1))²))/I0(ß)
/// where n is the sample index, N is the window size, I0 is the modified Bessel function of the first kind
/// of order zero, and ß (beta) is a parameter that controls the trade-off between the main lobe width
/// and side lobe amplitude.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Kaiser window:
/// - Creates a bell-shaped curve that can be adjusted using the beta parameter
/// - Allows you to control the trade-off between frequency resolution and spectral leakage
/// - Is highly versatile for different signal processing needs
/// 
/// Think of the Kaiser window like adjustable eyeglasses. The beta parameter works like a focus control:
/// - With a low beta (e.g., 1-2), it's like wide-angle glasses that let you see more frequencies
///   but with less precision about their exact strength
/// - With a high beta (e.g., 8-10), it's like zoom glasses that show you precise frequency information
///   but might miss nearby frequencies
/// 
/// This adjustability makes the Kaiser window useful in many applications from audio processing
/// to telecommunications to radar systems.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class KaiserWindow<T> : IWindowFunction<T>
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
    /// The beta parameter that controls the shape of the Kaiser window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the trade-off between main lobe width and side lobe amplitude.
    /// Larger values provide better side lobe suppression but widen the main lobe, reducing frequency resolution.
    /// Smaller values narrow the main lobe but increase side lobe levels.
    /// </para>
    /// <para><b>For Beginners:</b> The beta parameter adjusts the shape of the window.
    /// 
    /// Beta controls the window's characteristics:
    /// - Lower values (1-3): Better for frequency detection (seeing what frequencies are present)
    /// - Middle values (4-6): Good general-purpose settings
    /// - Higher values (7-10): Better for amplitude accuracy (measuring how strong each frequency is)
    /// 
    /// A common way to choose beta is based on the amount of side lobe suppression needed:
    /// - ß = 2.0: provides about -46 dB side lobe suppression
    /// - ß = 4.0: provides about -75 dB side lobe suppression
    /// - ß = 6.0: provides about -90 dB side lobe suppression
    /// 
    /// The default value of 5.0 works well for many applications, offering a good balance.
    /// </para>
    /// </remarks>
    private readonly T _beta;

    /// <summary>
    /// Initializes a new instance of the <see cref="KaiserWindow{T}"/> class with the specified beta value.
    /// </summary>
    /// <param name="beta">
    /// The beta parameter that controls the shape of the window. 
    /// Default value is 5.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Kaiser window function with the specified beta parameter
    /// and initializes the numeric operations provider for the specified type T. The beta parameter
    /// controls the trade-off between main lobe width and side lobe amplitude in the window.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the window with your chosen shape setting.
    /// 
    /// When creating a Kaiser window:
    /// - You can specify the beta value to customize the window's behavior
    /// - The default (5.0) provides a good balance for general use
    /// - Higher values improve side lobe suppression at the cost of frequency resolution
    /// - Lower values improve frequency resolution at the cost of side lobe suppression
    /// 
    /// This is like choosing different settings on a camera before taking a picture - 
    /// different settings work better for different situations.
    /// </para>
    /// </remarks>
    public KaiserWindow(double beta = 5.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _beta = _numOps.FromDouble(beta);
    }

    /// <summary>
    /// Creates a Kaiser window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Kaiser window function formula using the modified Bessel function.
    /// It calculates the window function value for each point from 0 to windowSize-1 and normalizes
    /// the result so that the maximum value is 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates a value using the Kaiser formula and the Bessel function
    /// - The mathematics are complex, involving the specialized Bessel function
    /// - The values are normalized so the largest value is exactly 1.0
    /// 
    /// The Bessel function helps create a window shape that provides an optimal trade-off
    /// between the width of the main peak and the size of unwanted ripples. When you apply 
    /// this window to your data (by multiplying each data point by the corresponding window value), 
    /// it helps you analyze frequencies with the specific characteristics you chose with the beta parameter.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);

        if (windowSize == 1)
        {
            window[0] = _numOps.One;
            return window;
        }

        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.FromDouble(2 * n - windowSize + 1), _numOps.FromDouble(windowSize - 1));
            T arg = _numOps.Sqrt(_numOps.Subtract(_numOps.One, _numOps.Multiply(x, x)));
            window[n] = MathHelper.BesselI0(_numOps.Multiply(_beta, arg));
        }

        // Normalize
        T maxVal = window.Max();
        for (int n = 0; n < windowSize; n++)
        {
            window[n] = _numOps.Divide(window[n], maxVal);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Kaiser window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Kaiser;
}
