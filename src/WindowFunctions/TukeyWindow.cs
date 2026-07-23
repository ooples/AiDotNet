namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Tukey window function (also known as tapered cosine window) for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Tukey window is a flexible window function that combines a flat top (Rectangular window)
/// with cosine tapered edges. It is defined by a piecewise function controlled by the alpha parameter:
/// 
/// For 0 = n = aN/2:
///   w(n) = 0.5 * (1 + cos(p * (2n/aN - 1)))
/// For aN/2 &lt; n &lt; N - aN/2:
///   w(n) = 1
/// For N - aN/2 = n = N:
///   w(n) = 0.5 * (1 + cos(p * (2n/aN - 2/a + 1)))
/// 
/// where n is the sample index, N is (windowSize - 1), and a (alpha) is a parameter between 0 and 1
/// that controls the width of the cosine tapered regions.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Tukey window:
/// - Combines the best features of Rectangular and Cosine windows
/// - Has a flat middle section and smoothly tapered edges
/// - Can be adjusted using the alpha parameter to control how much tapering occurs
/// 
/// Think of it like a plateau with gentle slopes on both sides. The alpha parameter controls
/// how much of the window is plateau versus slope:
/// - An alpha of 0 creates a completely flat plateau (identical to a Rectangular window)
/// - An alpha of 1 creates a window with no plateau, just slopes (identical to a Hann window)
/// - Values in between (like the default 0.5) create a mix of plateau and slopes
/// 
/// This flexibility makes the Tukey window useful in many applications where you need
/// to balance preserving signal strength with reducing spectral leakage.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class TukeyWindow<T> : IWindowFunction<T>
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
    /// The alpha parameter that controls the shape of the Tukey window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the proportion of the window that has cosine tapered edges.
    /// It must be between 0 and 1, where:
    /// - a = 0 produces a Rectangular window (no tapering)
    /// - a = 1 produces a Hann window (fully tapered)
    /// - 0 &lt; a &lt; 1 produces a flat top with cosine tapered edges
    /// </para>
    /// <para><b>For Beginners:</b> The alpha parameter adjusts the balance between the flat section and tapered edges.
    /// 
    /// Alpha controls how much of the window is flat versus tapered:
    /// - Lower values (close to 0): Mostly flat with small tapered sections at the edges
    /// - Middle values (around 0.5): Equal balance of flat section and tapered edges
    /// - Higher values (close to 1): Mostly tapered with small flat section in the middle
    /// 
    /// The default value of 0.5 means that half of the window will be flat (the middle)
    /// and half will be tapered (1/4 on each edge). This provides a good balance for
    /// many applications, giving you some of the benefits of both window types.
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the <see cref="TukeyWindow{T}"/> class with the specified alpha value.
    /// </summary>
    /// <param name="alpha">
    /// The alpha parameter that controls the shape of the window. 
    /// Must be between 0 and 1. Default value is 0.5.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Tukey window function with the specified alpha parameter
    /// and initializes the numeric operations provider for the specified type T. The alpha parameter
    /// controls the proportion of the window that has cosine tapered edges.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the window with your chosen shape setting.
    /// 
    /// When creating a Tukey window:
    /// - You can specify the alpha value to customize the window's behavior
    /// - The default (0.5) works well for many general purposes
    /// - Lower alpha values preserve more of the original signal amplitude
    /// - Higher alpha values provide better spectral leakage reduction
    /// 
    /// This flexibility makes the Tukey window useful in many applications where
    /// you need to balance different signal processing requirements.
    /// </para>
    /// </remarks>
    public TukeyWindow(double alpha = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _alpha = _numOps.FromDouble(alpha);
    }

    /// <summary>
    /// Creates a Tukey window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Tukey window function formula, which is a piecewise function:
    /// 
    /// For 0 = n = aN/2:
    ///   w(n) = 0.5 * (1 + cos(p * (2n/aN - 1)))
    /// For aN/2 &lt; n &lt; N - aN/2:
    ///   w(n) = 1
    /// For N - aN/2 = n = N:
    ///   w(n) = 0.5 * (1 + cos(p * (2n/aN - 2/a + 1)))
    /// 
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method first determines which region the position falls in (left taper, middle, or right taper)
    /// - Based on the region, it applies the appropriate formula
    /// - The left and right edges use a cosine taper formula
    /// - The middle section is set to 1.0 (no change to the signal)
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it preserves the original amplitude in the middle while smoothly tapering
    /// the edges. This helps reduce unwanted artifacts in frequency analysis while maintaining
    /// the strength of the main signal components.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);

        // Special case: alpha = 0 is a rectangular window (all ones)
        if (_numOps.Equals(_alpha, _numOps.Zero))
        {
            for (int n = 0; n < windowSize; n++)
            {
                window[n] = _numOps.One;
            }
            return window;
        }

        T N = _numOps.FromDouble(windowSize - 1);
        T halfAlphaN = _numOps.Multiply(_alpha, _numOps.Divide(N, _numOps.FromDouble(2)));

        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            if (_numOps.LessThanOrEquals(nT, halfAlphaN))
            {
                // Left taper region: 0.5 * (1 + cos(π * (2n/(αN) - 1)))
                T x = _numOps.Subtract(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), nT), _numOps.Multiply(_alpha, N)), _numOps.One);
                window[n] = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Add(_numOps.One, MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(Math.PI), x))));
            }
            else if (_numOps.LessThan(nT, _numOps.Subtract(N, halfAlphaN)))
            {
                // Flat region
                window[n] = _numOps.One;
            }
            else
            {
                // Right taper region: 0.5 * (1 + cos(π * (2n/(αN) - 2/α + 1)))
                T x = _numOps.Add(_numOps.Subtract(_numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), nT), _numOps.Multiply(_alpha, N)), _numOps.Divide(_numOps.FromDouble(2), _alpha)), _numOps.One);
                window[n] = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Add(_numOps.One, MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(Math.PI), x))));
            }
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Tukey window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Tukey;
}
