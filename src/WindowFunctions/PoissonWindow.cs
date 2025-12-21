namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Poisson window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Poisson window is an exponential window function defined by the equation:
/// w(n) = exp(-a|n-N/2|/(N/2))
/// where n is the sample index, N is the window size, and a (alpha) is a parameter
/// that controls the rate of decay from the center of the window to the edges.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Poisson window:
/// - Creates a curve that decays exponentially from the center to the edges
/// - Has a shape that can be adjusted using the alpha parameter
/// - Provides a smooth transition that helps reduce artifacts in frequency analysis
/// 
/// Think of the Poisson window like a spotlight that gradually fades out from the center.
/// The alpha parameter controls how quickly this fading happens - a higher alpha means
/// a faster fade out from the center. This gradual fading helps when analyzing signals by
/// reducing the artificial effects that occur when you're only looking at a segment of
/// a longer signal.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class PoissonWindow<T> : IWindowFunction<T>
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
    /// The alpha parameter that controls the shape of the Poisson window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the rate of exponential decay from the center of the window to the edges.
    /// Higher values create a narrower window with faster decay, while lower values create a wider
    /// window with slower decay.
    /// </para>
    /// <para><b>For Beginners:</b> The alpha parameter adjusts how quickly the window tapers off.
    /// 
    /// Think of alpha like adjusting how focused the spotlight is:
    /// - A smaller alpha (e.g., 1.0) creates a gradual fade-out, like a wide spotlight
    /// - A larger alpha (e.g., 4.0) creates a rapid fade-out, like a narrow spotlight
    /// - The default value (2.0) provides a moderate fade-out suitable for general use
    /// 
    /// Adjusting alpha lets you control the trade-off between time resolution (how precisely you can
    /// pinpoint when something happens) and frequency resolution (how precisely you can identify
    /// what frequencies are present).
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Initializes a new instance of the <see cref="PoissonWindow{T}"/> class with the specified alpha value.
    /// </summary>
    /// <param name="alpha">
    /// The alpha parameter that controls the shape of the window. 
    /// Default value is 2.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Poisson window function with the specified alpha parameter
    /// and initializes the numeric operations provider for the specified type T. The alpha parameter
    /// controls the rate of exponential decay from the center of the window to the edges.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the window with your chosen shape setting.
    /// 
    /// When creating a Poisson window:
    /// - You can specify the alpha value to customize the window's behavior
    /// - The default (2.0) works well for many general purposes
    /// - Lower alpha values create a wider, more gradual window
    /// - Higher alpha values create a narrower, more focused window
    /// 
    /// This is like choosing the right tool for a job - different alpha values
    /// are better suited for different types of signals and analysis needs.
    /// </para>
    /// </remarks>
    public PoissonWindow(double alpha = 2.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _alpha = _numOps.FromDouble(alpha);
    }

    /// <summary>
    /// Creates a Poisson window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Poisson window function formula:
    /// w(n) = exp(-a|n-N/2|/(N/2))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method calculates how far the position is from the center
    /// - It then applies an exponential decay based on this distance
    /// - The decay rate is controlled by the alpha parameter
    /// - The result is highest in the middle (1.0) and decays toward the edges
    /// 
    /// When you apply this window to your data (by multiplying each data point by the corresponding
    /// window value), it helps focus your analysis on the central portion of your data while smoothly
    /// reducing the contribution from the edges. This helps reduce the artificial effects that can
    /// occur when analyzing a limited segment of a longer signal.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T halfN = _numOps.Divide(_numOps.FromDouble(windowSize - 1), _numOps.FromDouble(2));
        for (int n = 0; n < windowSize; n++)
        {
            T x = _numOps.Divide(_numOps.Abs(_numOps.Subtract(_numOps.FromDouble(n), halfN)), halfN);
            window[n] = _numOps.Exp(_numOps.Multiply(_numOps.Negate(_alpha), x));
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Poisson window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Poisson;
}
