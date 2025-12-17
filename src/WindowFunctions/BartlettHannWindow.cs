namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Bartlett-Hann window function for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Bartlett-Hann window function is a combination of the Bartlett and Hann windows, designed to 
/// provide better frequency resolution and reduced spectral leakage compared to either window used alone. 
/// It is defined by the equation: w(n) = 0.62 - 0.48|n/(N-1) - 0.5| - 0.38cos(2p(n/(N-1) - 0.5))
/// where n is the sample index and N is the window size.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that smooths out the edges of a signal.
/// 
/// Think of it like taking a photo through a frame:
/// - The frame (window) determines which parts of the scene are visible and how they appear
/// - The Bartlett-Hann window is a specific type of frame that gradually fades the edges
/// - It helps reduce unwanted artifacts when analyzing sounds, images, or other signals
/// 
/// In practice, this window function helps when:
/// - Analyzing frequencies in audio signals
/// - Processing data where you need to avoid sharp transitions
/// - Improving the accuracy of spectrum analysis
/// 
/// The Bartlett-Hann window combines the benefits of two simpler windows (Bartlett and Hann)
/// to create a more effective tool for signal processing.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BartlettHannWindow<T> : IWindowFunction<T>
{
    /// <summary>
    /// The numeric operations provider for performing calculations with type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores an object that provides operations like addition, multiplication, and
    /// trigonometric functions for the generic numeric type T. It's essential for implementing
    /// the window function's mathematical formula across different numeric types.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a calculator that knows how to do math with 
    /// different types of numbers (float, double, etc.).
    /// 
    /// It allows the code to:
    /// - Work with various number types without writing separate code for each
    /// - Perform mathematical operations like addition and multiplication
    /// - Calculate more complex operations like cosine functions
    /// 
    /// If T is a float, the operations will work with float precision.
    /// If T is a double, the operations will have double precision.
    /// </para>
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="BartlettHannWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Bartlett-Hann window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the window function for use.
    /// 
    /// When you create a new Bartlett-Hann window:
    /// - It initializes the calculator mentioned above
    /// - No window is created yet - you'll need to call the Create method
    /// - It prepares everything needed to generate the window when requested
    /// 
    /// Think of this like preparing your tools before starting a project.
    /// </para>
    /// </remarks>
    public BartlettHannWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Bartlett-Hann window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Bartlett-Hann window function formula:
    /// w(n) = 0.62 - 0.48|n/(N-1) - 0.5| - 0.38cos(2p(n/(N-1) - 0.5))
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the actual window values based on the size you need.
    /// 
    /// For each position in the window:
    /// - A value is calculated using the Bartlett-Hann formula
    /// - The value will be between 0 and 1
    /// - The values are higher in the middle and lower at the edges
    /// 
    /// The resulting window is like a weighting system that emphasizes the center of your data
    /// and reduces the importance of data at the edges, creating smoother transitions.
    /// 
    /// For example, if windowSize is 5, you might get values like [0.08, 0.52, 0.85, 0.52, 0.08]
    /// (exact values will vary).
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T term1 = _numOps.FromDouble(0.62);
            T term2 = _numOps.Multiply(_numOps.FromDouble(0.48), _numOps.Abs(_numOps.Subtract(_numOps.Divide(nT, _numOps.FromDouble(windowSize - 1)), _numOps.FromDouble(0.5))));
            T term3 = _numOps.Multiply(_numOps.FromDouble(0.38), MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(2 * Math.PI), _numOps.Subtract(_numOps.Divide(nT, _numOps.FromDouble(windowSize - 1)), _numOps.FromDouble(0.5)))));
            window[n] = _numOps.Subtract(_numOps.Subtract(term1, term2), term3);
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Bartlett-Hann window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.BartlettHann;
}
