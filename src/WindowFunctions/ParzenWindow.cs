namespace AiDotNet.WindowFunctions;

/// <summary>
/// Implements the Parzen window function (also known as the de la Vallée-Poussin window) for signal processing applications.
/// </summary>
/// <remarks>
/// <para>
/// The Parzen window is a piecewise cubic approximation of the Gaussian window. It is defined by a piecewise function:
/// For |n - N/2| = N/4:
///   w(n) = 1 - 6(2|n-N/2|/N)² + 6(2|n-N/2|/N)³
/// For N/4 &lt; |n - N/2| = N/2:
///   w(n) = 2(1 - 2|n-N/2|/N)³
/// where n is the sample index and N is the window size.
/// </para>
/// <para><b>For Beginners:</b> A window function is like a special filter that helps analyze signals more accurately.
/// 
/// The Parzen window:
/// - Creates a smooth curve that looks similar to a bell curve (Gaussian shape)
/// - Has excellent side lobe suppression (reduces unwanted frequency artifacts)
/// - Uses different mathematical formulas for different parts of the window
/// 
/// Think of the Parzen window like a carefully designed dimmer switch that doesn't just
/// turn the lights up and down, but does so in a mathematically precise way. The central
/// part of the window follows one formula, while the outer parts follow another. This
/// special design helps improve frequency analysis by reducing measurement errors that
/// occur when analyzing a limited segment of a continuous signal.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ParzenWindow<T> : IWindowFunction<T>
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
    /// Initializes a new instance of the <see cref="ParzenWindow{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor creates a new Parzen window function and initializes the numeric
    /// operations provider for the specified type T. The provider is obtained from a helper class
    /// that selects the appropriate implementation based on the type.
    /// </remarks>
    public ParzenWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Creates a Parzen window of the specified size.
    /// </summary>
    /// <param name="windowSize">The size of the window to create.</param>
    /// <returns>A vector containing the window function values.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Parzen window function formula, which is a piecewise function:
    /// For |n - N/2| = N/4:
    ///   w(n) = 1 - 6(2|n-N/2|/N)² + 6(2|n-N/2|/N)³
    /// For N/4 &lt; |n - N/2| = N/2:
    ///   w(n) = 2(1 - 2|n-N/2|/N)³
    /// It calculates the window function value for each point from 0 to windowSize-1.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates the actual window values based on the size you specify.
    /// 
    /// For each position in the window:
    /// - The method first calculates how far the position is from the center
    /// - Based on this distance, it applies one of two different formulas
    /// - Positions near the center use one formula, positions further out use another
    /// - This creates a smooth curve that's highest in the middle and tapers to zero at the edges
    /// 
    /// This piecewise approach (using different formulas for different parts) creates a window
    /// that works well for many signal processing applications. When you apply this window to your
    /// data (by multiplying each data point by the corresponding window value), it helps reduce
    /// unwanted artifacts in your frequency analysis, especially when you're analyzing a small
    /// segment of a longer signal.
    /// </para>
    /// </remarks>
    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T N = _numOps.FromDouble(windowSize - 1);
        T halfN = _numOps.Divide(N, _numOps.FromDouble(2));
        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T absN = _numOps.Abs(_numOps.Subtract(nT, halfN));
            if (_numOps.LessThanOrEquals(absN, _numOps.Divide(N, _numOps.FromDouble(4))))
            {
                T term = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), absN), N);
                window[n] = _numOps.Subtract(_numOps.One, _numOps.Multiply(_numOps.FromDouble(6), _numOps.Add(_numOps.Square(term), _numOps.Multiply(_numOps.FromDouble(-1), _numOps.Power(term, _numOps.FromDouble(3))))));
            }
            else
            {
                T term = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), absN), N);
                window[n] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Power(_numOps.Subtract(_numOps.One, term), _numOps.FromDouble(3)));
            }
        }

        return window;
    }

    /// <summary>
    /// Gets the type identifier for this window function.
    /// </summary>
    /// <returns>The window function type enumeration value.</returns>
    /// <remarks>
    /// This method returns an enumeration value that uniquely identifies the Parzen window function.
    /// This identifier can be used by other components in the system to determine the type of window
    /// function being used without relying on the concrete type.
    /// </remarks>
    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Parzen;
}
