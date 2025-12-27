namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements the Derivative of Gaussian (DOG) wavelet, which is based on the nth derivative
/// of the Gaussian function and is useful for detecting changes in signals.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Derivative of Gaussian (DOG) wavelet is derived from taking derivatives of the Gaussian function.
/// It has excellent localization properties in both time and frequency domains and is particularly
/// useful for detecting changes or transitions in signals.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// The Derivative of Gaussian (DOG) wavelet is like a mathematical tool that's especially
/// good at finding places where your data changes quickly.
/// 
/// Key features of DOG wavelets:
/// - They're based on derivatives of the Gaussian function (bell curve)
/// - Different orders detect different types of changes
/// - They're symmetric (the same on both sides of the center)
/// - They have good localization in both time and frequency
/// 
/// Think of them as detectors that respond strongly when your data shows specific
/// patterns of change:
/// - 1st order (order=1): Detects edges (sudden jumps)
/// - 2nd order (order=2): Detects peaks and valleys (Mexican Hat wavelet)
/// - Higher orders: Detect more complex patterns of change
/// 
/// These wavelets are particularly useful for:
/// - Edge detection in signals and images
/// - Finding points where data changes rapidly
/// - Scale-space analysis
/// - Feature detection
/// - Signal analysis where transitions are important
/// 
/// The order parameter lets you choose which type of change you're looking for.
/// </para>
/// </remarks>
public class DOGWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the derivative of the Gaussian function.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the DOGWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the derivative of the Gaussian function. Default is 2.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines which derivative of the Gaussian function is used
    /// to create the wavelet.
    /// 
    /// Different orders create wavelets with different properties:
    /// 
    /// - Order 1: First derivative of Gaussian
    ///   - Looks like an odd function (antisymmetric)
    ///   - Good for detecting edges (sudden changes)
    ///   - Similar to the Haar wavelet but smoother
    /// 
    /// - Order 2: Second derivative of Gaussian (Mexican Hat wavelet)
    ///   - Looks like a Mexican hat or sombrero
    ///   - Good for detecting peaks and valleys
    ///   - Most commonly used DOG wavelet
    /// 
    /// - Higher orders:
    ///   - More oscillations
    ///   - Can detect more complex patterns of change
    ///   - More selective in frequency
    /// 
    /// The default order of 2 corresponds to the Mexican Hat wavelet, which is
    /// widely used for its ability to detect peaks and transitions in signals.
    /// </para>
    /// </remarks>
    public DOGWavelet(int order = 2)
    {
        _order = order;
    }

    /// <summary>
    /// Calculates the value of the DOG wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the DOG wavelet at a specific point.
    /// 
    /// The DOG wavelet is defined as the nth derivative of the Gaussian function:
    /// ?(x) = (-1)^n · d^n/dx^n(e^(-x²/2))
    /// 
    /// For specific orders, this gives:
    /// - Order 1: ?(x) = -x · e^(-x²/2)
    /// - Order 2: ?(x) = (x² - 1) · e^(-x²/2)
    /// - Order 3: ?(x) = -(x³ - 3x) · e^(-x²/2)
    /// 
    /// The implementation uses a combination of:
    /// 1. The appropriate polynomial term based on the order
    /// 2. The Gaussian envelope e^(-x²/2)
    /// 3. A normalization factor to ensure proper scaling
    /// 
    /// The result is a function that:
    /// - Has n+1 zero-crossings for order n
    /// - Decays to zero as x moves away from the center
    /// - Has specific vanishing moments
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        T x2 = NumOps.Square(x);
        T exp_term = NumOps.Exp(NumOps.Negate(NumOps.Divide(x2, NumOps.FromDouble(2))));

        // Compute the appropriate Hermite polynomial term for the nth derivative of Gaussian
        // Order 1: ψ(x) = -x · e^(-x²/2)
        // Order 2: ψ(x) = (x² - 1) · e^(-x²/2)  (Mexican Hat/Ricker wavelet)
        // Order 3: ψ(x) = (x³ - 3x) · e^(-x²/2)
        // Order 4: ψ(x) = (x⁴ - 6x² + 3) · e^(-x²/2)
        T polynomial_term;
        switch (_order)
        {
            case 1:
                polynomial_term = NumOps.Negate(x);
                break;
            case 2:
                polynomial_term = NumOps.Subtract(x2, NumOps.One);
                break;
            case 3:
                polynomial_term = NumOps.Subtract(NumOps.Multiply(x, x2), NumOps.Multiply(NumOps.FromDouble(3), x));
                break;
            case 4:
                T x4 = NumOps.Multiply(x2, x2);
                polynomial_term = NumOps.Add(NumOps.Subtract(x4, NumOps.Multiply(NumOps.FromDouble(6), x2)), NumOps.FromDouble(3));
                break;
            default:
                // For higher orders, fall back to a simple approximation
                polynomial_term = NumOps.FromDouble(Math.Pow(-1, _order));
                for (int i = 0; i < _order; i++)
                {
                    polynomial_term = NumOps.Multiply(polynomial_term, x);
                }
                break;
        }

        T result = NumOps.Multiply(polynomial_term, exp_term);

        // Normalization factor
        double norm_factor = 1.0 / (Math.Sqrt(Convert.ToDouble(MathHelper.Factorial<T>(_order))) * Math.Pow(2, (_order + 1.0) / 2.0));
        result = NumOps.Multiply(result, NumOps.FromDouble(norm_factor));

        return result;
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the DOG wavelet.
    /// </summary>
    /// <param name="input">The input signal vector to decompose.</param>
    /// <returns>A tuple containing the approximation coefficients and detail coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs one level of wavelet decomposition on your signal, splitting it into:
    /// 
    /// - Approximation coefficients: Represent the low-frequency components (the overall shape)
    /// - Detail coefficients: Represent the high-frequency components (the fine details)
    /// 
    /// The process works like this:
    /// 1. The input signal is convolved (filtered) with a Gaussian filter to get the approximation
    /// 2. The input signal is convolved with the DOG wavelet to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes the DOG wavelet special for this task is:
    /// - It's excellent at detecting changes or transitions in the signal
    /// - It has good localization properties in both time and frequency
    /// - Different orders can detect different types of features
    /// 
    /// For example, with order=2 (Mexican Hat), the detail coefficients will highlight
    /// places where the signal has peaks or valleys.
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which makes wavelet decomposition efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        var waveletCoeffs = GetWaveletCoefficients();
        var scalingCoeffs = GetScalingCoefficients();

        var approximation = Convolve(input, scalingCoeffs);
        var detail = Convolve(input, waveletCoeffs);

        // Downsample by 2
        approximation = Downsample(approximation, 2);
        detail = Downsample(detail, 2);

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling function coefficients for the DOG wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    /// 
    /// For DOG wavelets, these coefficients are derived from the Gaussian function:
    /// 
    /// g(x) = e^(-x²)
    /// 
    /// The Gaussian function is a natural choice for the scaling function because:
    /// - It's smooth and has good localization properties
    /// - It's the function whose derivatives create the DOG wavelets
    /// - It acts as an effective low-pass filter
    /// 
    /// This method:
    /// 1. Creates a discretized Gaussian function of specified length
    /// 2. Normalizes the coefficients so they sum to 1
    /// 
    /// The resulting filter captures the low-frequency components of the signal,
    /// providing the "approximation" part of the wavelet decomposition.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        int length = 64;
        var coeffs = new T[length];
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T x = NumOps.Divide(NumOps.FromDouble(i - length / 2), NumOps.FromDouble(length / 4));
            T value = NumOps.Exp(NumOps.Negate(NumOps.Multiply(x, x)));
            coeffs[i] = value;
            sum = NumOps.Add(sum, NumOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = NumOps.Divide(coeffs[i], sum);
        }

        return new Vector<T>(coeffs);
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the DOG wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For DOG wavelets, these coefficients are a discretized version of the
    /// first derivative of the Gaussian function:
    /// 
    /// ?(t) = -2t · e^(-t²)
    /// 
    /// This method:
    /// 1. Creates a discretized version of this function with specified length
    /// 2. Samples the function at regular intervals
    /// 3. Normalizes the coefficients to ensure proper energy preservation
    /// 
    /// The resulting filter is excellent at detecting edges and transitions in the signal.
    /// The characteristic shape with opposite signs on either side of the center
    /// makes it respond strongly to changes in the data.
    /// 
    /// Note that this implementation specifically uses the first derivative form
    /// regardless of the order parameter. For a complete implementation, this would
    /// need to be extended to use the appropriate derivative based on the order.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        int length = 256;
        var coeffs = new T[length];
        T sum = NumOps.Zero;

        for (int i = 0; i < length; i++)
        {
            T t = NumOps.Divide(NumOps.FromDouble(i - length / 2), NumOps.FromDouble(length / 4));
            T t2 = NumOps.Multiply(t, t);
            T value = NumOps.Multiply(
                NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(2), t)),
                NumOps.Exp(NumOps.Negate(t2))
            );
            coeffs[i] = value;
            sum = NumOps.Add(sum, NumOps.Abs(value));
        }

        // Normalize
        for (int i = 0; i < length; i++)
        {
            coeffs[i] = NumOps.Divide(coeffs[i], sum);
        }

        return new Vector<T>(coeffs);
    }

    /// <summary>
    /// Performs convolution of an input signal with a filter.
    /// </summary>
    /// <param name="input">The input signal vector.</param>
    /// <param name="kernel">The filter vector.</param>
    /// <returns>The convolved signal vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Convolution is a mathematical operation that combines two functions to produce a third function.
    /// In signal processing, it's like sliding a filter over a signal and calculating a weighted sum
    /// at each position.
    /// 
    /// The process works like this:
    /// 1. For each position in the output:
    ///    a. Center the filter at that position
    ///    b. Multiply each filter coefficient by the corresponding signal value
    ///    c. Sum these products to get the output value at that position
    /// 
    /// This implementation produces a result with length equal to input.length + kernel.length - 1,
    /// which is the full convolution without truncation. This ensures that no information is lost
    /// at the boundaries.
    /// 
    /// Convolution is fundamental to wavelet transforms and many other signal processing operations.
    /// It's how filters are applied to signals to extract specific frequency components.
    /// </para>
    /// </remarks>
    private Vector<T> Convolve(Vector<T> input, Vector<T> kernel)
    {
        int resultLength = input.Length + kernel.Length - 1;
        var result = new T[resultLength];

        for (int i = 0; i < resultLength; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < kernel.Length; j++)
            {
                if (i - j >= 0 && i - j < input.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[i - j], kernel[j]));
                }
            }
            result[i] = sum;
        }

        return new Vector<T>(result);
    }

    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// Downsampling reduces the sampling rate of a signal by keeping only a subset of the samples.
    /// 
    /// For example, with a factor of 2:
    /// - Original signal: [10, 20, 30, 40, 50, 60]
    /// - Downsampled signal: [10, 30, 50]
    /// 
    /// This method keeps every nth sample (where n is the factor) and discards the rest.
    /// In wavelet decomposition, downsampling by 2 is standard after filtering, because:
    /// 
    /// 1. The filters have already removed frequency components that would cause aliasing
    /// 2. It reduces the data size by half at each decomposition level
    /// 3. It ensures that the total size of approximation and detail coefficients equals the input size
    /// 
    /// Downsampling is crucial for the efficiency of wavelet transforms, especially for
    /// multi-level decomposition and compression applications.
    /// </para>
    /// </remarks>
    private Vector<T> Downsample(Vector<T> input, int factor)
    {
        int newLength = input.Length / factor;
        var result = new T[newLength];

        for (int i = 0; i < newLength; i++)
        {
            result[i] = input[i * factor];
        }

        return new Vector<T>(result);
    }
}
