namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Represents a Fejér-Korovkin wavelet function implementation for signal processing and analysis.
/// </summary>
/// <remarks>
/// <para>
/// The Fejér-Korovkin wavelet is a mathematical function used in signal processing for decomposing
/// signals into different frequency components. This implementation supports various orders of the
/// wavelet and provides methods for calculating wavelet values and decomposing signals using the
/// wavelet transform.
/// </para>
/// <para><b>For Beginners:</b> A wavelet is a special type of mathematical function that can help analyze data.
/// 
/// Think of wavelets like special magnifying glasses that can zoom in on different parts of your data:
/// - They can detect patterns at different scales (big patterns and small details)
/// - They're great for analyzing signals that change over time (like sound or sensor readings)
/// - They can compress data while preserving important features
/// 
/// The Fejér-Korovkin wavelet is a specific type of wavelet with smooth properties that make it
/// useful for various applications in signal processing, image analysis, and data compression.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public partial class FejérKorovkinWavelet<T> : WaveletFunctionBase<T>
{

    /// <summary>
    /// The order of the Fejér-Korovkin wavelet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the order of the wavelet, which determines the number of coefficients
    /// and the overall characteristics of the wavelet. Higher orders generally provide better
    /// frequency localization but with increased computational complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the "resolution setting" for the wavelet.
    /// 
    /// The order:
    /// - Controls how many coefficients are used in calculations
    /// - Affects how precisely the wavelet can analyze different frequencies
    /// - Higher values give more precise results but require more computation
    /// 
    /// Think of it like pixels in a camera - more pixels (higher order) gives you a more
    /// detailed picture, but the file size is larger and processing takes longer.
    /// </para>
    /// </remarks>
    private readonly int _order;

    /// <summary>
    /// The Fejér-Korovkin wavelet coefficients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the precomputed Fejér-Korovkin coefficients that define the specific
    /// characteristics of this wavelet. These coefficients are calculated during initialization
    /// based on the specified order and are used in the wavelet function calculations.
    /// </para>
    /// <para><b>For Beginners:</b> These are the special numbers that define this particular wavelet's "shape".
    /// 
    /// The coefficients:
    /// - Are calculated using mathematical formulas
    /// - Give the wavelet its unique properties
    /// - Are used in all calculations involving this wavelet
    /// 
    /// This is like the DNA of the wavelet - these specific values are what make
    /// a Fejér-Korovkin wavelet different from other types of wavelets.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _coefficients;

    /// <summary>
    /// The scaling coefficients used for signal decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the scaling coefficients (also known as low-pass filter coefficients) used
    /// in the wavelet transform to calculate the approximation components of a signal during decomposition.
    /// These coefficients are initialized during object construction and normalized for proper energy conservation.
    /// </para>
    /// <para><b>For Beginners:</b> These coefficients help extract the "big picture" from your data.
    /// 
    /// The scaling coefficients:
    /// - Work like a smoothing filter on your data
    /// - Capture the overall trends and low-frequency components
    /// - Help create the "approximation" part of the decomposed signal
    /// 
    /// Think of these as a filter that removes the fine details and keeps only the general shape,
    /// like looking at something through frosted glass where you can see outlines but not details.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _scalingCoefficients;

    /// <summary>
    /// The wavelet coefficients used for signal decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the wavelet coefficients (also known as high-pass filter coefficients) used
    /// in the wavelet transform to calculate the detail components of a signal during decomposition.
    /// These coefficients are initialized during object construction and normalized for proper energy conservation.
    /// </para>
    /// <para><b>For Beginners:</b> These coefficients help extract the fine details from your data.
    /// 
    /// The wavelet coefficients:
    /// - Work like a detail-enhancing filter on your data
    /// - Capture the rapid changes and high-frequency components
    /// - Help create the "detail" part of the decomposed signal
    /// 
    /// Think of these as a filter that removes the overall shape and keeps only the fine details,
    /// like an edge detection filter that highlights boundaries and textures in an image.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _waveletCoefficients;

    /// <summary>
    /// Initializes a new instance of the <see cref="FejérKorovkinWavelet{T}"/> class with the specified order.
    /// </summary>
    /// <param name="order">The order of the Fejér-Korovkin wavelet. Must be an even number greater than or equal to 4. Defaults to 4.</param>
    /// <exception cref="ArgumentException">Thrown when the order is less than 4 or not an even number.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes the wavelet with the specified order, which determines the number of
    /// coefficients and the wavelet's properties. Higher orders generally provide better frequency localization
    /// but at the cost of increased computational complexity.
    /// </para>
    /// <para><b>For Beginners:</b> The order parameter controls how complex the wavelet will be.
    /// 
    /// Think of the order like the "resolution" of your analysis tool:
    /// - Lower order (4, 6, 8): Faster calculations but less precise analysis
    /// - Higher order (10, 12, 14): More detailed analysis but slower calculations
    /// 
    /// The default order of 4 works well for many applications, but you can increase it
    /// if you need more precise analysis of your data.
    /// </para>
    /// </remarks>
    public FejérKorovkinWavelet(int order = 4)
    {
        _order = order;
        _coefficients = GetFejérKorovkinCoefficients(_order);
        _scalingCoefficients = new Vector<T>(_order);
        _waveletCoefficients = new Vector<T>(_order);
        InitializeCoefficients();
    }

    /// <summary>
    /// Calculates the wavelet function value at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to calculate the wavelet value.</param>
    /// <returns>The calculated wavelet function value at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the value of the wavelet function at the given input point by applying
    /// the Fejér-Korovkin coefficients to the scaling function. The result represents how much the
    /// signal at that point contributes to the frequency band represented by this wavelet.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how strongly a specific point in your data
    /// matches the wavelet pattern.
    /// 
    /// When you use this method:
    /// - You provide a point (x) in your data
    /// - The method returns a value indicating how well the wavelet matches your data at that point
    /// - Higher values mean stronger matches
    /// 
    /// This is like asking: "How much does my data at this point look like this specific pattern?"
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        if (_coefficients.Length == 0)
        {
            return NumOps.Zero;
        }

        var engine = AiDotNetEngine.Current;

        // sum_k c[k] * phi(x - k) with the box scaling function phi(t) = 1 for 0 <= t < 1, else 0.
        var shifts = engine.TensorArange<T>(0, _coefficients.Length, 1);
        var shifted = engine.ScalarMinusTensor(x, shifts);
        var notNegative = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(shifted, NumOps.Zero));
        var box = engine.TensorMultiply(notNegative, engine.TensorLessThan(shifted, NumOps.One));
        return engine.TensorSum(engine.TensorMultiply(Tensor<T>.FromVector(_coefficients), box));
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the wavelet transform.
    /// </summary>
    /// <param name="input">The input signal to decompose.</param>
    /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
    /// <exception cref="ArgumentException">Thrown when the input vector length is less than the wavelet order.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the discrete wavelet transform, which decomposes the input signal into
    /// two components: approximation coefficients (low-frequency components) and detail coefficients
    /// (high-frequency components). The approximation coefficients represent the overall shape of the signal,
    /// while the detail coefficients capture the fine details.
    /// </para>
    /// <para><b>For Beginners:</b> This method breaks down your data into two parts: the big picture and the details.
    /// 
    /// Think of it like analyzing a photo:
    /// - Approximation coefficients: The blurry, overall shape (like a thumbnail)
    /// - Detail coefficients: The fine details that make the image sharp
    /// 
    /// When you decompose a signal:
    /// - You can analyze the major trends separately from the small variations
    /// - You can compress data by keeping only the important parts
    /// - You can filter out noise by removing unwanted detail coefficients
    /// 
    /// The input must have at least as many elements as the wavelet order.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length < _order)
        {
            throw new ArgumentException($"Input vector must have at least {_order} elements.");
        }

        int outputLength = input.Length / 2;

        // a[i] = sum_j h[j] * x[(2i + j) mod n], d[i] = sum_j g[j] * x[(2i + j) mod n]
        var approximation = WaveletEngineOps<T>.PeriodicCorrelate(input, _scalingCoefficients, 2, outputLength);
        var detail = WaveletEngineOps<T>.PeriodicCorrelate(input, _waveletCoefficients, 2, outputLength);
        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling coefficients used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the scaling coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the scaling coefficients used in the wavelet transform. These coefficients
    /// are used to calculate the approximation (low-frequency) components of the signal during decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the "low-pass filter" values used in the wavelet transform.
    /// 
    /// The scaling coefficients:
    /// - Help capture the overall shape or trend in your data
    /// - Act like a smoothing filter that keeps the main features
    /// - Are used to create the "approximation" part when decomposing a signal
    /// 
    /// You might need these coefficients if you want to implement your own wavelet algorithms
    /// or understand the mathematical details of how this wavelet works.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        return _scalingCoefficients;
    }

    /// <summary>
    /// Gets the wavelet coefficients used in the wavelet transform.
    /// </summary>
    /// <returns>A vector containing the wavelet coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the wavelet coefficients used in the wavelet transform. These coefficients
    /// are used to calculate the detail (high-frequency) components of the signal during decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This method gives you the "high-pass filter" values used in the wavelet transform.
    /// 
    /// The wavelet coefficients:
    /// - Help capture the detailed variations or changes in your data
    /// - Act like a detail-enhancing filter that highlights quick changes
    /// - Are used to create the "detail" part when decomposing a signal
    /// 
    /// These coefficients are particularly useful for detecting edges, sudden changes,
    /// or high-frequency components in your data.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        return _waveletCoefficients;
    }

    /// <summary>
    /// Evaluates the scaling function at the specified point.
    /// </summary>
    /// <param name="x">The input point at which to evaluate the scaling function.</param>
    /// <returns>The value of the scaling function at the specified point.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the scaling function (also known as the father wavelet), which is a
    /// box function that returns 1 for inputs in the range [0,1) and 0 elsewhere. The scaling function
    /// forms the basis for constructing the wavelet function.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method checks if a point falls within a specific range.
    /// 
    /// The scaling function is very simple:
    /// - It returns 1 if x is between 0 (inclusive) and 1 (exclusive)
    /// - It returns 0 for all other values
    /// 
    /// This is like a simple "detector" that only activates when a value is in a specific range.
    /// It serves as a building block for the more complex wavelet functions.
    /// </para>
    /// </remarks>
    private T ScalingFunction(T x)
    {
        if (NumOps.GreaterThanOrEquals(x, NumOps.Zero) && NumOps.LessThan(x, NumOps.One))
        {
            return NumOps.One;
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Calculates the Fejér-Korovkin coefficients for the specified order.
    /// </summary>
    /// <param name="order">The order of the Fejér-Korovkin wavelet.</param>
    /// <returns>A vector containing the calculated Fejér-Korovkin coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the order is less than 4 or not an even number.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the Fejér-Korovkin coefficients for the specified order using the
    /// mathematical definition of these wavelets. The coefficients are calculated using trigonometric
    /// functions and then normalized to ensure proper scaling.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the special numbers that define this particular wavelet.
    /// 
    /// The Fejér-Korovkin coefficients:
    /// - Are calculated using mathematical formulas based on trigonometric functions
    /// - Define the specific "shape" of this wavelet
    /// - Are normalized so they add up correctly (sum to 1)
    /// 
    /// You don't need to understand the math to use the wavelet, but these coefficients
    /// are what make this wavelet unique compared to other types of wavelets.
    /// </para>
    /// </remarks>
    private Vector<T> GetFejérKorovkinCoefficients(int order)
    {
        if (order < 4 || order % 2 != 0)
        {
            throw new ArgumentException("Order must be an even number greater than or equal to 4.");
        }

        var engine = AiDotNetEngine.Current;
        int n = order / 2;
        int count = 2 * n + 1;

        // The coefficients are computed in double and converted once: their exact sum is zero, and the 1e-15 zero
        // test only holds for double cancellation residue (float leaves ~1.5e-8, and dividing by it blows the
        // coefficients up to ~1e6).
        // c_k = sum_{j=1..n} (1 - cos(theta_j)) * cos(2 k theta_j) / (2n + 1), theta_j = pi j / (2n + 1), k = -n .. n
        var theta = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<double>(1, n + 1, 1), Math.PI), (double)count);
        var twiceK = engine.TensorMultiplyScalar(engine.TensorArange<double>(-n, n + 1, 1), 2.0);
        var weights = engine.ScalarMinusTensor(1.0, engine.TensorCos(theta));
        var terms = engine.TensorMultiply(
            engine.TensorCos(engine.TensorOuter(twiceK, theta)),
            engine.TensorBroadcastTo(weights.Reshape(1, n), new[] { count, n }));
        var coefficients = engine.ReduceSum(engine.TensorDivideScalar(terms, (double)count), new[] { 1 }, false);

        // Normalize coefficients (or fall back to a uniform filter when they sum to zero)
        double sum = engine.TensorSum(coefficients);
        if (Math.Abs(sum) > 1e-15)
        {
            return engine.TensorDivideScalar(coefficients, sum).Cast<T>().ToVector();
        }

        return engine.TensorFullLike(coefficients, 1.0 / count).Cast<T>().ToVector();
    }

    /// <summary>
    /// Initializes the scaling and wavelet coefficients used for signal decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the scaling and wavelet coefficients that are used in the wavelet
    /// transform for signal decomposition. The scaling coefficients are related to low-pass filtering,
    /// while the wavelet coefficients are related to high-pass filtering. Both sets of coefficients
    /// are normalized to ensure proper energy conservation.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the filters used to break down signals.
    /// 
    /// The initialization:
    /// - Creates two sets of coefficients (scaling and wavelet)
    /// - Uses trigonometric functions (sine and cosine) to define them
    /// - Normalizes them to ensure mathematical correctness
    /// 
    /// These coefficients work together:
    /// - Scaling coefficients capture the "smooth" parts of your data
    /// - Wavelet coefficients capture the "detailed" parts
    /// 
    /// This setup happens automatically when you create a new FejérKorovkinWavelet object.
    /// </para>
    /// </remarks>
    private void InitializeCoefficients()
    {
        var engine = AiDotNetEngine.Current;
        if (_order == 4)
        {
            // fk4 (Nielsen 2001; MATLAB fejerkorovkin('fk4')): an orthonormal scaling filter (sum sqrt(2), unit norm,
            // orthogonal to its even shifts) with its quadrature-mirror high-pass, so Reconstruct inverts Decompose.
            _scalingCoefficients = new Tensor<double>(
                new[] { 0.6539275555697651, 0.7532724928394872, 0.0531792287790598, -0.0461657148152177 },
                new[] { 4 }).Cast<T>().ToVector();
            _waveletCoefficients = WaveletEngineOps<T>.QuadratureMirror(_scalingCoefficients);
            return;
        }

        // t_k = (k + 0.5) * pi / order; scaling sqrt(2 / order) * cos(t_k), wavelet sqrt(2 / order) * sin(t_k)
        T amplitude = NumOps.FromDouble(Math.Sqrt(2.0 / _order));
        var t = engine.TensorDivideScalar(
            engine.TensorMultiplyScalar(
                engine.TensorAddScalar(engine.TensorArange<T>(0, _order, 1), NumOps.FromDouble(0.5)),
                NumOps.FromDouble(Math.PI)),
            NumOps.FromDouble(_order));
        _scalingCoefficients = NormalizeCoefficients(engine.TensorMultiplyScalar(engine.TensorCos(t), amplitude).ToVector());
        _waveletCoefficients = NormalizeCoefficients(engine.TensorMultiplyScalar(engine.TensorSin(t), amplitude).ToVector());
    }

    /// <summary>
    /// Normalizes a set of coefficients to ensure they have unit energy.
    /// </summary>
    /// <param name="coefficients">The vector of coefficients to normalize.</param>
    /// <remarks>
    /// <para>
    /// This method normalizes the provided coefficients by dividing each element by the square root
    /// of the sum of squares of all elements. This ensures that the coefficients have unit energy,
    /// which is important for preserving energy during wavelet transforms.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the coefficient values to ensure mathematical correctness.
    /// 
    /// Normalization:
    /// - Makes sure the energy (sum of squared values) equals 1
    /// - Prevents the transform from amplifying or reducing signal energy incorrectly
    /// - Ensures consistent results regardless of scale
    /// 
    /// Think of it like calibrating a measuring tool:
    /// - Before using a scale, you make sure it shows zero when nothing is on it
    /// - Similarly, normalization "calibrates" the coefficients to ensure accurate analysis
    /// </para>
    /// </remarks>
    private static Vector<T> NormalizeCoefficients(Vector<T> coefficients)
    {
        var engine = AiDotNetEngine.Current;
        T sum = engine.DotProduct(coefficients, coefficients);

        // Avoid division by zero
        if (NumOps.LessThanOrEquals(sum, NumOps.FromDouble(1e-15)))
        {
            return coefficients;
        }

        T normalizationFactor = NumOps.Sqrt(NumOps.Divide(NumOps.One, sum));
        return engine.Multiply(coefficients, normalizationFactor);
    }

    /// <summary>
    /// Reconstructs the original signal from approximation and detail coefficients.
    /// </summary>
    /// <param name="approximation">The approximation coefficients from decomposition.</param>
    /// <param name="detail">The detail coefficients from decomposition.</param>
    /// <returns>The reconstructed signal.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method reverses the decomposition process to get back the original signal.
    ///
    /// For Fejer-Korovkin wavelets, reconstruction works by:
    /// 1. Upsampling (inserting zeros between each coefficient)
    /// 2. Convolving with time-reversed reconstruction filters
    /// 3. Adding the approximation and detail contributions together
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">
    /// The order is not 4: only the fk4 taps are an orthonormal filter bank here, and synthesis with any other
    /// filters would not return the decomposed signal.
    /// </exception>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        if (_order != 4)
        {
            throw new NotSupportedException(
                "Fejer-Korovkin reconstruction is available for order 4 (fk4) only; other orders have no orthonormal filter bank here.");
        }

        // Adjoint of the periodic analysis in Decompose:
        // x[m] = sum_j h[j] * up(a)[(m - j) mod n] + g[j] * up(d)[(m - j) mod n].
        return WaveletEngineOps<T>.PeriodicSynthesis(approximation, detail, _scalingCoefficients, _waveletCoefficients);
    }
}
