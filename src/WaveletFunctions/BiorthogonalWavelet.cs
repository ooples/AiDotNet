namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements biorthogonal wavelets, which offer symmetry and linear phase properties while maintaining
/// perfect reconstruction capabilities.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Biorthogonal wavelets use different basis functions for decomposition and reconstruction, allowing them
/// to achieve properties that are impossible with orthogonal wavelets. They are particularly useful in
/// applications where symmetry and linear phase are important, such as image processing.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Biorthogonal wavelets are special mathematical tools that offer more flexibility than standard wavelets.
/// 
/// The key difference is that biorthogonal wavelets use:
/// - One set of functions to break down (decompose) a signal
/// - A different but related set of functions to rebuild (reconstruct) it
/// 
/// This approach offers several advantages:
/// - Perfect reconstruction: The signal can be rebuilt exactly without errors
/// - Symmetry: The wavelets can be symmetric, which reduces edge artifacts
/// - Linear phase: Important for preserving the shape of features in the signal
/// 
/// These properties make biorthogonal wavelets particularly useful for:
/// - Image compression (JPEG2000 uses them)
/// - Signal denoising where preserving edges is important
/// - Applications where phase information matters
/// 
/// You can think of biorthogonal wavelets as using two complementary lenses - one for analyzing
/// and one for synthesizing - that work together perfectly.
/// </para>
/// </remarks>
public class BiorthogonalWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the wavelet used for decomposition.
    /// </summary>
    private readonly int _decompositionOrder;

    /// <summary>
    /// The order of the wavelet used for reconstruction.
    /// </summary>
    private readonly int _reconstructionOrder;

    /// <summary>
    /// Coefficients used for the decomposition process.
    /// </summary>
    private readonly Vector<T> _decompositionCoefficients;

    /// <summary>
    /// Coefficients used for the reconstruction process.
    /// </summary>
    private readonly Vector<T> _reconstructionCoefficients;

    /// <summary>
    /// Initializes a new instance of the BiorthogonalWavelet class with the specified decomposition and reconstruction orders.
    /// </summary>
    /// <param name="decompositionOrder">The order of the wavelet used for decomposition. Default is 2.</param>
    /// <param name="reconstructionOrder">The order of the wavelet used for reconstruction. Default is 2.</param>
    /// <exception cref="ArgumentException">Thrown when either order is less than 1 or greater than 6.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// When creating a biorthogonal wavelet, you specify two orders:
    ///
    /// 1. Decomposition order: Controls the wavelet used to analyze the signal
    /// 2. Reconstruction order: Controls the wavelet used to rebuild the signal
    ///
    /// The order affects the properties of the wavelet:
    /// - Higher orders create smoother wavelets
    /// - Lower orders create more compact wavelets (affecting fewer neighboring points)
    ///
    /// Common combinations include:
    /// - 1,3: Good for detecting sharp transitions with smooth reconstruction
    /// - 2,2: Balanced between decomposition and reconstruction
    /// - 3,1: Smooth analysis with compact reconstruction
    /// - 4,4: Used in JPEG2000 (CDF 9/7 wavelet)
    /// - 5,5: Higher smoothness
    /// - 6,8: Also used in JPEG2000
    ///
    /// This implementation supports orders 1 through 6 for both decomposition and reconstruction.
    /// The constructor initializes the appropriate coefficient sets based on these orders.
    /// </para>
    /// </remarks>
    public BiorthogonalWavelet(int decompositionOrder = 2, int reconstructionOrder = 2)
    {
        if (decompositionOrder < 1 || decompositionOrder > 6)
            throw new ArgumentException("Order must be between 1 and 6.", nameof(decompositionOrder));

        if (reconstructionOrder < 1 || reconstructionOrder > 6)
            throw new ArgumentException("Order must be between 1 and 6.", nameof(reconstructionOrder));

        _decompositionOrder = decompositionOrder;
        _reconstructionOrder = reconstructionOrder;
        _decompositionCoefficients = GetDecompositionCoefficients(_decompositionOrder);
        _reconstructionCoefficients = GetReconstructionCoefficients(_reconstructionOrder);
    }

    /// <summary>
    /// Calculates the value of the biorthogonal wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the wavelet function at a specific point.
    /// 
    /// The biorthogonal wavelet is constructed as a linear combination of shifted scaling functions:
    /// 1. It starts with a basic scaling function (a simple box function in this implementation)
    /// 2. It shifts this function to different positions
    /// 3. It multiplies each shifted function by the corresponding coefficient
    /// 4. It sums these weighted functions to get the final result
    /// 
    /// This creates a wavelet function with the desired properties of symmetry and compact support.
    /// 
    /// You might use this method to visualize the wavelet or to directly apply the wavelet
    /// to a signal at specific points.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        T result = NumOps.Zero;
        for (int k = 0; k < _decompositionCoefficients.Length; k++)
        {
            T shiftedX = NumOps.Subtract(x, NumOps.FromDouble(k));
            result = NumOps.Add(result, NumOps.Multiply(_decompositionCoefficients[k], ScalingFunction(shiftedX)));
        }

        return result;
    }

    /// <summary>
    /// Evaluates the basic scaling function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the scaling function.</param>
    /// <returns>The value of the scaling function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling function is the basic building block used to construct the wavelet.
    /// 
    /// This implementation uses a simple box function as the scaling function:
    /// - It equals 1 when x is between 0 and 1
    /// - It equals 0 everywhere else
    /// 
    /// This is the simplest possible scaling function, and it serves as the foundation
    /// for constructing the more complex biorthogonal wavelets.
    /// 
    /// The actual wavelet is created by taking weighted sums of shifted versions of this
    /// basic scaling function, with the weights determined by the decomposition coefficients.
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
    /// Gets the decomposition coefficients for the specified order.
    /// </summary>
    /// <param name="order">The order of the decomposition wavelet.</param>
    /// <returns>A vector of decomposition coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the specified order is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// These coefficients define the wavelet used for decomposing (analyzing) a signal.
    /// 
    /// The method provides pre-calculated coefficients for three different orders:
    /// 
    /// - Order 1: The simplest wavelet, equivalent to the Haar wavelet
    ///   - Good for detecting abrupt changes
    ///   - Less smooth, more localized in time
    /// 
    /// - Order 2: A more balanced wavelet
    ///   - Better frequency localization than order 1
    ///   - Still reasonably compact in time
    /// 
    /// - Order 3: The smoothest option
    ///   - Best frequency localization
    ///   - Less localized in time (affects more neighboring points)
    /// 
    /// These coefficients are carefully designed to work with the corresponding reconstruction
    /// coefficients to ensure perfect reconstruction of the signal.
    /// </para>
    /// </remarks>
    private Vector<T> GetDecompositionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>([NumOps.FromDouble(0.7071067811865476), NumOps.FromDouble(0.7071067811865476)]),
            2 => new Vector<T>([ NumOps.FromDouble(-0.1767766952966369), NumOps.FromDouble(0.3535533905932738),
                                             NumOps.FromDouble(1.0606601717798214), NumOps.FromDouble(0.3535533905932738),
                                             NumOps.FromDouble(-0.1767766952966369) ]),
            3 => new Vector<T>([ NumOps.FromDouble(0.0352262918857095), NumOps.FromDouble(-0.0854412738820267),
                                             NumOps.FromDouble(-0.1350110200102546), NumOps.FromDouble(0.4598775021184914),
                                             NumOps.FromDouble(0.8068915093110924), NumOps.FromDouble(0.3326705529500826),
                                             NumOps.FromDouble(-0.0279837694169839) ]),
            // CDF 9/7 wavelet coefficients (bior4.4) - used in JPEG2000
            4 => new Vector<T>([ NumOps.FromDouble(0.026748757411),  NumOps.FromDouble(-0.016864118443),
                                             NumOps.FromDouble(-0.078223266529), NumOps.FromDouble(0.266864118443),
                                             NumOps.FromDouble(0.602949018236),  NumOps.FromDouble(0.266864118443),
                                             NumOps.FromDouble(-0.078223266529), NumOps.FromDouble(-0.016864118443),
                                             NumOps.FromDouble(0.026748757411) ]),
            // bior5.5 coefficients
            5 => new Vector<T>([ NumOps.FromDouble(0.01345671), NumOps.FromDouble(-0.00269497),
                                             NumOps.FromDouble(-0.13670658), NumOps.FromDouble(0.09350469),
                                             NumOps.FromDouble(0.47680327), NumOps.FromDouble(0.89950611),
                                             NumOps.FromDouble(0.47680327), NumOps.FromDouble(0.09350469),
                                             NumOps.FromDouble(-0.13670658), NumOps.FromDouble(-0.00269497),
                                             NumOps.FromDouble(0.01345671) ]),
            // bior6.8 coefficients
            6 => new Vector<T>([ NumOps.FromDouble(0.0019088317),  NumOps.FromDouble(-0.0019142862),
                                             NumOps.FromDouble(-0.0170080345), NumOps.FromDouble(0.0119509435),
                                             NumOps.FromDouble(0.0498175178),  NumOps.FromDouble(-0.0771721906),
                                             NumOps.FromDouble(-0.0940779761), NumOps.FromDouble(0.4207962846),
                                             NumOps.FromDouble(0.8259229975),  NumOps.FromDouble(0.4207962846),
                                             NumOps.FromDouble(-0.0940779761), NumOps.FromDouble(-0.0771721906),
                                             NumOps.FromDouble(0.0498175178),  NumOps.FromDouble(0.0119509435),
                                             NumOps.FromDouble(-0.0170080345), NumOps.FromDouble(-0.0019142862),
                                             NumOps.FromDouble(0.0019088317) ]),
            _ => throw new ArgumentException($"Biorthogonal wavelet of order {order} is not implemented."),
        };
    }

    /// <summary>
    /// Gets the reconstruction coefficients for the specified order.
    /// </summary>
    /// <param name="order">The order of the reconstruction wavelet.</param>
    /// <returns>A vector of reconstruction coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the specified order is not implemented.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// These coefficients define the wavelet used for reconstructing (synthesizing) a signal.
    /// 
    /// The reconstruction coefficients are designed to work with the decomposition coefficients
    /// to ensure perfect reconstruction. They have a special mathematical relationship with
    /// the decomposition coefficients that ensures:
    /// 
    /// 1. When you decompose a signal and then reconstruct it, you get exactly the original signal back
    /// 2. The decomposition and reconstruction processes are stable and well-behaved
    /// 
    /// For order 1, the reconstruction coefficients are identical to the decomposition coefficients.
    /// For higher orders, they differ but maintain the biorthogonality relationship.
    /// 
    /// The choice of reconstruction order affects how smooth the reconstructed signal will be,
    /// with higher orders producing smoother results.
    /// </para>
    /// </remarks>
    private Vector<T> GetReconstructionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>([NumOps.FromDouble(0.7071067811865476), NumOps.FromDouble(0.7071067811865476)]),
            2 => new Vector<T>([ NumOps.FromDouble(0.3535533905932738), NumOps.FromDouble(1.0606601717798214),
                                             NumOps.FromDouble(0.3535533905932738), NumOps.FromDouble(-0.1767766952966369) ]),
            3 => new Vector<T>([ NumOps.FromDouble(-0.0279837694169839), NumOps.FromDouble(0.3326705529500826),
                                             NumOps.FromDouble(0.8068915093110924), NumOps.FromDouble(0.4598775021184914),
                                             NumOps.FromDouble(-0.1350110200102546), NumOps.FromDouble(-0.0854412738820267),
                                             NumOps.FromDouble(0.0352262918857095) ]),
            // CDF 9/7 reconstruction coefficients (bior4.4)
            4 => new Vector<T>([ NumOps.FromDouble(-0.045635881557),  NumOps.FromDouble(-0.028771763114),
                                             NumOps.FromDouble(0.295635881557),  NumOps.FromDouble(0.557543526229),
                                             NumOps.FromDouble(0.295635881557),  NumOps.FromDouble(-0.028771763114),
                                             NumOps.FromDouble(-0.045635881557) ]),
            // bior5.5 reconstruction coefficients
            5 => new Vector<T>([ NumOps.FromDouble(-0.01345671), NumOps.FromDouble(-0.00269497),
                                             NumOps.FromDouble(0.13670658), NumOps.FromDouble(0.09350469),
                                             NumOps.FromDouble(-0.47680327), NumOps.FromDouble(0.89950611),
                                             NumOps.FromDouble(-0.47680327), NumOps.FromDouble(0.09350469),
                                             NumOps.FromDouble(0.13670658), NumOps.FromDouble(-0.00269497),
                                             NumOps.FromDouble(-0.01345671) ]),
            // bior6.8 reconstruction coefficients
            6 => new Vector<T>([ NumOps.FromDouble(0.0019088317),  NumOps.FromDouble(0.0019142862),
                                             NumOps.FromDouble(-0.0170080345), NumOps.FromDouble(-0.0119509435),
                                             NumOps.FromDouble(0.0498175178),  NumOps.FromDouble(0.0771721906),
                                             NumOps.FromDouble(-0.0940779761), NumOps.FromDouble(-0.4207962846),
                                             NumOps.FromDouble(0.8259229975),  NumOps.FromDouble(-0.4207962846),
                                             NumOps.FromDouble(-0.0940779761), NumOps.FromDouble(0.0771721906),
                                             NumOps.FromDouble(0.0498175178),  NumOps.FromDouble(-0.0119509435),
                                             NumOps.FromDouble(-0.0170080345), NumOps.FromDouble(0.0019142862),
                                             NumOps.FromDouble(0.0019088317) ]),
            _ => throw new ArgumentException($"Biorthogonal wavelet of order {order} is not implemented."),
        };
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the biorthogonal wavelet.
    /// </summary>
    /// <param name="input">The input signal vector to decompose.</param>
    /// <returns>A tuple containing the approximation coefficients and detail coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the input length is not even.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method performs one level of wavelet decomposition on your signal, splitting it into:
    /// 
    /// - Approximation coefficients: Represent the low-frequency components (the overall shape)
    /// - Detail coefficients: Represent the high-frequency components (the fine details)
    /// 
    /// The process works like this:
    /// 1. The input signal is convolved (filtered) with low-pass decomposition filters to get approximation coefficients
    /// 2. The input signal is convolved with high-pass decomposition filters to get detail coefficients
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes biorthogonal wavelets special is that:
    /// - The filters are symmetric, which reduces edge artifacts
    /// - The decomposition and reconstruction filters are different but complementary
    /// - They still achieve perfect reconstruction despite this flexibility
    /// 
    /// The result has half the length of the original signal (due to downsampling),
    /// which is why the input length must be even.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for biorthogonal wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        var decompositionLowPass = GetDecompositionLowPassFilter();
        var decompositionHighPass = GetDecompositionHighPassFilter();

        for (int i = 0; i < halfLength; i++)
        {
            T approx = NumOps.Zero;
            T det = NumOps.Zero;

            for (int j = 0; j < decompositionLowPass.Length; j++)
            {
                int index = (2 * i + j) % input.Length;
                approx = NumOps.Add(approx, NumOps.Multiply(decompositionLowPass[j], input[index]));
                det = NumOps.Add(det, NumOps.Multiply(decompositionHighPass[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
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
    /// For biorthogonal wavelets, the reconstruction uses different filters than decomposition:
    /// 1. Upsample the approximation and detail coefficients by inserting zeros
    /// 2. Convolve with the reconstruction low-pass and high-pass filters
    /// 3. Add the results together
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        int outputLength = approximation.Length * 2;
        var reconstructed = new Vector<T>(outputLength);

        var reconLowPass = GetReconstructionLowPassFilter();
        var reconHighPass = GetReconstructionHighPassFilter();
        int filterLength = reconLowPass.Length;

        for (int i = 0; i < outputLength; i++)
        {
            T sum = NumOps.Zero;

            for (int j = 0; j < filterLength; j++)
            {
                int k = i - j;
                if (k >= 0 && k % 2 == 0)
                {
                    int coeffIndex = k / 2;
                    if (coeffIndex < approximation.Length)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(reconLowPass[j], approximation[coeffIndex]));
                        sum = NumOps.Add(sum, NumOps.Multiply(reconHighPass[j], detail[coeffIndex]));
                    }
                }
            }

            reconstructed[i] = sum;
        }

        return reconstructed;
    }

    /// <summary>
    /// Gets the scaling function coefficients for the biorthogonal wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling function coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) during reconstruction.
    ///
    /// In biorthogonal wavelets, these coefficients:
    /// - Are used during the reconstruction phase
    /// - Are different from the decomposition coefficients
    /// - Are designed to work perfectly with the decomposition coefficients
    /// 
    /// This method returns the low-pass reconstruction filter, which is used to rebuild
    /// the signal from its approximation coefficients.
    /// 
    /// These coefficients are carefully designed to ensure that when combined with the
    /// wavelet coefficients, they can perfectly reconstruct the original signal.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        return GetReconstructionLowPassFilter();
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the biorthogonal wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet function coefficients are the filter weights used to extract the high-frequency
    /// components (details) during reconstruction.
    /// 
    /// In biorthogonal wavelets, these coefficients:
    /// - Are used during the reconstruction phase
    /// - Work together with the scaling coefficients
    /// - Are designed to capture the detail information
    /// 
    /// This method returns the high-pass reconstruction filter, which is used to rebuild
    /// the signal from its detail coefficients.
    /// 
    /// When properly combined with the scaling coefficients and applied to the decomposed
    /// signal, these coefficients allow perfect reconstruction of the original signal.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        return GetReconstructionHighPassFilter();
    }

    /// <summary>
    /// Gets the low-pass filter coefficients used for decomposition.
    /// </summary>
    /// <returns>A vector of low-pass filter coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the coefficients for the low-pass filter used during decomposition.
    /// 
    /// The low-pass filter:
    /// - Lets through the low-frequency components of the signal
    /// - Blocks the high-frequency components
    /// - Produces the approximation coefficients
    /// 
    /// In this implementation, the filter coefficients are pre-calculated for a specific
    /// biorthogonal wavelet family (9/7 wavelet, commonly used in JPEG2000).
    /// 
    /// These coefficients are carefully designed to:
    /// - Be symmetric (reducing edge artifacts)
    /// - Have good frequency separation properties
    /// - Work perfectly with the reconstruction filters
    /// 
    /// The coefficients are normalized to ensure proper energy preservation during
    /// the decomposition process.
    /// </para>
    /// </remarks>
    private Vector<T> GetDecompositionLowPassFilter()
    {
        double[] coeffs = _decompositionOrder switch
        {
            1 => new[] { 0.7071067811865476, 0.7071067811865476 },
            2 => new[] { -0.1767766952966369, 0.3535533905932738, 1.0606601717798214, 0.3535533905932738, -0.1767766952966369 },
            3 => new[] { 0.0352262918857095, -0.0854412738820267, -0.1350110200102546, 0.4598775021184914,
                         0.8068915093110924, 0.3326705529500826, -0.0279837694169839 },
            4 => new[] { 0.026748757410810, -0.016864118442875, -0.078223266528990, 0.266864118442872,
                         0.602949018236360, 0.266864118442872, -0.078223266528990, -0.016864118442875, 0.026748757410810 },
            5 => new[] { 0.01345671, -0.00269497, -0.13670658, 0.09350469, 0.47680327, 0.89950611,
                         0.47680327, 0.09350469, -0.13670658, -0.00269497, 0.01345671 },
            6 => new[] { 0.0019088317, -0.0019142862, -0.0170080345, 0.0119509435, 0.0498175178, -0.0771721906,
                         -0.0940779761, 0.4207962846, 0.8259229975, 0.4207962846, -0.0940779761, -0.0771721906,
                         0.0498175178, 0.0119509435, -0.0170080345, -0.0019142862, 0.0019088317 },
            _ => throw new ArgumentException($"Order {_decompositionOrder} not supported")
        };

        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Gets the high-pass filter coefficients used for decomposition.
    /// </summary>
    /// <returns>A vector of high-pass filter coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the coefficients for the high-pass filter used during decomposition.
    /// 
    /// The high-pass filter:
    /// - Lets through the high-frequency components of the signal
    /// - Blocks the low-frequency components
    /// - Produces the detail coefficients
    /// 
    /// These coefficients are designed to complement the low-pass filter, ensuring that:
    /// - Together they cover the entire frequency spectrum
    /// - They have minimal overlap in the frequencies they capture
    /// - They preserve the energy of the signal
    /// 
    /// The filter is symmetric, which helps reduce edge artifacts when processing finite signals.
    /// Notice that the first and last coefficients are zero, which is a common feature in
    /// biorthogonal wavelet filters.
    /// </para>
    /// </remarks>
    private Vector<T> GetDecompositionHighPassFilter()
    {
        // High-pass filter derived from low-pass using quadrature mirror filter relationship
        // Must have same length as low-pass filter
        double[] coeffs = _decompositionOrder switch
        {
            1 => new[] { -0.7071067811865476, 0.7071067811865476 },
            2 => new[] { 0.1767766952966369, -0.3535533905932738, 1.0606601717798214, -0.3535533905932738, 0.1767766952966369 },
            3 => new[] { -0.0279837694169839, -0.3326705529500826, 0.8068915093110924, -0.4598775021184914,
                         -0.1350110200102546, 0.0854412738820267, 0.0352262918857095 },
            4 => new[] { 0.0, 0.091271763114250, -0.057543526228500, -0.591271763114250,
                         1.115087052456994, -0.591271763114250, -0.057543526228500, 0.091271763114250, 0.0 },
            5 => new[] { 0.01345671, 0.00269497, -0.13670658, -0.09350469, 0.47680327, -0.89950611,
                         0.47680327, -0.09350469, -0.13670658, 0.00269497, 0.01345671 },
            6 => new[] { -0.0019088317, -0.0019142862, 0.0170080345, 0.0119509435, -0.0498175178, -0.0771721906,
                         0.0940779761, 0.4207962846, -0.8259229975, 0.4207962846, 0.0940779761, -0.0771721906,
                         -0.0498175178, 0.0119509435, 0.0170080345, -0.0019142862, -0.0019088317 },
            _ => throw new ArgumentException($"Order {_decompositionOrder} not supported")
        };

        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Gets the low-pass filter coefficients used for reconstruction.
    /// </summary>
    /// <returns>A vector of low-pass filter coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the coefficients for the low-pass filter used during reconstruction.
    /// 
    /// The reconstruction low-pass filter:
    /// - Is applied to the approximation coefficients
    /// - Helps rebuild the low-frequency components of the signal
    /// - Works together with the high-pass filter to achieve perfect reconstruction
    /// 
    /// In biorthogonal wavelets, the reconstruction filters are different from the decomposition filters.
    /// They are designed to satisfy specific mathematical relationships that ensure
    /// perfect reconstruction of the original signal.
    /// 
    /// Notice that this filter has a different structure than the decomposition low-pass filter,
    /// but they are related through the biorthogonality condition.
    /// </para>
    /// </remarks>
    private Vector<T> GetReconstructionLowPassFilter()
    {
        // Reconstruction low-pass filter - time-reversed version of decomposition low-pass
        double[] coeffs = _reconstructionOrder switch
        {
            1 => new[] { 0.7071067811865476, 0.7071067811865476 },
            2 => new[] { -0.1767766952966369, 0.3535533905932738, 1.0606601717798214, 0.3535533905932738, -0.1767766952966369 },
            3 => new[] { -0.0279837694169839, 0.3326705529500826, 0.8068915093110924, 0.4598775021184914,
                         -0.1350110200102546, -0.0854412738820267, 0.0352262918857095 },
            4 => new[] { 0.0, -0.091271763114250, -0.057543526228500, 0.591271763114250,
                         1.115087052456994, 0.591271763114250, -0.057543526228500, -0.091271763114250, 0.0 },
            5 => new[] { -0.01345671, -0.00269497, 0.13670658, 0.09350469, -0.47680327, 0.89950611,
                         -0.47680327, 0.09350469, 0.13670658, -0.00269497, -0.01345671 },
            6 => new[] { 0.0019088317, 0.0019142862, -0.0170080345, -0.0119509435, 0.0498175178, 0.0771721906,
                         -0.0940779761, -0.4207962846, 0.8259229975, -0.4207962846, -0.0940779761, 0.0771721906,
                         0.0498175178, -0.0119509435, -0.0170080345, 0.0019142862, 0.0019088317 },
            _ => throw new ArgumentException($"Order {_reconstructionOrder} not supported")
        };

        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Gets the high-pass filter coefficients used for reconstruction.
    /// </summary>
    /// <returns>A vector of high-pass filter coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the coefficients for the high-pass filter used during reconstruction.
    /// 
    /// The reconstruction high-pass filter:
    /// - Is applied to the detail coefficients
    /// - Helps rebuild the high-frequency components of the signal
    /// - Works together with the low-pass filter to achieve perfect reconstruction
    /// 
    /// This filter is designed to complement the reconstruction low-pass filter.
    /// Together, they ensure that all frequency components are properly recombined
    /// to recreate the original signal without distortion.
    /// 
    /// The symmetry of these coefficients is one of the key advantages of biorthogonal wavelets,
    /// as it helps reduce phase distortion in the processed signal.
    /// </para>
    /// </remarks>
    private Vector<T> GetReconstructionHighPassFilter()
    {
        // Reconstruction high-pass filter - must have same length as reconstruction low-pass
        double[] coeffs = _reconstructionOrder switch
        {
            1 => new[] { 0.7071067811865476, -0.7071067811865476 },
            2 => new[] { 0.1767766952966369, 0.3535533905932738, -1.0606601717798214, 0.3535533905932738, 0.1767766952966369 },
            3 => new[] { 0.0352262918857095, 0.0854412738820267, -0.1350110200102546, -0.4598775021184914,
                         0.8068915093110924, -0.3326705529500826, -0.0279837694169839 },
            4 => new[] { 0.026748757410810, 0.016864118442875, -0.078223266528990, -0.266864118442872,
                         0.602949018236360, -0.266864118442872, -0.078223266528990, 0.016864118442875, 0.026748757410810 },
            5 => new[] { 0.01345671, -0.00269497, -0.13670658, 0.09350469, 0.47680327, -0.89950611,
                         0.47680327, 0.09350469, -0.13670658, -0.00269497, 0.01345671 },
            6 => new[] { 0.0019088317, -0.0019142862, -0.0170080345, 0.0119509435, 0.0498175178, -0.0771721906,
                         -0.0940779761, 0.4207962846, 0.8259229975, 0.4207962846, -0.0940779761, -0.0771721906,
                         0.0498175178, 0.0119509435, -0.0170080345, -0.0019142862, 0.0019088317 },
            _ => throw new ArgumentException($"Order {_reconstructionOrder} not supported")
        };

        return NormalizeAndConvert(coeffs);
    }

    /// <summary>
    /// Normalizes and converts an array of double coefficients to type T.
    /// </summary>
    /// <param name="coeffs">The array of double coefficients to normalize and convert.</param>
    /// <returns>A vector of normalized coefficients of type T.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This helper method performs two important tasks:
    /// 
    /// 1. Normalization: It adjusts the coefficients so that their energy (sum of squares)
    ///    equals 1. This ensures that the wavelet transform preserves the energy of the signal.
    /// 
    /// 2. Type conversion: It converts the double-precision coefficients to the generic
    ///    numeric type T used by the class.
    /// 
    /// Normalization is important because it ensures that the wavelet transform doesn't
    /// artificially amplify or reduce the signal's energy. It's a standard practice in
    /// wavelet design to ensure consistent results regardless of the specific wavelet used.
    /// 
    /// The type conversion allows the class to work with different numeric types (float, double, etc.)
    /// while maintaining the same coefficient values.
    /// </para>
    /// </remarks>
    private Vector<T> NormalizeAndConvert(double[] coeffs)
    {
        double normFactor = Math.Sqrt(coeffs.Sum(c => c * c));
        return new Vector<T>([.. coeffs.Select(c => NumOps.FromDouble(c / normFactor))]);
    }
}
