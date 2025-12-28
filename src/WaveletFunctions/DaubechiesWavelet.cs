namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements Daubechies wavelets, which are a family of orthogonal wavelets characterized by
/// a maximal number of vanishing moments for a given support width.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Daubechies wavelets, named after mathematician Ingrid Daubechies, are a family of orthogonal
/// wavelets with compact support and a maximal number of vanishing moments for a given support width.
/// They are widely used in signal processing, image compression, and numerical analysis.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Daubechies wavelets are like special mathematical magnifying glasses that can analyze
/// signals at different levels of detail. They're named after Ingrid Daubechies, a mathematician
/// who made groundbreaking contributions to wavelet theory.
/// 
/// Key features of Daubechies wavelets:
/// - They have compact support (affect only a limited region)
/// - They can have a specified number of vanishing moments
/// - They're orthogonal (no redundancy in the transform)
/// - They're asymmetric (unlike some other wavelets)
/// 
/// "Vanishing moments" means the wavelet can ignore certain polynomial trends in the data.
/// For example, a wavelet with 2 vanishing moments will be "blind" to constant and linear trends,
/// allowing it to focus on more complex patterns.
/// 
/// These wavelets are particularly useful for:
/// - Image compression (JPEG2000 uses them)
/// - Signal denoising
/// - Feature extraction
/// - Data compression
/// 
/// The order parameter (typically denoted as D2, D4, D6, etc., where the number is twice the order)
/// controls how many vanishing moments the wavelet has, with higher orders providing more
/// vanishing moments but wider support.
/// </para>
/// </remarks>
public class DaubechiesWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the Daubechies wavelet.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// The scaling function coefficients of the Daubechies wavelet.
    /// </summary>
    private readonly Vector<T> _scalingCoefficients;

    /// <summary>
    /// The wavelet function coefficients of the Daubechies wavelet.
    /// </summary>
    private readonly Vector<T> _waveletCoefficients;

    /// <summary>
    /// Initializes a new instance of the DaubechiesWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the Daubechies wavelet. Default is 4.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines the properties of the Daubechies wavelet.
    /// 
    /// In Daubechies wavelets, the order N means:
    /// - The wavelet has N/2 vanishing moments
    /// - The support width is N-1
    /// - There are N coefficients in the filter
    /// 
    /// Common orders include:
    /// - D2 (order=2): Haar wavelet, simplest form
    /// - D4 (order=4): 2 vanishing moments, support width of 3
    /// - D6 (order=6): 3 vanishing moments, support width of 5
    /// - And so on...
    /// 
    /// Higher orders create wavelets that:
    /// - Can ignore more complex polynomial trends
    /// - Are smoother
    /// - Have wider support (affect more neighboring points)
    /// 
    /// The default order of 4 (often called D4) provides a good balance for most applications.
    /// It has 2 vanishing moments and a support width of 3, making it effective yet computationally
    /// efficient.
    /// 
    /// Note: In some literature, Daubechies wavelets are denoted as DbN where N is the order,
    /// so D4 would be Db4.
    /// </para>
    /// </remarks>
    public DaubechiesWavelet(int order = 4)
    {
        _order = order;
        _scalingCoefficients = ComputeScalingCoefficients();
        _waveletCoefficients = ComputeWaveletCoefficients();
    }

    /// <summary>
    /// Calculates the value of the Daubechies wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the Daubechies wavelet at a specific point.
    /// 
    /// Daubechies wavelets don't have a simple formula like some other wavelets.
    /// Instead, they're defined implicitly through their scaling coefficients and a
    /// recursive relationship called the two-scale relation:
    /// 
    /// f(t) = S h_k · f(2t-k)
    /// 
    /// This method approximates the wavelet value using:
    /// 1. The cascade algorithm to compute the scaling function values
    /// 2. A linear combination of shifted scaling functions to get the wavelet value
    /// 
    /// The calculation is only valid within the support of the wavelet (0 to order-1).
    /// Outside this range, the function returns zero.
    /// 
    /// This approach provides a reasonable approximation of the wavelet function,
    /// which is useful for visualization and understanding the wavelet's shape.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        if (t < 0 || t > _order - 1)
            return NumOps.Zero;

        T result = NumOps.Zero;
        for (int k = 0; k < _scalingCoefficients.Length; k++)
        {
            double shiftedT = t - k;
            if (shiftedT >= 0 && shiftedT < 1)
            {
                result = NumOps.Add(result, NumOps.Multiply(_scalingCoefficients[k], NumOps.FromDouble(CascadeAlgorithm(shiftedT))));
            }
        }

        return result;
    }

    /// <summary>
    /// Implements the cascade algorithm to approximate the scaling function at point t.
    /// </summary>
    /// <param name="t">The point at which to evaluate the scaling function.</param>
    /// <param name="iterations">The number of iterations for the cascade algorithm. Default is 7.</param>
    /// <returns>The approximated value of the scaling function at point t.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The cascade algorithm is a numerical method to compute values of the scaling function
    /// when no explicit formula exists.
    /// 
    /// The scaling function satisfies a two-scale relation:
    /// f(t) = S h_k · f(2t-k)
    /// 
    /// This is a recursive definition, which makes exact calculation challenging.
    /// The cascade algorithm solves this by:
    /// 
    /// 1. Starting with a simple approximation (a single impulse)
    /// 2. Repeatedly applying the two-scale relation to refine the approximation
    /// 3. After several iterations, converging to a good approximation of the scaling function
    /// 
    /// Each iteration doubles the resolution of the approximation, which is why
    /// the number of points grows exponentially (2^iterations).
    /// 
    /// The default of 7 iterations provides a reasonable balance between accuracy
    /// and computational efficiency for most purposes.
    /// </para>
    /// </remarks>
    private double CascadeAlgorithm(double t, int iterations = 7)
    {
        if (t < 0 || t > 1)
            return 0;

        double[] values = new double[1 << iterations];
        values[0] = 1;

        for (int iter = 0; iter < iterations; iter++)
        {
            double[] newValues = new double[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
            {
                for (int k = 0; k < _scalingCoefficients.Length; k++)
                {
                    int index = (2 * i + k) % newValues.Length;
                    newValues[index] += Convert.ToDouble(_scalingCoefficients[k]) * values[i];
                }
            }
            values = newValues;
        }

        int index_t = (int)(t * values.Length);
        return values[index_t];
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the Daubechies wavelet.
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
    /// 1. The input signal is convolved (filtered) with the scaling coefficients to get the approximation
    /// 2. The input signal is convolved with the wavelet coefficients to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes Daubechies wavelets special for this task is:
    /// - Their vanishing moments allow them to ignore polynomial trends
    /// - Their orthogonality ensures no redundancy in the transform
    /// - Their compact support makes computation efficient
    /// 
    /// The requirement for even-length input ensures that the downsampling works correctly.
    /// The result has half the length of the original signal, which makes wavelet decomposition
    /// efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Daubechies wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        for (int i = 0; i < halfLength; i++)
        {
            T approx = NumOps.Zero;
            T det = NumOps.Zero;

            for (int j = 0; j < _scalingCoefficients.Length; j++)
            {
                int index = (2 * i + j) % input.Length;
                approx = NumOps.Add(approx, NumOps.Multiply(_scalingCoefficients[j], input[index]));
                det = NumOps.Add(det, NumOps.Multiply(_waveletCoefficients[j], input[index]));
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
    /// The reconstruction process for orthogonal wavelets like Daubechies:
    /// 1. Upsample the approximation and detail coefficients by inserting zeros
    /// 2. Convolve with the time-reversed reconstruction filters
    /// 3. Add the results together
    ///
    /// For orthogonal wavelets, perfect reconstruction is guaranteed when:
    /// - The filters satisfy the orthogonality conditions
    /// - The signal length is compatible with the filter length
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        int outputLength = approximation.Length * 2;
        var reconstructed = new Vector<T>(outputLength);
        int filterLength = _scalingCoefficients.Length;

        // For orthogonal wavelets, reconstruction filters are time-reversed analysis filters
        var h = _scalingCoefficients.ToArray();
        var g = _waveletCoefficients.ToArray();

        for (int i = 0; i < outputLength; i++)
        {
            T sum = NumOps.Zero;

            for (int j = 0; j < filterLength; j++)
            {
                // Upsampling: only even indices contribute
                int approxIndex = (i - j + filterLength * outputLength) / 2 % approximation.Length;
                int detailIndex = approxIndex;

                // Check if this index corresponds to an even position (after upsampling)
                if ((i - j + filterLength * outputLength) % 2 == 0)
                {
                    // Use time-reversed filters for reconstruction
                    int revJ = filterLength - 1 - j;
                    sum = NumOps.Add(sum, NumOps.Multiply(h[revJ], approximation[approxIndex]));
                    sum = NumOps.Add(sum, NumOps.Multiply(g[revJ], detail[detailIndex]));
                }
            }

            reconstructed[i] = sum;
        }

        return reconstructed;
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Daubechies wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal.
    /// 
    /// For Daubechies wavelets, these coefficients:
    /// - Are carefully designed to have specific vanishing moments
    /// - Create a low-pass filter that captures the overall shape of the signal
    /// - Satisfy specific mathematical conditions (orthogonality, etc.)
    /// 
    /// These coefficients are the foundation of the Daubechies wavelet transform.
    /// They determine the wavelet's properties like smoothness, support width,
    /// and number of vanishing moments.
    /// 
    /// The current implementation provides coefficients for the D4 wavelet,
    /// which has 2 vanishing moments and 4 coefficients.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        return _scalingCoefficients;
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Daubechies wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal.
    /// 
    /// For Daubechies wavelets, these coefficients are derived from the scaling coefficients using
    /// the quadrature mirror filter relationship:
    /// 
    /// g[n] = (-1)^n · h[L-1-n]
    /// 
    /// Where:
    /// - g[n] are the wavelet coefficients
    /// - h[n] are the scaling coefficients
    /// - L is the length of the filter
    /// - n is the index
    /// 
    /// This relationship ensures that:
    /// - The wavelet filter captures frequencies that the scaling filter misses
    /// - Together, they cover the entire frequency spectrum without gaps
    /// - The decomposition preserves the energy of the signal
    /// 
    /// The alternating signs ((-1)^n) create the oscillating nature that is characteristic of wavelets.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        return _waveletCoefficients;
    }

    /// <summary>
    /// Computes the scaling function coefficients for the Daubechies wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates the scaling coefficients for the Daubechies wavelet.
    /// 
    /// Currently, it implements the coefficients for the D4 wavelet (order=4), which are:
    /// 
    /// h[0] = (1+v3)/(4v2)
    /// h[1] = (3+v3)/(4v2)
    /// h[2] = (3-v3)/(4v2)
    /// h[3] = (1-v3)/(4v2)
    /// 
    /// These specific values were derived by Ingrid Daubechies to satisfy several
    /// mathematical conditions:
    /// 
    /// 1. Orthogonality: The wavelet transform preserves energy
    /// 2. Vanishing moments: The wavelet ignores polynomial trends up to a certain order
    /// 3. Compact support: The wavelet affects only a limited region
    /// 
    /// For a complete implementation, this method would need to be extended to
    /// calculate coefficients for other orders of Daubechies wavelets.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeScalingCoefficients()
    {
        // Coefficients for Daubechies-4 wavelet
        double[] d4Coefficients =
        [
            (1 + Math.Sqrt(3)) / (4 * Math.Sqrt(2)),
            (3 + Math.Sqrt(3)) / (4 * Math.Sqrt(2)),
            (3 - Math.Sqrt(3)) / (4 * Math.Sqrt(2)),
            (1 - Math.Sqrt(3)) / (4 * Math.Sqrt(2))
        ];

        return new Vector<T>([.. d4Coefficients.Select(c => NumOps.FromDouble(c))]);
    }

    /// <summary>
    /// Computes the wavelet function coefficients for the Daubechies wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method calculates the wavelet coefficients from the scaling coefficients
    /// using the quadrature mirror filter relationship.
    /// 
    /// The formula used is:
    /// g[n] = (-1)^n · h[L-1-n]
    /// 
    /// Where:
    /// - g[n] are the wavelet coefficients
    /// - h[n] are the scaling coefficients
    /// - L is the length of the filter
    /// - n is the index
    /// 
    /// For the D4 wavelet, this gives:
    /// g[0] = h[3]
    /// g[1] = -h[2]
    /// g[2] = h[1]
    /// g[3] = -h[0]
    /// 
    /// This relationship creates a high-pass filter that complements the low-pass filter
    /// formed by the scaling coefficients. Together, they allow the wavelet transform
    /// to separate a signal into low-frequency (approximation) and high-frequency (detail)
    /// components.
    /// 
    /// The alternating signs ensure that the wavelet coefficients form an oscillating
    /// function, which is characteristic of wavelets and essential for detecting
    /// high-frequency components in the signal.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeWaveletCoefficients()
    {
        var coeffs = _scalingCoefficients.ToArray();
        int L = coeffs.Length;
        var waveletCoeffs = new T[L];

        for (int i = 0; i < L; i++)
        {
            waveletCoeffs[i] = NumOps.Multiply(NumOps.FromDouble(Math.Pow(-1, i)), coeffs[L - 1 - i]);
        }

        return new Vector<T>(waveletCoeffs);
    }
}
