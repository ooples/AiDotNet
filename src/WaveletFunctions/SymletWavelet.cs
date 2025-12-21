namespace AiDotNet.WaveletFunctions;

/// <summary>
/// Implements Symlet wavelets, which are nearly symmetric wavelets proposed by Daubechies
/// as modifications to the Daubechies wavelets with increased symmetry.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Symlet wavelets are a family of nearly symmetric wavelets proposed by Ingrid Daubechies.
/// They are modifications of the Daubechies wavelets designed to have increased symmetry
/// while retaining most of the properties of the Daubechies wavelets.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Symlet wavelets are like improved versions of Daubechies wavelets that are more symmetric.
/// Symmetry is a desirable property in wavelets because it helps prevent phase distortion
/// when processing signals.
/// 
/// Key features of Symlet wavelets:
/// - Nearly symmetric (more symmetric than Daubechies wavelets)
/// - Orthogonal (no redundancy in the transform)
/// - Compact support (affect only a limited region)
/// - Have a specified number of vanishing moments
/// 
/// "Vanishing moments" means the wavelet can ignore certain polynomial trends in the data.
/// For example, a wavelet with 4 vanishing moments will be "blind" to cubic and lower-order
/// polynomial trends, allowing it to focus on more complex patterns.
/// 
/// These wavelets are particularly useful for:
/// - Signal and image processing where phase preservation is important
/// - Feature extraction
/// - Data compression
/// - Applications where both time and frequency localization are needed
/// 
/// The order parameter (typically denoted as sym2, sym4, sym6, etc.) controls how many
/// vanishing moments the wavelet has, with higher orders providing more vanishing moments
/// but wider support.
/// </para>
/// </remarks>
public class SymletWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// The order of the Symlet wavelet.
    /// </summary>
    private readonly int _order;

    /// <summary>
    /// The decomposition low-pass filter coefficients.
    /// </summary>
    private readonly Vector<T> _lowDecomp;

    /// <summary>
    /// The decomposition high-pass filter coefficients.
    /// </summary>
    private readonly Vector<T> _highDecomp;

    /// <summary>
    /// The reconstruction low-pass filter coefficients.
    /// </summary>
    private readonly Vector<T> _lowRecon;

    /// <summary>
    /// The reconstruction high-pass filter coefficients.
    /// </summary>
    private readonly Vector<T> _highRecon;

    /// <summary>
    /// Initializes a new instance of the SymletWavelet class with the specified order.
    /// </summary>
    /// <param name="order">The order of the Symlet wavelet. Default is 4.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The order parameter determines the properties of the Symlet wavelet.
    /// 
    /// In Symlet wavelets, the order N means:
    /// - The wavelet has N/2 vanishing moments
    /// - The support width is 2N-1
    /// - There are 2N coefficients in the filter
    /// 
    /// Common orders include:
    /// - sym2 (order=2): 1 vanishing moment, support width of 3
    /// - sym4 (order=4): 2 vanishing moments, support width of 7
    /// - sym6 (order=6): 3 vanishing moments, support width of 11
    /// - sym8 (order=8): 4 vanishing moments, support width of 15
    /// 
    /// Higher orders create wavelets that:
    /// - Can ignore more complex polynomial trends
    /// - Are smoother
    /// - Have wider support (affect more neighboring points)
    /// - Can better represent complex signals
    /// 
    /// The default order of 4 (sym4) provides a good balance for most applications.
    /// It has 2 vanishing moments and a support width of 7, making it effective yet
    /// computationally efficient.
    /// </para>
    /// </remarks>
    public SymletWavelet(int order = 4)
    {
        _order = order;
        (_lowDecomp, _highDecomp, _lowRecon, _highRecon) = GetSymletCoefficients(_order);
    }

    /// <summary>
    /// Calculates the value of the Symlet wavelet function at point x.
    /// </summary>
    /// <param name="x">The point at which to evaluate the wavelet function.</param>
    /// <returns>The value of the wavelet function at point x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the actual value of the Symlet wavelet at a specific point.
    /// 
    /// Like Daubechies wavelets, Symlet wavelets don't have a simple formula.
    /// Instead, they're defined implicitly through their filter coefficients and a
    /// recursive relationship called the two-scale relation.
    /// 
    /// This method approximates the wavelet value using the cascade algorithm:
    /// 1. Start with a delta function (a single spike)
    /// 2. Repeatedly apply the two-scale relation to refine the approximation
    /// 3. After several iterations, interpolate to find the value at the specific point
    /// 
    /// The cascade algorithm works by:
    /// - Starting with a simple approximation
    /// - Repeatedly applying the filter coefficients to refine the approximation
    /// - Doubling the resolution with each iteration
    /// 
    /// After multiple iterations, this converges to a good approximation of the wavelet function.
    /// Linear interpolation is then used to find the value at the exact point requested.
    /// </para>
    /// </remarks>
    public override T Calculate(T x)
    {
        // Approximate the wavelet function using the cascade algorithm
        int iterations = 8; // Number of iterations for approximation
        int points = 1024; // Number of points to evaluate

        var phi = new Vector<T>(points);
        phi[points / 2] = NumOps.One; // Initial delta function

        for (int i = 0; i < iterations; i++)
        {
            var newPhi = new Vector<T>(points * 2);
            for (int j = 0; j < points; j++)
            {
                for (int k = 0; k < _lowRecon.Length; k++)
                {
                    int ind = (2 * j + k) % (points * 2);
                    newPhi[ind] = NumOps.Add(newPhi[ind], NumOps.Multiply(_lowRecon[k], phi[j]));
                }
            }

            phi = new Vector<T>(newPhi.Take(points));
        }

        // Interpolate to find the value at x
        T xScaled = NumOps.Multiply(x, NumOps.FromDouble(points - 1));
        int index = (int)Convert.ToDouble(xScaled);
        T fraction = NumOps.Subtract(xScaled, NumOps.FromDouble(index));

        if (index >= points - 1)
            return phi[points - 1];

        return NumOps.Add(
            phi[index],
            NumOps.Multiply(fraction, NumOps.Subtract(phi[index + 1], phi[index]))
        );
    }

    /// <summary>
    /// Decomposes an input signal into approximation and detail coefficients using the Symlet wavelet.
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
    /// 1. The input signal is convolved (filtered) with the low-pass decomposition filter to get the approximation
    /// 2. The input signal is convolved with the high-pass decomposition filter to get the details
    /// 3. Both results are downsampled (every other value is kept)
    /// 
    /// What makes Symlet wavelets special for this task is:
    /// - Their near symmetry helps preserve phase information
    /// - Their vanishing moments allow them to ignore polynomial trends
    /// - Their orthogonality ensures no redundancy in the transform
    /// 
    /// The result has half the length of the original signal, which makes wavelet decomposition
    /// efficient for compression and multi-resolution analysis.
    /// </para>
    /// </remarks>
    public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        int n = input.Length;
        var approximation = new Vector<T>(n / 2);
        var detail = new Vector<T>(n / 2);

        for (int i = 0; i < n / 2; i++)
        {
            T approx = NumOps.Zero;
            T det = NumOps.Zero;

            for (int j = 0; j < _lowDecomp.Length; j++)
            {
                int index = (2 * i + j) % n;
                approx = NumOps.Add(approx, NumOps.Multiply(_lowDecomp[j], input[index]));
                det = NumOps.Add(det, NumOps.Multiply(_highDecomp[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
    }

    /// <summary>
    /// Gets the scaling function coefficients for the Symlet wavelet.
    /// </summary>
    /// <returns>A vector of scaling function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The scaling coefficients are the filter weights used to extract the low-frequency
    /// components (approximation) from a signal during reconstruction.
    /// 
    /// For Symlet wavelets, these coefficients:
    /// - Are designed to create wavelets with near symmetry
    /// - Create a low-pass filter that captures the overall shape of the signal
    /// - Satisfy specific mathematical conditions (orthogonality, vanishing moments)
    /// 
    /// These coefficients are the foundation of the Symlet wavelet transform.
    /// They determine the wavelet's properties like symmetry, support width,
    /// and number of vanishing moments.
    /// 
    /// In signal processing terms, these coefficients define the impulse response
    /// of the reconstruction low-pass filter used in the inverse wavelet transform.
    /// </para>
    /// </remarks>
    public override Vector<T> GetScalingCoefficients()
    {
        return _lowRecon;
    }

    /// <summary>
    /// Gets the wavelet function coefficients for the Symlet wavelet.
    /// </summary>
    /// <returns>A vector of wavelet function coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// The wavelet coefficients are the filter weights used to extract the high-frequency
    /// components (details) from a signal during reconstruction.
    /// 
    /// For Symlet wavelets, these coefficients are derived from the scaling coefficients using
    /// the quadrature mirror filter relationship, but with specific modifications to achieve
    /// near symmetry.
    /// 
    /// These coefficients:
    /// - Create a high-pass filter that captures the detailed features of the signal
    /// - Are designed to complement the scaling coefficients
    /// - Together with the scaling coefficients, allow perfect reconstruction of the signal
    /// 
    /// In signal processing terms, these coefficients define the impulse response
    /// of the reconstruction high-pass filter used in the inverse wavelet transform.
    /// </para>
    /// </remarks>
    public override Vector<T> GetWaveletCoefficients()
    {
        return _highRecon;
    }

    /// <summary>
    /// Gets the filter coefficients for the Symlet wavelet of the specified order.
    /// </summary>
    /// <param name="order">The order of the Symlet wavelet.</param>
    /// <returns>A tuple containing the decomposition low-pass, decomposition high-pass, reconstruction low-pass, and reconstruction high-pass filter coefficients.</returns>
    /// <exception cref="ArgumentException">Thrown when the specified order is not implemented or supported.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method provides the four sets of filter coefficients needed for the wavelet transform:
    /// 
    /// 1. Decomposition low-pass (lowDecomp): Used to extract approximation coefficients during analysis
    /// 2. Decomposition high-pass (highDecomp): Used to extract detail coefficients during analysis
    /// 3. Reconstruction low-pass (lowRecon): Used to reconstruct from approximation coefficients
    /// 4. Reconstruction high-pass (highRecon): Used to reconstruct from detail coefficients
    /// 
    /// For orthogonal wavelets like Symlets, these filters have special relationships:
    /// - The reconstruction filters are time-reversed versions of the decomposition filters
    /// - The high-pass filters are derived from the low-pass filters with alternating signs
    /// 
    /// The method supports several common Symlet orders (2, 4, 6, and 8), each with different
    /// properties and coefficient lengths. Higher orders have more coefficients and provide
    /// better frequency localization but wider support.
    /// 
    /// These coefficients have been carefully designed to create wavelets with:
    /// - Near symmetry
    /// - Orthogonality
    /// - Specific numbers of vanishing moments
    /// - Compact support
    /// </para>
    /// </remarks>
    private (Vector<T> lowDecomp, Vector<T> highDecomp, Vector<T> lowRecon, Vector<T> highRecon) GetSymletCoefficients(int order)
    {
        return order switch
        {
            2 => (
                    new Vector<T>(new[] {
                        NumOps.FromDouble(-0.12940952255092145),
                        NumOps.FromDouble(0.22414386804185735),
                        NumOps.FromDouble(0.836516303737469),
                        NumOps.FromDouble(0.48296291314469025)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.48296291314469025),
                        NumOps.FromDouble(0.836516303737469),
                        NumOps.FromDouble(-0.22414386804185735),
                        NumOps.FromDouble(-0.12940952255092145)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(0.48296291314469025),
                        NumOps.FromDouble(0.836516303737469),
                        NumOps.FromDouble(0.22414386804185735),
                        NumOps.FromDouble(-0.12940952255092145)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.12940952255092145),
                        NumOps.FromDouble(-0.22414386804185735),
                        NumOps.FromDouble(0.836516303737469),
                        NumOps.FromDouble(-0.48296291314469025)
                            })
                        ),
            4 => (
                        new Vector<T>(new[] {
                        NumOps.FromDouble(-0.07576571), NumOps.FromDouble(-0.02963552),
                        NumOps.FromDouble(0.49761866), NumOps.FromDouble(0.80373875),
                        NumOps.FromDouble(0.29785779), NumOps.FromDouble(-0.09921954),
                        NumOps.FromDouble(-0.01260396), NumOps.FromDouble(0.03222310)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.03222310), NumOps.FromDouble(-0.01260396),
                        NumOps.FromDouble(0.09921954), NumOps.FromDouble(0.29785779),
                        NumOps.FromDouble(-0.80373875), NumOps.FromDouble(0.49761866),
                        NumOps.FromDouble(0.02963552), NumOps.FromDouble(-0.07576571)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(0.03222310), NumOps.FromDouble(-0.01260396),
                        NumOps.FromDouble(-0.09921954), NumOps.FromDouble(0.29785779),
                        NumOps.FromDouble(0.80373875), NumOps.FromDouble(0.49761866),
                        NumOps.FromDouble(-0.02963552), NumOps.FromDouble(-0.07576571)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.07576571), NumOps.FromDouble(0.02963552),
                        NumOps.FromDouble(0.49761866), NumOps.FromDouble(-0.80373875),
                        NumOps.FromDouble(0.29785779), NumOps.FromDouble(0.09921954),
                        NumOps.FromDouble(-0.01260396), NumOps.FromDouble(-0.03222310)
                            })
                        ),
            6 => (
                        new Vector<T>(new[] {
                        NumOps.FromDouble(0.015404109327027373),
                        NumOps.FromDouble(0.0034907120842174702),
                        NumOps.FromDouble(-0.11799011114819057),
                        NumOps.FromDouble(-0.048311742585633),
                        NumOps.FromDouble(0.4910559419267466),
                        NumOps.FromDouble(0.787641141030194),
                        NumOps.FromDouble(0.3379294217276218),
                        NumOps.FromDouble(-0.07263752278646252),
                        NumOps.FromDouble(-0.021060292512300564),
                        NumOps.FromDouble(0.04472490177066578),
                        NumOps.FromDouble(0.0017677118642428036),
                        NumOps.FromDouble(-0.007800708325034148)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(0.007800708325034148),
                        NumOps.FromDouble(-0.0017677118642428036),
                        NumOps.FromDouble(-0.04472490177066578),
                        NumOps.FromDouble(0.021060292512300564),
                        NumOps.FromDouble(0.07263752278646252),
                        NumOps.FromDouble(-0.3379294217276218),
                        NumOps.FromDouble(-0.787641141030194),
                        NumOps.FromDouble(0.4910559419267466),
                        NumOps.FromDouble(0.048311742585633),
                        NumOps.FromDouble(-0.11799011114819057),
                        NumOps.FromDouble(-0.0034907120842174702),
                        NumOps.FromDouble(0.015404109327027373)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.007800708325034148),
                        NumOps.FromDouble(-0.0017677118642428036),
                        NumOps.FromDouble(0.04472490177066578),
                        NumOps.FromDouble(0.021060292512300564),
                        NumOps.FromDouble(-0.07263752278646252),
                        NumOps.FromDouble(-0.3379294217276218),
                        NumOps.FromDouble(0.787641141030194),
                        NumOps.FromDouble(0.4910559419267466),
                        NumOps.FromDouble(-0.048311742585633),
                        NumOps.FromDouble(-0.11799011114819057),
                        NumOps.FromDouble(0.0034907120842174702),
                        NumOps.FromDouble(0.015404109327027373)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(0.015404109327027373),
                        NumOps.FromDouble(-0.0034907120842174702),
                        NumOps.FromDouble(-0.11799011114819057),
                        NumOps.FromDouble(0.048311742585633),
                        NumOps.FromDouble(0.4910559419267466),
                        NumOps.FromDouble(-0.787641141030194),
                        NumOps.FromDouble(0.3379294217276218),
                        NumOps.FromDouble(0.07263752278646252),
                        NumOps.FromDouble(-0.021060292512300564),
                        NumOps.FromDouble(-0.04472490177066578),
                        NumOps.FromDouble(0.0017677118642428036),
                        NumOps.FromDouble(0.007800708325034148)
                        })
                    ),
            8 => (
                        new Vector<T>(new[] {
                        NumOps.FromDouble(-0.0033824159510061256),
                        NumOps.FromDouble(-0.0005421323317911481),
                        NumOps.FromDouble(0.03169508781149298),
                        NumOps.FromDouble(0.007607487324917605),
                        NumOps.FromDouble(-0.1432942383508097),
                        NumOps.FromDouble(-0.061273359067658524),
                        NumOps.FromDouble(0.4813596512583722),
                        NumOps.FromDouble(0.7771857517005235),
                        NumOps.FromDouble(0.3644418948353314),
                        NumOps.FromDouble(-0.05194583810770904),
                        NumOps.FromDouble(-0.027333068345077982),
                        NumOps.FromDouble(0.049137179673607506),
                        NumOps.FromDouble(0.003808752013890615),
                        NumOps.FromDouble(-0.01495225833704823),
                        NumOps.FromDouble(-0.0003029205147213668),
                        NumOps.FromDouble(0.0018899503327594609)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.0018899503327594609),
                        NumOps.FromDouble(-0.0003029205147213668),
                        NumOps.FromDouble(0.01495225833704823),
                        NumOps.FromDouble(0.003808752013890615),
                        NumOps.FromDouble(-0.049137179673607506),
                        NumOps.FromDouble(-0.027333068345077982),
                        NumOps.FromDouble(0.05194583810770904),
                        NumOps.FromDouble(0.3644418948353314),
                        NumOps.FromDouble(-0.7771857517005235),
                        NumOps.FromDouble(0.4813596512583722),
                        NumOps.FromDouble(0.061273359067658524),
                        NumOps.FromDouble(-0.1432942383508097),
                        NumOps.FromDouble(-0.007607487324917605),
                        NumOps.FromDouble(0.03169508781149298),
                        NumOps.FromDouble(0.0005421323317911481),
                        NumOps.FromDouble(-0.0033824159510061256)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(0.0018899503327594609),
                        NumOps.FromDouble(-0.0003029205147213668),
                        NumOps.FromDouble(-0.01495225833704823),
                        NumOps.FromDouble(0.003808752013890615),
                        NumOps.FromDouble(0.049137179673607506),
                        NumOps.FromDouble(-0.027333068345077982),
                        NumOps.FromDouble(-0.05194583810770904),
                        NumOps.FromDouble(0.3644418948353314),
                        NumOps.FromDouble(0.7771857517005235),
                        NumOps.FromDouble(0.4813596512583722),
                        NumOps.FromDouble(-0.061273359067658524),
                        NumOps.FromDouble(-0.1432942383508097),
                        NumOps.FromDouble(0.007607487324917605),
                        NumOps.FromDouble(0.03169508781149298),
                        NumOps.FromDouble(-0.0005421323317911481),
                        NumOps.FromDouble(-0.0033824159510061256)
                                }),
                                new Vector<T>(new[] {
                        NumOps.FromDouble(-0.0033824159510061256),
                        NumOps.FromDouble(0.0005421323317911481),
                        NumOps.FromDouble(0.03169508781149298),
                        NumOps.FromDouble(-0.007607487324917605),
                        NumOps.FromDouble(-0.1432942383508097),
                        NumOps.FromDouble(0.061273359067658524),
                        NumOps.FromDouble(0.4813596512583722),
                        NumOps.FromDouble(-0.7771857517005235),
                        NumOps.FromDouble(0.3644418948353314),
                        NumOps.FromDouble(0.05194583810770904),
                        NumOps.FromDouble(-0.027333068345077982),
                        NumOps.FromDouble(-0.049137179673607506),
                        NumOps.FromDouble(0.003808752013890615),
                        NumOps.FromDouble(0.01495225833704823),
                        NumOps.FromDouble(-0.0003029205147213668),
                        NumOps.FromDouble(-0.0018899503327594609)
                    })
                ),
            _ => throw new ArgumentException($"Symlet wavelet of order {order} is not implemented or not supported. Please use a supported order (e.g., 2, 4, 6, or 8).", nameof(order)),
        };
    }
}
