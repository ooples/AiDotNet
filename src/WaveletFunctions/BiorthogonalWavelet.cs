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
public partial class BiorthogonalWavelet<T> : WaveletFunctionBase<T>
{
    /// <summary>
    /// Coefficients used for the decomposition process.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _decompositionCoefficients;

    /// <summary>
    /// Coefficients used for the reconstruction process.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _reconstructionCoefficients;

    /// <summary>
    /// The four PyWavelets filter-tap tables (<c>dec_lo</c>, <c>dec_hi</c>, <c>rec_lo</c>, <c>rec_hi</c>) of the
    /// selected <c>bior{reconstructionOrder}.{decompositionOrder}</c> wavelet.
    /// </summary>
    private readonly (double[] DecLo, double[] DecHi, double[] RecLo, double[] RecHi) _taps;

    /// <summary>
    /// Initializes a new instance of the BiorthogonalWavelet class with the specified decomposition and reconstruction orders.
    /// </summary>
    /// <param name="decompositionOrder">The order of the wavelet used for decomposition. Default is 2.</param>
    /// <param name="reconstructionOrder">The order of the wavelet used for reconstruction. Default is 2.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the pair does not name a biorthogonal wavelet: the supported
    /// <c>bior{reconstructionOrder}.{decompositionOrder}</c> pairs are 1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3,
    /// 3.5, 3.7, 3.9, 4.4, 5.5 and 6.8, the same set PyWavelets provides.
    /// </exception>
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
    /// The pair maps to PyWavelets' <c>bior{reconstructionOrder}.{decompositionOrder}</c>, so for example
    /// decompositionOrder 4 with reconstructionOrder 2 is <c>bior2.4</c>. The filters are exactly PyWavelets'
    /// <c>dec_lo</c>, <c>dec_hi</c>, <c>rec_lo</c> and <c>rec_hi</c>. A biorthogonal pair only exists for
    /// certain combinations, so any other pair is rejected.
    /// </para>
    /// </remarks>
    public BiorthogonalWavelet(int decompositionOrder = 2, int reconstructionOrder = 2)
    {
        if (reconstructionOrder < 1 || reconstructionOrder > 6)
            throw new ArgumentException("Reconstruction order must be between 1 and 6.", nameof(reconstructionOrder));

        var taps = GetFilterTaps(reconstructionOrder, decompositionOrder);
        if (taps is null)
        {
            throw new ArgumentException(
                $"bior{reconstructionOrder}.{decompositionOrder} is not a biorthogonal wavelet. Supported pairs " +
                "(reconstruction.decomposition): 1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, " +
                "5.5, 6.8.",
                nameof(decompositionOrder));
        }

        _taps = taps.Value;
        _decompositionCoefficients = ToFilter(_taps.DecLo);
        _reconstructionCoefficients = ToFilter(_taps.RecLo);
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
        if (_decompositionCoefficients.Length == 0)
        {
            return NumOps.Zero;
        }

        var engine = AiDotNetEngine.Current;

        // sum_k c[k] * phi(x - k) with the box scaling function phi(t) = 1 for 0 <= t < 1, else 0.
        var shifts = engine.TensorArange<T>(0, _decompositionCoefficients.Length, 1);
        var shifted = engine.ScalarMinusTensor(x, shifts);
        var notNegative = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(shifted, NumOps.Zero));
        var box = engine.TensorMultiply(notNegative, engine.TensorLessThan(shifted, NumOps.One));
        return engine.TensorSum(engine.TensorMultiply(Tensor<T>.FromVector(_decompositionCoefficients), box));
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
    /// Gets PyWavelets' filter taps for <c>bior{reconstructionOrder}.{decompositionOrder}</c>.
    /// </summary>
    /// <param name="reconstructionOrder">The reconstruction (synthesis) order, the first number of the name.</param>
    /// <param name="decompositionOrder">The decomposition (analysis) order, the second number of the name.</param>
    /// <returns>
    /// The <c>dec_lo</c>, <c>dec_hi</c>, <c>rec_lo</c> and <c>rec_hi</c> taps of
    /// <c>pywt.Wavelet("biorNr.Nd")</c>, or <c>null</c> when the pair is not a biorthogonal wavelet.
    /// </returns>
    private static (double[] DecLo, double[] DecHi, double[] RecLo, double[] RecHi)? GetFilterTaps(
        int reconstructionOrder, int decompositionOrder)
    {
        return (reconstructionOrder, decompositionOrder) switch
        {
            (1, 1) => (
                new[]
                {
                    0.7071067811865476, 0.7071067811865476
                },
                new[]
                {
                    -0.7071067811865476, 0.7071067811865476
                },
                new[]
                {
                    0.7071067811865476, 0.7071067811865476
                },
                new[]
                {
                    0.7071067811865476, -0.7071067811865476
                }),
            (1, 3) => (
                new[]
                {
                    -0.08838834764831845, 0.08838834764831845, 0.7071067811865476, 0.7071067811865476,
                    0.08838834764831845, -0.08838834764831845
                },
                new[]
                {
                    -0.0, 0.0, -0.7071067811865476, 0.7071067811865476, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0
                },
                new[]
                {
                    -0.08838834764831845, -0.08838834764831845, 0.7071067811865476, -0.7071067811865476,
                    0.08838834764831845, 0.08838834764831845
                }),
            (1, 5) => (
                new[]
                {
                    0.016572815184059706, -0.016572815184059706, -0.12153397801643785, 0.12153397801643785,
                    0.7071067811865476, 0.7071067811865476, 0.12153397801643785, -0.12153397801643785,
                    -0.016572815184059706, 0.016572815184059706
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.0, -0.7071067811865476, 0.7071067811865476, -0.0, 0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476, 0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    0.016572815184059706, 0.016572815184059706, -0.12153397801643785, -0.12153397801643785,
                    0.7071067811865476, -0.7071067811865476, 0.12153397801643785, 0.12153397801643785,
                    -0.016572815184059706, -0.016572815184059706
                }),
            (2, 2) => (
                new[]
                {
                    0.0, -0.1767766952966369, 0.3535533905932738, 1.0606601717798212, 0.3535533905932738,
                    -0.1767766952966369
                },
                new[]
                {
                    -0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738, 0.0, 0.0
                },
                new[]
                {
                    0.0, 0.1767766952966369, 0.3535533905932738, -1.0606601717798212, 0.3535533905932738,
                    0.1767766952966369
                }),
            (2, 4) => (
                new[]
                {
                    0.0, 0.03314563036811941, -0.06629126073623882, -0.1767766952966369, 0.4198446513295126,
                    0.9943689110435825, 0.4198446513295126, -0.1767766952966369, -0.06629126073623882,
                    0.03314563036811941
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, -0.0, 0.0, -0.0,
                    0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738, 0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    0.0, -0.03314563036811941, -0.06629126073623882, 0.1767766952966369, 0.4198446513295126,
                    -0.9943689110435825, 0.4198446513295126, 0.1767766952966369, -0.06629126073623882,
                    -0.03314563036811941
                }),
            (2, 6) => (
                new[]
                {
                    0.0, -0.006905339660024878, 0.013810679320049757, 0.04695630968816917, -0.1077232986963881,
                    -0.16987135563661201, 0.4474660099696121, 0.966747552403483, 0.4474660099696121,
                    -0.16987135563661201, -0.1077232986963881, 0.04695630968816917, 0.013810679320049757,
                    -0.006905339660024878
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.0, -0.0, 0.3535533905932738, -0.7071067811865476, 0.3535533905932738, -0.0,
                    0.0, -0.0, 0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    0.0, 0.006905339660024878, 0.013810679320049757, -0.04695630968816917, -0.1077232986963881,
                    0.16987135563661201, 0.4474660099696121, -0.966747552403483, 0.4474660099696121,
                    0.16987135563661201, -0.1077232986963881, -0.04695630968816917, 0.013810679320049757,
                    0.006905339660024878
                }),
            (2, 8) => (
                new[]
                {
                    0.0, 0.0015105430506304422, -0.0030210861012608843, -0.012947511862546647, 0.02891610982635418,
                    0.05299848189069094, -0.13491307360773605, -0.16382918343409023, 0.46257144047591653,
                    0.9516421218971786, 0.46257144047591653, -0.16382918343409023, -0.13491307360773605,
                    0.05299848189069094, 0.02891610982635418, -0.012947511862546647, -0.0030210861012608843,
                    0.0015105430506304422
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.3535533905932738, -0.7071067811865476,
                    0.3535533905932738, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3535533905932738, 0.7071067811865476, 0.3535533905932738,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    0.0, -0.0015105430506304422, -0.0030210861012608843, 0.012947511862546647, 0.02891610982635418,
                    -0.05299848189069094, -0.13491307360773605, 0.16382918343409023, 0.46257144047591653,
                    -0.9516421218971786, 0.46257144047591653, 0.16382918343409023, -0.13491307360773605,
                    -0.05299848189069094, 0.02891610982635418, 0.012947511862546647, -0.0030210861012608843,
                    -0.0015105430506304422
                }),
            (3, 1) => (
                new[]
                {
                    -0.3535533905932738, 1.0606601717798212, 1.0606601717798212, -0.3535533905932738
                },
                new[]
                {
                    -0.1767766952966369, 0.5303300858899106, -0.5303300858899106, 0.1767766952966369
                },
                new[]
                {
                    0.1767766952966369, 0.5303300858899106, 0.5303300858899106, 0.1767766952966369
                },
                new[]
                {
                    -0.3535533905932738, -1.0606601717798212, 1.0606601717798212, 0.3535533905932738
                }),
            (3, 3) => (
                new[]
                {
                    0.06629126073623882, -0.1988737822087165, -0.15467960838455727, 0.9943689110435825,
                    0.9943689110435825, -0.15467960838455727, -0.1988737822087165, 0.06629126073623882
                },
                new[]
                {
                    -0.0, 0.0, -0.1767766952966369, 0.5303300858899106, -0.5303300858899106, 0.1767766952966369,
                    -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.1767766952966369, 0.5303300858899106, 0.5303300858899106, 0.1767766952966369, 0.0,
                    0.0
                },
                new[]
                {
                    0.06629126073623882, 0.1988737822087165, -0.15467960838455727, -0.9943689110435825,
                    0.9943689110435825, 0.15467960838455727, -0.1988737822087165, -0.06629126073623882
                }),
            (3, 5) => (
                new[]
                {
                    -0.013810679320049757, 0.04143203796014927, 0.052480581416189075, -0.26792717880896527,
                    -0.07181553246425873, 0.966747552403483, 0.966747552403483, -0.07181553246425873,
                    -0.26792717880896527, 0.052480581416189075, 0.04143203796014927, -0.013810679320049757
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.0, -0.1767766952966369, 0.5303300858899106, -0.5303300858899106,
                    0.1767766952966369, -0.0, 0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.0, 0.1767766952966369, 0.5303300858899106, 0.5303300858899106,
                    0.1767766952966369, 0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    -0.013810679320049757, -0.04143203796014927, 0.052480581416189075, 0.26792717880896527,
                    -0.07181553246425873, -0.966747552403483, 0.966747552403483, 0.07181553246425873,
                    -0.26792717880896527, -0.052480581416189075, 0.04143203796014927, 0.013810679320049757
                }),
            (3, 7) => (
                new[]
                {
                    0.0030210861012608843, -0.009063258303782653, -0.01683176542131064, 0.074663985074019,
                    0.03133297870736289, -0.301159125922835, -0.02649924094534547, 0.9516421218971786,
                    0.9516421218971786, -0.02649924094534547, -0.301159125922835, 0.03133297870736289,
                    0.074663985074019, -0.01683176542131064, -0.009063258303782653, 0.0030210861012608843
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.1767766952966369, 0.5303300858899106, -0.5303300858899106,
                    0.1767766952966369, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1767766952966369, 0.5303300858899106, 0.5303300858899106,
                    0.1767766952966369, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    0.0030210861012608843, 0.009063258303782653, -0.01683176542131064, -0.074663985074019,
                    0.03133297870736289, 0.301159125922835, -0.02649924094534547, -0.9516421218971786,
                    0.9516421218971786, 0.02649924094534547, -0.301159125922835, -0.03133297870736289,
                    0.074663985074019, 0.01683176542131064, -0.009063258303782653, -0.0030210861012608843
                }),
            (3, 9) => (
                new[]
                {
                    -0.0006797443727836989, 0.002039233118351097, 0.005060319219611981, -0.020618912641105536,
                    -0.014112787930175844, 0.09913478249423216, 0.012300136269419315, -0.32019196836077857,
                    0.0020500227115698858, 0.9421257006782068, 0.9421257006782068, 0.0020500227115698858,
                    -0.32019196836077857, 0.012300136269419315, 0.09913478249423216, -0.014112787930175844,
                    -0.020618912641105536, 0.005060319219611981, 0.002039233118351097, -0.0006797443727836989
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.1767766952966369, 0.5303300858899106,
                    -0.5303300858899106, 0.1767766952966369, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1767766952966369, 0.5303300858899106,
                    0.5303300858899106, 0.1767766952966369, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                },
                new[]
                {
                    -0.0006797443727836989, -0.002039233118351097, 0.005060319219611981, 0.020618912641105536,
                    -0.014112787930175844, -0.09913478249423216, 0.012300136269419315, 0.32019196836077857,
                    0.0020500227115698858, -0.9421257006782068, 0.9421257006782068, -0.0020500227115698858,
                    -0.32019196836077857, -0.012300136269419315, 0.09913478249423216, 0.014112787930175844,
                    -0.020618912641105536, -0.005060319219611981, 0.002039233118351097, 0.0006797443727836989
                }),
            (4, 4) => (
                new[]
                {
                    0.0, 0.03782845550726404, -0.023849465019556843, -0.11062440441843718, 0.37740285561283066,
                    0.8526986790088938, 0.37740285561283066, -0.11062440441843718, -0.023849465019556843,
                    0.03782845550726404
                },
                new[]
                {
                    -0.0, -0.06453888262869706, 0.04068941760916406, 0.41809227322161724, -0.7884856164055829,
                    0.41809227322161724, 0.04068941760916406, -0.06453888262869706, -0.0, 0.0
                },
                new[]
                {
                    0.0, -0.06453888262869706, -0.04068941760916406, 0.41809227322161724, 0.7884856164055829,
                    0.41809227322161724, -0.04068941760916406, -0.06453888262869706, 0.0, 0.0
                },
                new[]
                {
                    0.0, -0.03782845550726404, -0.023849465019556843, 0.11062440441843718, 0.37740285561283066,
                    -0.8526986790088938, 0.37740285561283066, 0.11062440441843718, -0.023849465019556843,
                    -0.03782845550726404
                }),
            (5, 5) => (
                new[]
                {
                    0.0, 0.0, 0.03968708834740544, 0.007948108637240322, -0.05446378846823691, 0.34560528195603346,
                    0.7366601814282105, 0.34560528195603346, -0.05446378846823691, 0.007948108637240322,
                    0.03968708834740544, 0.0
                },
                new[]
                {
                    -0.013456709459118716, -0.002694966880111507, 0.13670658466432914, -0.09350469740093886,
                    -0.47680326579848425, 0.8995061097486484, -0.47680326579848425, -0.09350469740093886,
                    0.13670658466432914, -0.002694966880111507, -0.013456709459118716, 0.0
                },
                new[]
                {
                    0.013456709459118716, -0.002694966880111507, -0.13670658466432914, -0.09350469740093886,
                    0.47680326579848425, 0.8995061097486484, 0.47680326579848425, -0.09350469740093886,
                    -0.13670658466432914, -0.002694966880111507, 0.013456709459118716, 0.0
                },
                new[]
                {
                    0.0, -0.0, 0.03968708834740544, -0.007948108637240322, -0.05446378846823691,
                    -0.34560528195603346, 0.7366601814282105, -0.34560528195603346, -0.05446378846823691,
                    -0.007948108637240322, 0.03968708834740544, -0.0
                }),
            (6, 8) => (
                new[]
                {
                    0.0, 0.0019088317364812906, -0.0019142861290887667, -0.016990639867602342, 0.01193456527972926,
                    0.04973290349094079, -0.07726317316720414, -0.09405920349573646, 0.4207962846098268,
                    0.8259229974584023, 0.4207962846098268, -0.09405920349573646, -0.07726317316720414,
                    0.04973290349094079, 0.01193456527972926, -0.016990639867602342, -0.0019142861290887667,
                    0.0019088317364812906
                },
                new[]
                {
                    -0.0, 0.0, -0.0, 0.014426282505624435, -0.014467504896790148, -0.07872200106262882,
                    0.04036797903033992, 0.41784910915027457, -0.7589077294536541, 0.41784910915027457,
                    0.04036797903033992, -0.07872200106262882, -0.014467504896790148, 0.014426282505624435, -0.0,
                    0.0, -0.0, 0.0
                },
                new[]
                {
                    0.0, 0.0, 0.0, 0.014426282505624435, 0.014467504896790148, -0.07872200106262882,
                    -0.04036797903033992, 0.41784910915027457, 0.7589077294536541, 0.41784910915027457,
                    -0.04036797903033992, -0.07872200106262882, 0.014467504896790148, 0.014426282505624435, 0.0,
                    0.0, 0.0, 0.0
                },
                new[]
                {
                    0.0, -0.0019088317364812906, -0.0019142861290887667, 0.016990639867602342, 0.01193456527972926,
                    -0.04973290349094079, -0.07726317316720414, 0.09405920349573646, 0.4207962846098268,
                    -0.8259229974584023, 0.4207962846098268, 0.09405920349573646, -0.07726317316720414,
                    -0.04973290349094079, 0.01193456527972926, 0.016990639867602342, -0.0019142861290887667,
                    -0.0019088317364812906
                }),
            _ => null,
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
        // PyWavelets' dec_lo / dec_hi are convolution taps; correlating with the time-reversed taps gives
        // a[i] = sum_j h[j] * x[(2i + j) mod n] and d[i] = sum_j g[j] * x[(2i + j) mod n], which
        // Reconstruct inverts exactly with rec_lo / rec_hi.
        var decompositionLowPass = WaveletEngineOps<T>.Reverse(GetDecompositionLowPassFilter());
        var decompositionHighPass = WaveletEngineOps<T>.Reverse(GetDecompositionHighPassFilter());
        var approximation = WaveletEngineOps<T>.PeriodicCorrelate(input, decompositionLowPass, 2, halfLength);
        var detail = WaveletEngineOps<T>.PeriodicCorrelate(input, decompositionHighPass, 2, halfLength);
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
    /// 2. Circularly convolve with the reconstruction low-pass and high-pass filters
    /// 3. Add the results together
    ///
    /// This is the inverse of the Decompose method, so:
    /// Reconstruct(Decompose(signal)) should equal the original signal.
    /// </para>
    /// </remarks>
    public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
    {
        if (approximation.Length != detail.Length)
        {
            throw new ArgumentException("Approximation and detail coefficients must have the same length.", nameof(detail));
        }

        // Periodic synthesis x[m] = sum_i a[i] rec_lo[(m - 2i) mod n] + d[i] rec_hi[(m - 2i) mod n], the exact
        // inverse of the periodic analysis in Decompose.
        return WaveletEngineOps<T>.PeriodicSynthesis(approximation, detail, GetReconstructionLowPassFilter(),
            GetReconstructionHighPassFilter());
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
    /// The coefficients are PyWavelets' dec_lo taps for the selected biorNr.Nd wavelet. They are
    /// symmetric (reducing edge artifacts) and work perfectly with the reconstruction filters.
    /// </para>
    /// </remarks>
    private Vector<T> GetDecompositionLowPassFilter()
    {
        return ToFilter(_taps.DecLo);
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
        return ToFilter(_taps.DecHi);
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
        return ToFilter(_taps.RecLo);
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
        return ToFilter(_taps.RecHi);
    }

    /// <summary>
    /// Converts a table of PyWavelets filter taps to a vector of type T.
    /// </summary>
    /// <param name="taps">The filter taps, in PyWavelets order.</param>
    /// <returns>The taps as a vector of type T, unscaled.</returns>
    private Vector<T> ToFilter(double[] taps)
    {
        // The taps are used as published: biorthogonal filters are not unit-norm, and rescaling them would
        // break perfect reconstruction.
        return new Vector<T>([.. taps.Select(c => NumOps.FromDouble(c))]);
    }
}
