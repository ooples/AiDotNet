namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements wavelet-based decomposition methods for time series data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Wavelet decomposition is like breaking down a complex signal (like music)
/// into different frequency bands. Think of it as separating bass, mid-range, and treble in music.
/// This helps identify patterns at different time scales - from long-term trends to short-term fluctuations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public class WaveletDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    /// <summary>
    /// The number of decomposition levels to perform.
    /// </summary>
    private readonly int _levels;

    /// <summary>
    /// The wavelet function used for decomposition.
    /// </summary>
    private readonly IWaveletFunction<T> _wavelet;

    /// <summary>
    /// The type of wavelet algorithm to use.
    /// </summary>
    private readonly WaveletAlgorithmType _algorithm;

    /// <summary>
    /// The minimum length of data required to continue decomposition.
    /// </summary>
    private readonly int _minimumDecompositionLength;

    /// <summary>
    /// Initializes a new instance of the WaveletDecomposition class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the wavelet decomposition with your time series data.
    /// Think of it as preparing your toolbox before analyzing the data. You can choose:
    /// - How many levels to break down the data (like zooming in multiple times)
    /// - Which wavelet function to use (different shapes for different analysis needs)
    /// - Which algorithm to use (different methods with different strengths)
    /// </para>
    /// </remarks>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="wavelet">The wavelet function to use (defaults to Haar wavelet if not specified).</param>
    /// <param name="levels">The number of decomposition levels (default is 3).</param>
    /// <param name="algorithm">The wavelet algorithm type to use (default is DWT).</param>
    /// <param name="minimumDecompositionLength">The minimum length required to continue decomposition (default is 2).</param>
    public WaveletDecomposition(Vector<T> timeSeries, IWaveletFunction<T>? wavelet = null, int levels = 3,
        WaveletAlgorithmType algorithm = WaveletAlgorithmType.DWT, int minimumDecompositionLength = 2)
        : base(timeSeries)
    {
        _levels = levels;
        _wavelet = wavelet ?? new HaarWavelet<T>();
        _algorithm = algorithm;
        _minimumDecompositionLength = minimumDecompositionLength;
        Decompose();
    }

    /// <summary>
    /// Performs the wavelet decomposition using the selected algorithm.
    /// </summary>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case WaveletAlgorithmType.DWT:
                DecomposeDWT();
                break;
            case WaveletAlgorithmType.MODWT:
                DecomposeMODWT();
                break;
            case WaveletAlgorithmType.SWT:
                DecomposeSWT();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(_algorithm), _algorithm, "Unsupported wavelet decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs Discrete Wavelet Transform (DWT) decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DWT is like repeatedly splitting your data into two parts:
    /// - A smoothed version (approximation) that captures the overall shape
    /// - The details that were removed during smoothing
    /// 
    /// This process is repeated on the smoothed version multiple times, creating a
    /// multi-level breakdown of your data.
    /// </para>
    /// </remarks>
    private void DecomposeDWT()
    {
        var approximation = new Vector<T>(TimeSeries);
        var details = new List<Vector<T>>();

        for (int level = 0; level < _levels; level++)
        {
            var (newApproximation, detail) = _wavelet.Decompose(approximation);
            details.Add(new Vector<T>(detail));
            approximation = new Vector<T>(newApproximation);

            if (approximation.Length <= _minimumDecompositionLength)
            {
                break; // Stop if the signal becomes too short
            }
        }

        details.Reverse(); // Reverse to have details from coarse to fine

        Vector<T> trend = new Vector<T>(approximation);
        Vector<T> seasonal = new Vector<T>(TimeSeries.Length);
        Vector<T> residual = new Vector<T>(TimeSeries.Length);

        for (int i = 0; i < details.Count; i++)
        {
            var paddedDetail = PadToLength(details[i], TimeSeries.Length);
            if (i < details.Count / 2)
            {
                seasonal = seasonal.Add(paddedDetail);
            }
            else
            {
                residual = residual.Add(paddedDetail);
            }
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Extends a vector to the specified length by padding with zeros.
    /// </summary>
    /// <param name="vector">The vector to pad.</param>
    /// <param name="length">The desired length.</param>
    /// <returns>A new vector of the specified length.</returns>
    private Vector<T> PadToLength(Vector<T> vector, int length)
    {
        if (vector.Length == length)
        {
            return vector;
        }

        var padded = new Vector<T>(length);
        for (int i = 0; i < Math.Min(vector.Length, length); i++)
        {
            padded[i] = vector[i];
        }

        return padded;
    }

    /// <summary>
    /// Performs Maximal Overlap Discrete Wavelet Transform (MODWT) decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MODWT is an enhanced version of the basic wavelet transform.
    /// Unlike the standard DWT, it:
    /// - Preserves the original data length at each level
    /// - Handles any data length (not just powers of 2)
    /// - Is less sensitive to the starting point of your data
    /// 
    /// This makes it more suitable for many time series analysis tasks.
    /// </para>
    /// </remarks>
    private void DecomposeMODWT()
    {
        var N = TimeSeries.Length;
        var J = Math.Min(_levels, (int)MathHelper.Log2(N));
        var W = new List<Vector<T>>();
        var V = new List<Vector<T>>();

        var w = new Vector<T>(TimeSeries);

        for (int j = 1; j <= J; j++)
        {
            var (wj, vj) = MODWTStep(w, j);
            W.Add(wj);
            V.Add(vj);
            w = vj;
        }

        // Combine components
        var trend = V[J - 1];
        var seasonal = new Vector<T>(N);
        var residual = new Vector<T>(N);

        for (int j = 0; j < J; j++)
        {
            if (j < J / 2)
            {
                seasonal = seasonal.Add(W[j]);
            }
            else
            {
                residual = residual.Add(W[j]);
            }
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Performs a single step of the MODWT decomposition.
    /// </summary>
    /// <param name="v">The input vector to decompose.</param>
    /// <param name="j">The current decomposition level.</param>
    /// <returns>A tuple containing the detail coefficients and approximation coefficients.</returns>
    private (Vector<T>, Vector<T>) MODWTStep(Vector<T> v, int j)
    {
        var N = v.Length;
        var wj = new Vector<T>(N);
        var vj = new Vector<T>(N);

        var h = _wavelet.GetScalingCoefficients();
        var g = _wavelet.GetWaveletCoefficients();

        for (int t = 0; t < N; t++)
        {
            for (int l = 0; l < h.Length; l++)
            {
                int index = (t - l * (int)Math.Pow(2, j - 1) + N) % N;
                wj[t] = NumOps.Add(wj[t], NumOps.Multiply(g[l], v[index]));
                vj[t] = NumOps.Add(vj[t], NumOps.Multiply(h[l], v[index]));
            }
        }

        return (wj, vj);
    }

    /// <summary>
    /// Performs Stationary Wavelet Transform (SWT) decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SWT is a variation of wavelet transform that's particularly useful
    /// for pattern recognition and noise reduction. Unlike standard DWT:
    /// - It keeps the same data length at each level
    /// - It's shift-invariant (the results don't change if you shift your data)
    /// - It achieves this by modifying the wavelet filters at each level instead of downsampling the data
    /// 
    /// This makes it better for detecting patterns regardless of where they appear in your data.
    /// </para>
    /// </remarks>
    private void DecomposeSWT()
    {
        var N = TimeSeries.Length;
        var J = Math.Min(_levels, (int)MathHelper.Log2(N));
        var W = new List<Vector<T>>();
        var V = new List<Vector<T>>();

        var v = new Vector<T>(TimeSeries);

        for (int j = 1; j <= J; j++)
        {
            var (wj, vj) = SWTStep(v, j);
            W.Add(wj);
            V.Add(vj);
            v = vj;
        }

        // Combine components
        var trend = V[J - 1];
        var seasonal = new Vector<T>(N);
        var residual = new Vector<T>(N);

        for (int j = 0; j < J; j++)
        {
            if (j < J / 2)
            {
                seasonal = seasonal.Add(W[j]);
            }
            else
            {
                residual = residual.Add(W[j]);
            }
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    /// <summary>
    /// Performs a single step of the Stationary Wavelet Transform (SWT) decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method performs one level of the SWT decomposition. It takes your data
    /// and creates two new sets of values:
    /// - Detail coefficients (wj): These capture the high-frequency changes or "details" in your data
    /// - Approximation coefficients (vj): These capture the low-frequency components or overall "shape"
    /// 
    /// The method uses special filter coefficients (h and g) that are like mathematical patterns
    /// to extract different features from your data.
    /// </para>
    /// </remarks>
    /// <param name="v">The input vector to decompose (your time series data).</param>
    /// <param name="j">The current decomposition level (determines how "zoomed out" the analysis is).</param>
    /// <returns>A tuple containing the detail coefficients and approximation coefficients.</returns>
    private (Vector<T>, Vector<T>) SWTStep(Vector<T> v, int j)
    {
        var N = v.Length;
        var wj = new Vector<T>(N);
        var vj = new Vector<T>(N);

        var h = _wavelet.GetScalingCoefficients();
        var g = _wavelet.GetWaveletCoefficients();

        var h_upsampled = UpsampleFilter(h, j);
        var g_upsampled = UpsampleFilter(g, j);

        for (int t = 0; t < N; t++)
        {
            for (int l = 0; l < h_upsampled.Length; l++)
            {
                int index = (t - l + N) % N;
                wj[t] = NumOps.Add(wj[t], NumOps.Multiply(g_upsampled[l], v[index]));
                vj[t] = NumOps.Add(vj[t], NumOps.Multiply(h_upsampled[l], v[index]));
            }
        }

        return (wj, vj);
    }

    /// <summary>
    /// Expands a filter by inserting zeros between its elements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method "stretches" a filter by inserting zeros between its values.
    /// For example, if your filter is [1, 2, 3] and j=2, the result would be [1, 0, 2, 0, 3].
    /// 
    /// This is a key part of the SWT algorithm that allows it to analyze data at different scales
    /// without changing the data length. Think of it like zooming in on a map - you see more detail
    /// but the map itself doesn't change size.
    /// </para>
    /// </remarks>
    /// <param name="filter">The original filter coefficients.</param>
    /// <param name="j">The decomposition level that determines how many zeros to insert.</param>
    /// <returns>A new vector containing the upsampled filter.</returns>
    private Vector<T> UpsampleFilter(Vector<T> filter, int j)
    {
        var upsampled = new Vector<T>((int)Math.Pow(2, j - 1) * filter.Length);
        for (int i = 0; i < filter.Length; i++)
        {
            upsampled[i * (int)Math.Pow(2, j - 1)] = filter[i];
        }

        return upsampled;
    }
}
