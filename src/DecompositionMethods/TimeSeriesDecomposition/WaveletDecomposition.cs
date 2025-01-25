namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class WaveletDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _levels;
    private readonly IWaveletFunction<T> _wavelet;
    private readonly WaveletDecompositionAlgorithm _algorithm;
    private readonly int _minimumDecompositionLength;

    public WaveletDecomposition(Vector<T> timeSeries, IWaveletFunction<T>? wavelet = null, int levels = 3, 
        WaveletDecompositionAlgorithm algorithm = WaveletDecompositionAlgorithm.DWT, int minimumDecompositionLength = 2)
        : base(timeSeries)
    {
        _levels = levels;
        _wavelet = wavelet ?? new HaarWavelet<T>();
        _algorithm = algorithm;
        _minimumDecompositionLength = minimumDecompositionLength;
        Decompose();
    }

    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case WaveletDecompositionAlgorithm.DWT:
                DecomposeDWT();
                break;
            case WaveletDecompositionAlgorithm.MODWT:
                DecomposeMODWT();
                break;
            case WaveletDecompositionAlgorithm.SWT:
                DecomposeSWT();
                break;
            default:
                throw new NotImplementedException($"Wavelet decomposition algorithm {_algorithm} is not implemented.");
        }
    }

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