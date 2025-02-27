namespace AiDotNet.WaveletFunctions;

public class DaubechiesWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;
    private readonly Vector<T> _scalingCoefficients;
    private readonly Vector<T> _waveletCoefficients;

    public DaubechiesWavelet(int order = 4)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        _scalingCoefficients = ComputeScalingCoefficients();
        _waveletCoefficients = ComputeWaveletCoefficients();
    }

    public T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        if (t < 0 || t > _order - 1)
            return _numOps.Zero;

        T result = _numOps.Zero;
        for (int k = 0; k < _scalingCoefficients.Length; k++)
        {
            double shiftedT = t - k;
            if (shiftedT >= 0 && shiftedT < 1)
            {
                result = _numOps.Add(result, _numOps.Multiply(_scalingCoefficients[k], _numOps.FromDouble(CascadeAlgorithm(shiftedT))));
            }
        }

        return result;
    }

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

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Daubechies wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        for (int i = 0; i < halfLength; i++)
        {
            T approx = _numOps.Zero;
            T det = _numOps.Zero;

            for (int j = 0; j < _scalingCoefficients.Length; j++)
            {
                int index = (2 * i + j) % input.Length;
                approx = _numOps.Add(approx, _numOps.Multiply(_scalingCoefficients[j], input[index]));
                det = _numOps.Add(det, _numOps.Multiply(_waveletCoefficients[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        return _scalingCoefficients;
    }

    public Vector<T> GetWaveletCoefficients()
    {
        return _waveletCoefficients;
    }

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

        return new Vector<T>([.. d4Coefficients.Select(c => _numOps.FromDouble(c))]);
    }

    private Vector<T> ComputeWaveletCoefficients()
    {
        var coeffs = _scalingCoefficients.ToArray();
        int L = coeffs.Length;
        var waveletCoeffs = new T[L];

        for (int i = 0; i < L; i++)
        {
            waveletCoeffs[i] = _numOps.Multiply(_numOps.FromDouble(Math.Pow(-1, i)), coeffs[L - 1 - i]);
        }

        return new Vector<T>(waveletCoeffs);
    }
}