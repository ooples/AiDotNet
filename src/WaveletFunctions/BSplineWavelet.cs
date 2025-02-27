namespace AiDotNet.WaveletFunctions;

public class BSplineWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;

    public BSplineWavelet(int order = 3)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
    }

    public T Calculate(T x)
    {
        return BSpline(x, _order);
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        var lowPass = GetDecompositionLowPassFilter();
        var highPass = GetDecompositionHighPassFilter();

        var approximation = Convolve(input, lowPass);
        var detail = Convolve(input, highPass);

        // Downsample by 2
        approximation = BSplineWavelet<T>.Downsample(approximation, 2);
        detail = BSplineWavelet<T>.Downsample(detail, 2);

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        return GetDecompositionLowPassFilter();
    }

    public Vector<T> GetWaveletCoefficients()
    {
        return GetDecompositionHighPassFilter();
    }

    private Vector<T> GetDecompositionLowPassFilter()
    {
        var coeffs = BSplineWavelet<T>.GetBSplineCoefficients(_order);
        return NormalizeAndConvert(coeffs);
    }

    private Vector<T> GetDecompositionHighPassFilter()
    {
        var coeffs = BSplineWavelet<T>.GetBSplineCoefficients(_order + 1);
        for (int i = 1; i < coeffs.Length; i += 2)
        {
            coeffs[i] = -coeffs[i];
        }

        return NormalizeAndConvert(coeffs);
    }

    private static double[] GetBSplineCoefficients(int order)
    {
        if (order == 1)
        {
            return [1, 1];
        }

        var prev = BSplineWavelet<T>.GetBSplineCoefficients(order - 1);
        var result = new double[prev.Length + 1];

        for (int i = 0; i < result.Length; i++)
        {
            double t = (double)i / (order - 1);
            result[i] = (1 - t) * (i > 0 ? prev[i - 1] : 0) + t * (i < prev.Length ? prev[i] : 0);
        }

        return result;
    }

    private Vector<T> NormalizeAndConvert(double[] coeffs)
    {
        double normFactor = Math.Sqrt(coeffs.Sum(c => c * c));
        return new Vector<T>(coeffs.Select(c => _numOps.FromDouble(c / normFactor)).ToArray());
    }

    private Vector<T> Convolve(Vector<T> input, Vector<T> filter)
    {
        var result = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = _numOps.Zero;
            for (int j = 0; j < filter.Length; j++)
            {
                int k = i - j + filter.Length / 2;
                if (k >= 0 && k < input.Length)
                {
                    result[i] = _numOps.Add(result[i], _numOps.Multiply(input[k], filter[j]));
                }
            }
        }

        return new Vector<T>(result);
    }

    private static Vector<T> Downsample(Vector<T> input, int factor)
    {
        return new Vector<T>([.. input.Where((_, i) => i % factor == 0)]);
    }

    private T BSpline(T x, int n)
    {
        if (n == 0)
        {
            return _numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.One) ? _numOps.One : _numOps.Zero;
        }

        T term1 = _numOps.Multiply(_numOps.Divide(x, _numOps.FromDouble(n)), BSpline(_numOps.Subtract(x, _numOps.One), n - 1));
        T term2 = _numOps.Multiply(_numOps.Divide(_numOps.Subtract(_numOps.FromDouble(n + 1), x), _numOps.FromDouble(n)), BSpline(x, n - 1));

        return _numOps.Add(term1, term2);
    }
}