namespace AiDotNet.Wavelets;

public class DaubechiesWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T[] _coefficients;
    private readonly int _order;

    public DaubechiesWavelet(int order = 4)
    {
        if (order < 4 || order % 2 != 0)
            throw new ArgumentException("Order must be an even number greater than or equal to 4.", nameof(order));

        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        _coefficients = GetDaubechiesCoefficients(order);
    }

    public T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        if (t < 0 || t > _order - 1)
            return _numOps.Zero;

        T result = _numOps.Zero;
        for (int k = 0; k < _order; k++)
        {
            double shiftedT = t - k;
            if (shiftedT >= 0 && shiftedT < 1)
            {
                result = _numOps.Add(result, _numOps.Multiply(_coefficients[k], _numOps.FromDouble(ScalingFunction(shiftedT))));
            }
        }

        return result;
    }

    private double ScalingFunction(double t)
    {
        if (t < 0 || t > 1)
            return 0;

        double result = 0;
        for (int k = 0; k < _order; k++)
        {
            result += Convert.ToDouble(_coefficients[k]) * ScalingFunction(2 * t - k);
        }

        return result;
    }

    private T[] GetDaubechiesCoefficients(int order)
    {
        int N = order / 2;
        double[] a = new double[N];
        a[0] = 1;
        double[] coef = new double[order];

        for (int k = 1; k < N; k++)
        {
            a[k] = a[k - 1] * (N + k - 1) / k * (0.5 - k) / (N - k);
        }

        double sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += a[k] * a[k];
        }

        sum = Math.Sqrt(sum);

        for (int k = 0; k < N; k++)
        {
            a[k] /= sum;
        }

        for (int k = 0; k < order; k++)
        {
            double sum2 = 0;
            for (int n = 0; n <= N; n++)
            {
                if (k - 2 * n >= 0 && k - 2 * n < N)
                {
                    sum2 += a[n] * a[k - 2 * n];
                }
            }
            coef[k] = Math.Pow(-1, k) * sum2 * Math.Sqrt(2);
        }

        return [.. coef.Select(c => _numOps.FromDouble(c))];
    }

    public Vector<T> GetCoefficients() => new Vector<T>(_coefficients);
}