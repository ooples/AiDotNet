namespace AiDotNet.Wavelets;

public class FejérKorovkinWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;
    private readonly Vector<T> _coefficients;

    public FejérKorovkinWavelet(int order = 4)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        _coefficients = GetFejérKorovkinCoefficients(_order);
    }

    public T Calculate(T x)
    {
        T result = _numOps.Zero;
        for (int k = 0; k < _coefficients.Length; k++)
        {
            T shiftedX = _numOps.Subtract(x, _numOps.FromDouble(k));
            result = _numOps.Add(result, _numOps.Multiply(_coefficients[k], ScalingFunction(shiftedX)));
        }

        return result;
    }

    private T ScalingFunction(T x)
    {
        if (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.One))
        {
            return _numOps.One;
        }

        return _numOps.Zero;
    }

    private Vector<T> GetFejérKorovkinCoefficients(int order)
    {
        if (order < 4 || order % 2 != 0)
        {
            throw new ArgumentException("Order must be an even number greater than or equal to 4.");
        }

        int n = order / 2;
        var coefficients = new List<double>();

        for (int k = -n; k <= n; k++)
        {
            double coeff = 0;
            for (int j = 1; j <= n; j++)
            {
                double theta = Math.PI * j / (2 * n + 1);
                coeff += (1 - Math.Cos(theta)) * Math.Cos(2 * k * theta) / (2 * n + 1);
            }
            coefficients.Add(coeff);
        }

        // Normalize the coefficients
        double sum = coefficients.Sum();
        coefficients = [.. coefficients.Select(c => c / sum)];

        // Convert to type T and return as Vector<T>
        return new Vector<T>([.. coefficients.Select(c => _numOps.FromDouble(c))]);
    }
}