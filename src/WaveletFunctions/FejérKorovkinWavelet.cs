namespace AiDotNet.WaveletFunctions;

public class FejérKorovkinWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _order;
    private readonly Vector<T> _coefficients;
    private Vector<T> _scalingCoefficients;
    private Vector<T> _waveletCoefficients;

    public FejérKorovkinWavelet(int order = 4)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        _coefficients = GetFejérKorovkinCoefficients(_order);
        _scalingCoefficients = new Vector<T>(_order);
        _waveletCoefficients = new Vector<T>(_order);
        InitializeCoefficients();
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

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length < _order)
        {
            throw new ArgumentException($"Input vector must have at least {_order} elements.");
        }

        int outputLength = input.Length / 2;
        var approximation = new Vector<T>(outputLength);
        var detail = new Vector<T>(outputLength);

        for (int i = 0; i < outputLength; i++)
        {
            T approx = _numOps.Zero;
            T det = _numOps.Zero;

            for (int j = 0; j < _order; j++)
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

    private void InitializeCoefficients()
    {
        _scalingCoefficients = new Vector<T>(_order);
        _waveletCoefficients = new Vector<T>(_order);

        for (int k = 0; k < _order; k++)
        {
            double t = (k + 0.5) * Math.PI / _order;
            double scalingCoeff = Math.Sqrt(2.0 / _order) * Math.Cos(t);
            double waveletCoeff = Math.Sqrt(2.0 / _order) * Math.Sin(t);

            _scalingCoefficients[k] = _numOps.FromDouble(scalingCoeff);
            _waveletCoefficients[k] = _numOps.FromDouble(waveletCoeff);
        }

        NormalizeCoefficients(_scalingCoefficients);
        NormalizeCoefficients(_waveletCoefficients);
    }

    private void NormalizeCoefficients(Vector<T> coefficients)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < coefficients.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Square(coefficients[i]));
        }

        T normalizationFactor = _numOps.Sqrt(_numOps.Divide(_numOps.One, sum));

        for (int i = 0; i < coefficients.Length; i++)
        {
            coefficients[i] = _numOps.Multiply(coefficients[i], normalizationFactor);
        }
    }
}