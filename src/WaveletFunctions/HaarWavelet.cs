namespace AiDotNet.WaveletFunctions;

public class HaarWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public HaarWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        if (_numOps.GreaterThanOrEquals(x, _numOps.Zero) && _numOps.LessThan(x, _numOps.FromDouble(0.5)))
            return _numOps.One;
        if (_numOps.GreaterThanOrEquals(x, _numOps.FromDouble(0.5)) && _numOps.LessThan(x, _numOps.One))
            return _numOps.FromDouble(-1);

        return _numOps.Zero;
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for Haar wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        for (int i = 0; i < halfLength; i++)
        {
            T sum = _numOps.Add(input[2 * i], input[2 * i + 1]);
            T diff = _numOps.Subtract(input[2 * i], input[2 * i + 1]);

            approximation[i] = _numOps.Multiply(_numOps.FromDouble(1.0 / Math.Sqrt(2)), sum);
            detail[i] = _numOps.Multiply(_numOps.FromDouble(1.0 / Math.Sqrt(2)), diff);
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        return new Vector<T>(new T[]
        {
            _numOps.FromDouble(1.0 / Math.Sqrt(2)),
            _numOps.FromDouble(1.0 / Math.Sqrt(2))
        });
    }

    public Vector<T> GetWaveletCoefficients()
    {
        return new Vector<T>(new T[]
        {
            _numOps.FromDouble(1.0 / Math.Sqrt(2)),
            _numOps.FromDouble(-1.0 / Math.Sqrt(2))
        });
    }
}