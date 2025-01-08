namespace AiDotNet.Wavelets;

public class ReverseBiorthogonalWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _decompositionOrder;
    private readonly int _reconstructionOrder;
    private readonly Vector<T> _decompositionCoefficients;
    private readonly Vector<T> _reconstructionCoefficients;

    public ReverseBiorthogonalWavelet(int decompositionOrder = 1, int reconstructionOrder = 1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionOrder = decompositionOrder;
        _reconstructionOrder = reconstructionOrder;
        _decompositionCoefficients = GetDecompositionCoefficients(decompositionOrder);
        _reconstructionCoefficients = GetReconstructionCoefficients(reconstructionOrder);
    }

    public T Calculate(T x)
    {
        T result = _numOps.Zero;
        int n = _decompositionCoefficients.Length;

        for (int k = 0; k < n; k++)
        {
            T shiftedX = _numOps.Subtract(x, _numOps.FromDouble(k));
            T scaledX = _numOps.Multiply(_numOps.FromDouble(2), shiftedX);
            T scalingTerm = ScalingFunction(scaledX);
            result = _numOps.Add(result, _numOps.Multiply(_decompositionCoefficients[k], scalingTerm));
        }

        return result;
    }

    private T ScalingFunction(T x)
    {
        T absX = _numOps.Abs(x);
        T result = _numOps.Zero;

        if (_numOps.LessThan(absX, _numOps.One))
        {
            result = _numOps.Subtract(_numOps.One, absX);
        }
        else if (_numOps.LessThan(absX, _numOps.FromDouble(2)))
        {
            T temp = _numOps.Subtract(_numOps.FromDouble(2), absX);
            result = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Multiply(temp, temp));
        }

        return result;
    }

    private Vector<T> GetDecompositionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>(
                            [
                                _numOps.FromDouble(0),
                    _numOps.FromDouble(-0.1767766952966369),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(1.0606601717798214),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.1767766952966369)
                            ]),
            2 => new Vector<T>(new T[]
                            {
                    _numOps.FromDouble(0),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.7071067811865476),
                    _numOps.FromDouble(0.3535533905932738)
                            }),
            _ => throw new ArgumentException($"Decomposition order {order} is not supported."),
        };
    }

    private Vector<T> GetReconstructionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>(
                            [
                    _numOps.FromDouble(0),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.7071067811865476),
                    _numOps.FromDouble(0.3535533905932738)
                            ]),
            2 => new Vector<T>(
                [
                    _numOps.FromDouble(0),
                    _numOps.FromDouble(-0.1767766952966369),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(1.0606601717798214),
                    _numOps.FromDouble(0.3535533905932738),
                    _numOps.FromDouble(-0.1767766952966369)
                ]),
            _ => throw new ArgumentException($"Reconstruction order {order} is not supported."),
        };
    }
}