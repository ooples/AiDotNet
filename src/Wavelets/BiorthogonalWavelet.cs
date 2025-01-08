namespace AiDotNet.Wavelets;

public class BiorthogonalWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _decompositionOrder;
    private readonly int _reconstructionOrder;
    private readonly Vector<T> _decompositionCoefficients;
    private readonly Vector<T> _reconstructionCoefficients;

    public BiorthogonalWavelet(int decompositionOrder = 2, int reconstructionOrder = 2)
    {
        if (decompositionOrder < 1 || decompositionOrder > 3)
            throw new ArgumentException("Order must be between 1 and 3.", nameof(decompositionOrder));

        if (reconstructionOrder < 1 || reconstructionOrder > 3)
            throw new ArgumentException("Order must be between 1 and 3.", nameof(reconstructionOrder));

        _numOps = MathHelper.GetNumericOperations<T>();
        _decompositionOrder = decompositionOrder;
        _reconstructionOrder = reconstructionOrder;
        _decompositionCoefficients = GetDecompositionCoefficients(_decompositionOrder);
        _reconstructionCoefficients = GetReconstructionCoefficients(_reconstructionOrder);
    }

    public T Calculate(T x)
    {
        T result = _numOps.Zero;
        for (int k = 0; k < _decompositionCoefficients.Length; k++)
        {
            T shiftedX = _numOps.Subtract(x, _numOps.FromDouble(k));
            result = _numOps.Add(result, _numOps.Multiply(_decompositionCoefficients[k], ScalingFunction(shiftedX)));
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

    private Vector<T> GetDecompositionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>([_numOps.FromDouble(0.7071067811865476), _numOps.FromDouble(0.7071067811865476)]),
            2 => new Vector<T>([ _numOps.FromDouble(-0.1767766952966369), _numOps.FromDouble(0.3535533905932738),
                                             _numOps.FromDouble(1.0606601717798214), _numOps.FromDouble(0.3535533905932738),
                                             _numOps.FromDouble(-0.1767766952966369) ]),
            3 => new Vector<T>([ _numOps.FromDouble(0.0352262918857095), _numOps.FromDouble(-0.0854412738820267),
                                             _numOps.FromDouble(-0.1350110200102546), _numOps.FromDouble(0.4598775021184914),
                                             _numOps.FromDouble(0.8068915093110924), _numOps.FromDouble(0.3326705529500826),
                                             _numOps.FromDouble(-0.0279837694169839) ]),
            _ => throw new ArgumentException($"Biorthogonal wavelet of order {order} is not implemented."),
        };
    }

    private Vector<T> GetReconstructionCoefficients(int order)
    {
        return order switch
        {
            1 => new Vector<T>([_numOps.FromDouble(0.7071067811865476), _numOps.FromDouble(0.7071067811865476)]),
            2 => new Vector<T>([ _numOps.FromDouble(0.3535533905932738), _numOps.FromDouble(1.0606601717798214),
                                             _numOps.FromDouble(0.3535533905932738), _numOps.FromDouble(-0.1767766952966369) ]),
            3 => new Vector<T>([ _numOps.FromDouble(-0.0279837694169839), _numOps.FromDouble(0.3326705529500826),
                                             _numOps.FromDouble(0.8068915093110924), _numOps.FromDouble(0.4598775021184914),
                                             _numOps.FromDouble(-0.1350110200102546), _numOps.FromDouble(-0.0854412738820267),
                                             _numOps.FromDouble(0.0352262918857095) ]),
            _ => throw new ArgumentException($"Biorthogonal wavelet of order {order} is not implemented."),
        };
    }
}