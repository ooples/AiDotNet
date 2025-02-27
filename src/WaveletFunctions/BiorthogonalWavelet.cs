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

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        if (input.Length % 2 != 0)
            throw new ArgumentException("Input length must be even for biorthogonal wavelet decomposition.");

        int halfLength = input.Length / 2;
        var approximation = new Vector<T>(halfLength);
        var detail = new Vector<T>(halfLength);

        var decompositionLowPass = GetDecompositionLowPassFilter();
        var decompositionHighPass = GetDecompositionHighPassFilter();

        for (int i = 0; i < halfLength; i++)
        {
            T approx = _numOps.Zero;
            T det = _numOps.Zero;

            for (int j = 0; j < decompositionLowPass.Length; j++)
            {
                int index = (2 * i + j) % input.Length;
                approx = _numOps.Add(approx, _numOps.Multiply(decompositionLowPass[j], input[index]));
                det = _numOps.Add(det, _numOps.Multiply(decompositionHighPass[j], input[index]));
            }

            approximation[i] = approx;
            detail[i] = det;
        }

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        return GetReconstructionLowPassFilter();
    }

    public Vector<T> GetWaveletCoefficients()
    {
        return GetReconstructionHighPassFilter();
    }

    private Vector<T> GetDecompositionLowPassFilter()
    {
        double[] coeffs = {
            0.026748757410810,
            -0.016864118442875,
            -0.078223266528990,
            0.266864118442872,
            0.602949018236360,
            0.266864118442872,
            -0.078223266528990,
            -0.016864118442875,
            0.026748757410810
        };

        return NormalizeAndConvert(coeffs);
    }

    private Vector<T> GetDecompositionHighPassFilter()
    {
        double[] coeffs = {
            0.0,
            0.091271763114250,
            -0.057543526228500,
            -0.591271763114250,
            1.115087052456994,
            -0.591271763114250,
            -0.057543526228500,
            0.091271763114250,
            0.0
        };

        return NormalizeAndConvert(coeffs);
    }

    private Vector<T> GetReconstructionLowPassFilter()
    {
        double[] coeffs = {
            0.0,
            -0.091271763114250,
            -0.057543526228500,
            0.591271763114250,
            1.115087052456994,
            0.591271763114250,
            -0.057543526228500,
            -0.091271763114250,
            0.0
        };

        return NormalizeAndConvert(coeffs);
    }

    private Vector<T> GetReconstructionHighPassFilter()
    {
        double[] coeffs = {
            0.026748757410810,
            0.016864118442875,
            -0.078223266528990,
            -0.266864118442872,
            0.602949018236360,
            -0.266864118442872,
            -0.078223266528990,
            0.016864118442875,
            0.026748757410810
        };

        return NormalizeAndConvert(coeffs);
    }

    private Vector<T> NormalizeAndConvert(double[] coeffs)
    {
        double normFactor = Math.Sqrt(coeffs.Sum(c => c * c));
        return new Vector<T>([.. coeffs.Select(c => _numOps.FromDouble(c / normFactor))]);
    }
}