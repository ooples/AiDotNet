namespace AiDotNet.WaveletFunctions;

public class CoifletWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T[] _coefficients;
    private readonly int _order;

    public CoifletWavelet(int order = 2)
    {
        if (order < 1 || order > 5)
            throw new ArgumentException("Order must be between 1 and 5.", nameof(order));

        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        double[] doubleCoefficients = CoifletWavelet<T>.GetCoifletCoefficients(order);
        _coefficients = [.. doubleCoefficients.Select(c => _numOps.FromDouble(c))];
    }

    public T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        if (t < 0 || t > 6 * _order - 1)
            return _numOps.Zero;

        T result = _numOps.Zero;
        for (int k = 0; k < 6 * _order; k++)
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
        for (int k = 0; k < 6 * _order; k++)
        {
            result += Convert.ToDouble(_coefficients[k]) * ScalingFunction(2 * t - k);
        }

        return result;
    }

    public (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
    {
        var lowPass = GetScalingCoefficients();
        var highPass = GetWaveletCoefficients();

        var approximation = Convolve(input, lowPass);
        var detail = Convolve(input, highPass);

        // Downsample by 2
        approximation = Downsample(approximation, 2);
        detail = Downsample(detail, 2);

        return (approximation, detail);
    }

    public Vector<T> GetScalingCoefficients()
    {
        double[] coeffs = CoifletWavelet<T>.GetCoifletCoefficients(_order);
        return NormalizeAndConvert(coeffs);
    }

    public Vector<T> GetWaveletCoefficients()
    {
        double[] coeffs = CoifletWavelet<T>.GetCoifletCoefficients(_order);
        int n = coeffs.Length;
        double[] highPass = new double[n];
        for (int i = 0; i < n; i++)
        {
            highPass[i] = Math.Pow(-1, i) * coeffs[n - 1 - i];
        }

        return NormalizeAndConvert(highPass);
    }

    private static double[] GetCoifletCoefficients(int order)
    {
        return order switch
        {
            1 => [
                                -0.0156557281, -0.0727326195, 0.3848648469, 0.8525720202,
                    0.3378976625, -0.0727326195
                            ],
            2 => [
                    -0.0007205494, -0.0018232088, 0.0056114348, 0.0236801719,
                    -0.0594344186, -0.0764885990, 0.4170051844, 0.8127236354,
                    0.3861100668, -0.0673725547, -0.0414649367, 0.0163873364
                ],
            3 => [
                    -0.0000345997, -0.0000709833, 0.0004662169, 0.0011175870,
                    -0.0025745176, -0.0090079761, 0.0158805448, 0.0345550275,
                    -0.0823019271, -0.0717998808, 0.4284834638, 0.7937772226,
                    0.4051769024, -0.0611233900, -0.0657719112, 0.0234526961,
                    0.0077825964, -0.0037935128
                ],
            4 => [
                    -0.0000017849, -0.0000032596, 0.0000312298, 0.0000623390,
                    -0.0002599752, -0.0005890207, 0.0015880544, 0.0034555027,
                    -0.0096666678, -0.0166823799, 0.0237580206, 0.0594900371,
                    -0.0931738402, -0.0673766885, 0.4343860564, 0.7822389309,
                    0.4153084070, -0.0560773133, -0.0812666930, 0.0266823001,
                    0.0160689439, -0.0073461663, -0.0016294920, 0.0008923136
                ],
            5 => [
                    -0.0000000951, -0.0000001674, 0.0000020637, 0.0000039763,
                    -0.0000249846, -0.0000527754, 0.0002127442, 0.0004741614,
                    -0.0015070938, -0.0029365387, 0.0073875175, 0.0152161264,
                    -0.0206771736, -0.0430394707, 0.0375074841, 0.0778259642,
                    -0.1004362894, -0.0632407176, 0.4379916262, 0.7742896037,
                    0.4215662067, -0.0520431631, -0.0919200105, 0.0281680289,
                    0.0234081567, -0.0101131176, -0.0041705655, 0.0021782363,
                    0.0003585498, -0.0002120721
                ],
            _ => throw new ArgumentOutOfRangeException(nameof(order), "Unsupported Coiflet order."),
        };
    }

    private Vector<T> NormalizeAndConvert(double[] coeffs)
    {
        double normFactor = Math.Sqrt(coeffs.Sum(c => c * c));
        return new Vector<T>([.. coeffs.Select(c => _numOps.FromDouble(c / normFactor))]);
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
}