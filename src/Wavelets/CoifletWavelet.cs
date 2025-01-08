namespace AiDotNet.Wavelets;

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
        _coefficients = GetCoifletCoefficients(order);
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

    private T[] GetCoifletCoefficients(int order)
    {
        // Predefined Coiflet coefficients for orders 1 to 5
        double[][] coifCoeffs =
        [
            [-0.0156557281, -0.0727326195, 0.3848648469, 0.8525720202, 0.3378976625, -0.0727326195],
            [-0.0007205494, -0.0018232088, 0.0056114348, 0.0236801719, -0.0594344186, -0.0764885990, 0.4170051844, 0.8127236354, 0.3861100668, -0.0673725547],
            [-0.0000345997, -0.0000709833, 0.0004662169, 0.0011175490, -0.0025745176, -0.0090079761, 0.0158805448, 0.0345550275, -0.0823019271, -0.0717998808, 0.4284834763, 0.7937772226, 0.4051769024, -0.0611233900],
            [-0.0000017849, -0.0000032596, 0.0000312298, 0.0000623390, -0.0002599752, -0.0005890207, 0.0015880544, 0.0034555027, -0.0082301927, -0.0171799881, 0.0430834763, 0.0937772226, -0.1306289962, -0.0713289590, 0.4343860564, 0.7822389309, 0.4153084070, -0.0560773133],
            [-0.0000000951, -0.0000001674, 0.0000020637, 0.0000037346, -0.0000235543, -0.0000471493, 0.0001973325, 0.0003759148, -0.0012804831, -0.0024404618, 0.0063810507, 0.0120036702, -0.0316942697, -0.0520431461, 0.0897510894, 0.1619853843, -0.1740167489, -0.0666274742, 0.4379916262, 0.7742896037, 0.4215662067, -0.0520431461]
        ];

        return [.. coifCoeffs[order - 1].Select(c => _numOps.FromDouble(c))];
    }
}