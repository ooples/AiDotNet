namespace AiDotNet.Wavelets;

public class SymletWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Vector<T> _coefficients;
    private readonly int _order;

    public SymletWavelet(int order = 2)
    {
        if (order < 2 || order % 2 != 0)
            throw new ArgumentException("Order must be an even number greater than or equal to 2.", nameof(order));

        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order;
        _coefficients = GetSymletCoefficients(order);
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

    private Vector<T> GetSymletCoefficients(int order)
    {
        // Symlet coefficients for orders 2 to 10
        Dictionary<int, double[]> symletCoefficients = new()
        {
            {2, new double[] {0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551}},
            {4, new double[] {-0.07576571478, -0.02963552764, 0.49761866763, 0.80373875180, 0.29785779560, -0.09921954357, -0.01260396726, 0.03222310060}},
            {6, new double[] {0.01510864240, -0.00934458632, -0.11799011114, -0.04877372718, 0.49105594192, 0.78726836132, 0.33792942549, -0.07234789605, -0.02180815448, 0.03739502729}},
            {8, new double[] {0.00267279339, -0.00054213233, -0.03158203931, 0.00747897219, 0.06453888262, -0.03020833120, -0.18247860592, 0.03763509741, 0.55430561794, 0.76776431605, 0.28862963175, -0.14004724044, 0.00479489959, 0.01774979096}},
            {10, new double[] {0.00077015981, 0.00007701598, -0.00869063506, -0.00089159390, 0.03326700154, -0.01657454163, -0.05754352622, 0.13995779746, 0.01880748318, -0.05441584225, 0.60170454913, 0.72148436142, 0.23089289127, -0.13839521311, 0.03847133821, 0.00760748732}}
        };

        if (symletCoefficients.TryGetValue(order, out double[]? coeffs))
        {
            return new Vector<T>([.. coeffs.Select(c => _numOps.FromDouble(c))]);
        }
        else
        {
            // For orders not in the predefined set, use the approximate method
            return ApproximateSymletCoefficients(order);
        }
    }

    private Vector<T> ApproximateSymletCoefficients(int order)
    {
        var daubechies = new DaubechiesWavelet<T>(order);
        var daubCoeffs = daubechies.GetCoefficients();

        // Create a new vector to store the Symlet coefficients
        var symCoeffs = new Vector<T>(order);

        // Copy the first half of the Daubechies coefficients
        for (int i = 0; i < order / 2; i++)
        {
            symCoeffs[i] = daubCoeffs[i];
        }

        // Reverse and copy the second half of the Daubechies coefficients
        for (int i = order / 2; i < order; i++)
        {
            symCoeffs[i] = daubCoeffs[order - 1 - i];
        }

        // Normalize the coefficients
        T sum = _numOps.Zero;
        for (int i = 0; i < order; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(symCoeffs[i], symCoeffs[i]));
        }

        T normalizationFactor = _numOps.Sqrt(_numOps.Divide(_numOps.FromDouble(2.0), sum));
    
        for (int i = 0; i < order; i++)
        {
            symCoeffs[i] = _numOps.Multiply(symCoeffs[i], normalizationFactor);
        }

        return symCoeffs;
    }
}