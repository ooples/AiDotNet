namespace AiDotNet.Wavelets;

public class MeyerWavelet<T> : IWaveletFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public MeyerWavelet()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Calculate(T x)
    {
        double t = Convert.ToDouble(x);
        return _numOps.FromDouble(MeyerFunction(t));
    }

    private double MeyerFunction(double t)
    {
        double abst = Math.Abs(t);

        if (abst < 2 * Math.PI / 3)
        {
            return 0;
        }
        else if (abst < 4 * Math.PI / 3)
        {
            double y = 9 / 4.0 * Math.Pow(abst / (2 * Math.PI) - 1 / 3.0, 2);
            return Math.Sin(2 * Math.PI * y) * Math.Sqrt(2 / 3.0 * AuxiliaryFunction(y));
        }
        else if (abst < 8 * Math.PI / 3)
        {
            double y = 9 / 4.0 * Math.Pow(2 / 3.0 - abst / (2 * Math.PI), 2);
            return Math.Sin(2 * Math.PI * y) * Math.Sqrt(2 / 3.0 * AuxiliaryFunction(1 - y));
        }
        else
        {
            return 0;
        }
    }

    private double AuxiliaryFunction(double x)
    {
        if (x <= 0)
            return 0;
        if (x >= 1)
            return 1;

        return x * x * (3 - 2 * x);
    }
}