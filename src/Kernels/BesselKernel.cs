namespace AiDotNet.Kernels;

public class BesselKernel<T> : IKernelFunction<T>
{
    private readonly T _order;
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public BesselKernel(T? order = default, T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _order = order ?? _numOps.Zero; // Default order is 0 (Bessel function of the first kind, order 0)
        _sigma = sigma ?? _numOps.One;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T norm = x1.EuclideanDistance(x2);
        T scaledNorm = _numOps.Divide(norm, _sigma);
        
        return _numOps.Divide(BesselFunction(_order, scaledNorm), _numOps.Power(scaledNorm, _order));
    }

    private T BesselFunction(T order, T x)
    {
        // Convert order and x to double for easier comparisons
        double orderDouble = Convert.ToDouble(order);
        double xDouble = Convert.ToDouble(x);

        if (xDouble < 0)
        {
            throw new ArgumentException("x must be non-negative for Bessel function");
        }

        if (orderDouble < 0)
        {
            // Use the relation J_(-n)(x) = (-1)^n * J_n(x)
            T result = BesselFunction(_numOps.Abs(order), x);
            return orderDouble % 2 == 0 ? result : _numOps.Negate(result);
        }

        if (xDouble == 0)
        {
            return orderDouble == 0 ? _numOps.One : _numOps.Zero;
        }

        if (xDouble <= 12 || xDouble < Math.Abs(orderDouble))
        {
            // Use series expansion for small x or when x < |order|
            return BesselFunctionSeries(order, x);
        }
        else
        {
            // Use asymptotic expansion for large x
            return BesselFunctionAsymptotic(order, x);
        }
    }

    private T BesselFunctionSeries(T order, T x)
    {
        int maxIterations = 100;
        T sum = _numOps.Zero;
        T term = _numOps.One;
        T factorial = _numOps.One;
        T xSquaredOver4 = _numOps.Divide(_numOps.Square(x), _numOps.FromDouble(4));

        for (int k = 0; k < maxIterations; k++)
        {
            sum = _numOps.Add(sum, term);

            term = _numOps.Divide(
                _numOps.Multiply(_numOps.Negate(term), xSquaredOver4),
                _numOps.Multiply(factorial, _numOps.Add(order, _numOps.FromDouble(k + 1)))
            );

            if (_numOps.LessThan(_numOps.Abs(term), _numOps.FromDouble(1e-15)))
            {
                break;
            }

            factorial = _numOps.Multiply(factorial, _numOps.FromDouble(k + 1));
        }

        return _numOps.Multiply(sum, _numOps.Power(_numOps.Divide(x, _numOps.FromDouble(2)), order));
    }

    private T BesselFunctionAsymptotic(T order, T x)
    {
        T mu = _numOps.Subtract(_numOps.Multiply(order, order), _numOps.FromDouble(0.25));
        T theta = _numOps.Subtract(x, _numOps.Multiply(_numOps.FromDouble(0.25 * Math.PI), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), order), _numOps.One)));

        T p = _numOps.One;
        T q = _numOps.Divide(mu, _numOps.Multiply(_numOps.FromDouble(8), x));

        T cosTheta = MathHelper.Cos(theta);
        T sinTheta = MathHelper.Sin(theta);

        T sqrtX = _numOps.Sqrt(x);
        T sqrtPi = _numOps.Sqrt(_numOps.FromDouble(Math.PI));
        T factor = _numOps.Divide(_numOps.Sqrt(_numOps.FromDouble(2)), _numOps.Multiply(sqrtPi, sqrtX));

        return _numOps.Multiply(factor, _numOps.Add(_numOps.Multiply(p, cosTheta), _numOps.Multiply(q, sinTheta)));
    }
}