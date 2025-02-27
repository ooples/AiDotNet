namespace AiDotNet.Kernels;

public class MaternKernel<T> : IKernelFunction<T>
{
    private readonly T _nu;
    private readonly T _length;
    private readonly INumericOperations<T> _numOps;

    public MaternKernel(T? nu = default, T? length = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _nu = nu ?? _numOps.FromDouble(1.5);
        _length = length ?? _numOps.One;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T distance = _numOps.Sqrt(x1.Subtract(x2).DotProduct(x1.Subtract(x2)));
        T scaledDistance = _numOps.Multiply(_numOps.Sqrt(_numOps.Multiply(_numOps.FromDouble(2), _nu)), 
                                            _numOps.Divide(distance, _length));
        
        T besselTerm = ModifiedBesselFunction(_nu, scaledDistance);
        
        T powerTerm = _numOps.Power(_numOps.FromDouble(2), _numOps.Subtract(_numOps.One, _nu));
        T gammaTerm = StatisticsHelper<T>.Gamma(_nu);
        
        return _numOps.Multiply(_numOps.Multiply(powerTerm, _numOps.Divide(_numOps.One, gammaTerm)),
                                _numOps.Multiply(besselTerm, _numOps.Power(scaledDistance, _nu)));
    }

    private T ModifiedBesselFunction(T order, T x)
    {
        double nu = Convert.ToDouble(order);
        double xDouble = Convert.ToDouble(x);

        if (xDouble < 0)
        {
            throw new ArgumentException("x must be non-negative for modified Bessel function");
        }

        if (xDouble == 0)
        {
            return nu == 0 ? _numOps.FromDouble(double.PositiveInfinity) : _numOps.FromDouble(double.PositiveInfinity);
        }

        if (xDouble <= 2)
        {
            // Use series expansion for small x
            return ModifiedBesselFunctionSeries(order, x);
        }
        else
        {
            // Use asymptotic expansion for large x
            return ModifiedBesselFunctionAsymptotic(order, x);
        }
    }

    private T ModifiedBesselFunctionSeries(T order, T x)
    {
        T sum = _numOps.Zero;
        T term = _numOps.One;
        int k = 0;
        T xOver2 = _numOps.Divide(x, _numOps.FromDouble(2));

        while (true)
        {
            T a = _numOps.Power(xOver2, _numOps.Add(order, _numOps.FromDouble(2 * k)));
            T b = StatisticsHelper<T>.Gamma(_numOps.Add(order, _numOps.FromDouble(k + 1)));
            T c = StatisticsHelper<T>.Gamma(_numOps.FromDouble(k + 1));
            term = _numOps.Divide(a, _numOps.Multiply(b, c));

            if (_numOps.LessThan(_numOps.Abs(term), _numOps.FromDouble(1e-15)))
            {
                break;
            }

            sum = _numOps.Add(sum, term);
            k++;
        }

        return sum;
    }

    private T ModifiedBesselFunctionAsymptotic(T order, T x)
    {
        T sqrtPiOver2x = _numOps.Divide(_numOps.Sqrt(_numOps.FromDouble(Math.PI / 2)), _numOps.Sqrt(x));
        T exp_x = _numOps.Exp(_numOps.Negate(x));

        T p = _numOps.One;
        T q = _numOps.Divide(_numOps.Multiply(_numOps.Multiply(_numOps.FromDouble(4), order), _numOps.Subtract(order, _numOps.One)), _numOps.Multiply(_numOps.FromDouble(8), x));

        return _numOps.Multiply(sqrtPiOver2x, _numOps.Multiply(exp_x, _numOps.Add(p, q)));
    }
}