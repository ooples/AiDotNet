namespace AiDotNet.Interpolation;

public class LagrangePolynomialInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly INumericOperations<T> _numOps;

    public LagrangePolynomialInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        if (x.Length < 2)
        {
            throw new ArgumentException("Lagrange polynomial interpolation requires at least 2 points.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Interpolate(T x)
    {
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T term = _y[i];
            for (int j = 0; j < _x.Length; j++)
            {
                if (i != j)
                {
                    term = _numOps.Multiply(term, 
                        _numOps.Divide(
                            _numOps.Subtract(x, _x[j]),
                            _numOps.Subtract(_x[i], _x[j])
                        )
                    );
                }
            }
            result = _numOps.Add(result, term);
        }

        return result;
    }
}