namespace AiDotNet.Interpolation;

public class MovingLeastSquaresInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _z;
    private readonly T _smoothingLength;
    private readonly int _polynomialDegree;
    private readonly INumericOperations<T> _numOps;
    private readonly IMatrixDecomposition<T>? _decomposition;

    public MovingLeastSquaresInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, double smoothingLength = 1.0, int polynomialDegree = 2, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _smoothingLength = _numOps.FromDouble(smoothingLength);
        _polynomialDegree = polynomialDegree;
        _decomposition = decomposition;
    }

    public T Interpolate(T x, T y)
    {
        int basisSize = (_polynomialDegree + 1) * (_polynomialDegree + 2) / 2;
        var A = new Matrix<T>(_x.Length, basisSize);
        var W = new Matrix<T>(_x.Length, _x.Length);
        var b = new Vector<T>(_x.Length);

        for (int i = 0; i < _x.Length; i++)
        {
            T dx = _numOps.Subtract(x, _x[i]);
            T dy = _numOps.Subtract(y, _y[i]);
            T distance = _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
            T weight = CalculateWeight(distance);

            W[i, i] = weight;
            b[i] = _numOps.Multiply(_z[i], weight);

            int index = 0;
            for (int p = 0; p <= _polynomialDegree; p++)
            {
                for (int q = 0; q <= p; q++)
                {
                    A[i, index] = _numOps.Multiply(weight, _numOps.Multiply(_numOps.Power(_x[i], _numOps.FromDouble(p - q)), _numOps.Power(_y[i], _numOps.FromDouble(q))));
                    index++;
                }
            }
        }

        var AtW = A.Transpose().Multiply(W);
        var AtWA = AtW.Multiply(A);
        var AtWb = AtW.Multiply(b);

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(AtWA);
        var coefficients = MatrixSolutionHelper.SolveLinearSystem(AtWb, decomposition);

        T result = _numOps.Zero;
        int coeffIndex = 0;
        for (int p = 0; p <= _polynomialDegree; p++)
        {
            for (int q = 0; q <= p; q++)
            {
                result = _numOps.Add(result, _numOps.Multiply(coefficients[coeffIndex], 
                    _numOps.Multiply(_numOps.Power(x, _numOps.FromDouble(p - q)), _numOps.Power(y, _numOps.FromDouble(q)))));
                coeffIndex++;
            }
        }

        return result;
    }

    private T CalculateWeight(T distance)
    {
        if (_numOps.GreaterThanOrEquals(distance, _smoothingLength))
            return _numOps.Zero;

        T ratio = _numOps.Divide(distance, _smoothingLength);
        return _numOps.Subtract(_numOps.One, _numOps.Multiply(ratio, ratio));
    }
}