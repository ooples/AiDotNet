global using AiDotNet.RadialBasisFunctions;

namespace AiDotNet.Interpolation;

public class RadialBasisFunctionInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Vector<T> _z;
    private Vector<T> _weights;
    private readonly INumericOperations<T> _numOps;
    private readonly IMatrixDecomposition<T>? _decomposition;
    private readonly IRadialBasisFunction<T> _rbf;

    public RadialBasisFunctionInterpolation(Vector<T> x, Vector<T> y, Vector<T> z, 
        IRadialBasisFunction<T>? rbf = null, IMatrixDecomposition<T>? decomposition = null)
    {
        if (x.Length != y.Length || x.Length != z.Length)
            throw new ArgumentException("Input vectors must have the same length.");
        if (x.Length < 3)
            throw new ArgumentException("At least 3 points are required for RBF interpolation.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
        _weights = Vector<T>.Empty();
        _decomposition = decomposition;
        _rbf = rbf ?? new GaussianRBF<T>();

        CalculateWeights();
    }

    public T Interpolate(T x, T y)
    {
        T result = _numOps.Zero;

        for (int i = 0; i < _x.Length; i++)
        {
            T r = CalculateDistance(x, y, _x[i], _y[i]);
            result = _numOps.Add(result, _numOps.Multiply(_weights[i], _rbf.Compute(r)));
        }

        return result;
    }

    private void CalculateWeights()
    {
        int n = _x.Length;
        Matrix<T> A = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T r = CalculateDistance(_x[i], _y[i], _x[j], _y[j]);
                A[i, j] = _rbf.Compute(r);
            }
        }

        // Solve the system
        var decomposition = _decomposition ?? new LuDecomposition<T>(A);
        _weights = MatrixSolutionHelper.SolveLinearSystem(_z, decomposition);
    }

    private T CalculateDistance(T x1, T y1, T x2, T y2)
    {
        T dx = _numOps.Subtract(x1, x2);
        T dy = _numOps.Subtract(y1, y2);

        return _numOps.Sqrt(_numOps.Add(_numOps.Multiply(dx, dx), _numOps.Multiply(dy, dy)));
    }
}