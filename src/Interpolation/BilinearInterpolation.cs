namespace AiDotNet.Interpolation;

public class BilinearInterpolation<T> : I2DInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly Matrix<T> _z;
    private readonly INumericOperations<T> _numOps;

    public BilinearInterpolation(Vector<T> x, Vector<T> y, Matrix<T> z)
    {
        if (x.Length != z.Rows || y.Length != z.Columns)
            throw new ArgumentException("Input dimensions mismatch.");
        if (x.Length < 2 || y.Length < 2)
            throw new ArgumentException("Bilinear interpolation requires at least a 2x2 grid.");

        _x = x;
        _y = y;
        _z = z;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Interpolate(T x, T y)
    {
        int i = FindInterval(_x, x);
        int j = FindInterval(_y, y);

        T x1 = _x[i];
        T x2 = _x[i + 1];
        T y1 = _y[j];
        T y2 = _y[j + 1];

        T q11 = _z[i, j];
        T q21 = _z[i + 1, j];
        T q12 = _z[i, j + 1];
        T q22 = _z[i + 1, j + 1];

        T dx = _numOps.Subtract(x2, x1);
        T dy = _numOps.Subtract(y2, y1);

        T tx = _numOps.Divide(_numOps.Subtract(x, x1), dx);
        T ty = _numOps.Divide(_numOps.Subtract(y, y1), dy);

        T r1 = _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, tx), q11), _numOps.Multiply(tx, q21));
        T r2 = _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, tx), q12), _numOps.Multiply(tx, q22));

        return _numOps.Add(_numOps.Multiply(_numOps.Subtract(_numOps.One, ty), r1), _numOps.Multiply(ty, r2));
    }

    private int FindInterval(Vector<T> values, T point)
    {
        for (int i = 0; i < values.Length - 1; i++)
        {
            if (_numOps.LessThanOrEquals(point, values[i + 1]))
                return i;
        }

        return values.Length - 2;
    }
}