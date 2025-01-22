namespace AiDotNet.Interpolation;

public class NearestNeighborInterpolation<T> : IInterpolation<T>
{
    private readonly Vector<T> _x;
    private readonly Vector<T> _y;
    private readonly INumericOperations<T> _numOps;

    public NearestNeighborInterpolation(Vector<T> x, Vector<T> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Input vectors must have the same length.");
        }

        _x = x;
        _y = y;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public T Interpolate(T x)
    {
        int nearestIndex = FindNearestIndex(x);
        return _y[nearestIndex];
    }

    private int FindNearestIndex(T x)
    {
        int nearestIndex = 0;
        T minDistance = _numOps.Abs(_numOps.Subtract(x, _x[0]));

        for (int i = 1; i < _x.Length; i++)
        {
            T distance = _numOps.Abs(_numOps.Subtract(x, _x[i]));
            if (_numOps.LessThan(distance, minDistance))
            {
                minDistance = distance;
                nearestIndex = i;
            }
        }

        return nearestIndex;
    }
}