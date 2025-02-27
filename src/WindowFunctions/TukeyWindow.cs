namespace AiDotNet.WindowFunctions;

public class TukeyWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _alpha;

    public TukeyWindow(double alpha = 0.5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _alpha = _numOps.FromDouble(alpha);
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize);
        T N = _numOps.FromDouble(windowSize - 1);
        T alphaN = _numOps.Multiply(_alpha, N);

        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            if (_numOps.LessThanOrEquals(nT, _numOps.Multiply(_alpha, _numOps.Divide(N, _numOps.FromDouble(2)))))
            {
                window[n] = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Add(_numOps.One, MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(2), nT), alphaN)))));
            }
            else if (_numOps.GreaterThan(nT, _numOps.Multiply(_alpha, _numOps.Divide(N, _numOps.FromDouble(2)))) && _numOps.LessThan(nT, _numOps.Subtract(N, _numOps.Multiply(_alpha, _numOps.Divide(N, _numOps.FromDouble(2))))))
            {
                window[n] = _numOps.One;
            }
            else
            {
                window[n] = _numOps.Multiply(_numOps.FromDouble(0.5), _numOps.Add(_numOps.One, MathHelper.Cos(_numOps.Multiply(_numOps.FromDouble(Math.PI), _numOps.Add(_numOps.Multiply(_numOps.FromDouble(2), nT), _numOps.Subtract(_numOps.Multiply(_numOps.FromDouble(2), N), alphaN))))));
            }
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Tukey;
}