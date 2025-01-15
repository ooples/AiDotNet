namespace AiDotNet.WindowFunctions;

public class ParzenWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public ParzenWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        Vector<T> window = new Vector<T>(windowSize, _numOps);
        T N = _numOps.FromDouble(windowSize - 1);
        T halfN = _numOps.Divide(N, _numOps.FromDouble(2));

        for (int n = 0; n < windowSize; n++)
        {
            T nT = _numOps.FromDouble(n);
            T absN = _numOps.Abs(_numOps.Subtract(nT, halfN));

            if (_numOps.LessThanOrEquals(absN, _numOps.Divide(N, _numOps.FromDouble(4))))
            {
                T term = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), absN), N);
                window[n] = _numOps.Subtract(_numOps.One, _numOps.Multiply(_numOps.FromDouble(6), _numOps.Add(_numOps.Square(term), _numOps.Multiply(_numOps.FromDouble(-1), _numOps.Power(term, _numOps.FromDouble(3))))));
            }
            else
            {
                T term = _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2), absN), N);
                window[n] = _numOps.Multiply(_numOps.FromDouble(2), _numOps.Power(_numOps.Subtract(_numOps.One, term), _numOps.FromDouble(3)));
            }
        }

        return window;
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Parzen;
}