namespace AiDotNet.WindowFunctions;

public class RectangularWindow<T> : IWindowFunction<T>
{
    private readonly INumericOperations<T> _numOps;

    public RectangularWindow()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    public Vector<T> Create(int windowSize)
    {
        return Vector<T>.CreateDefault(windowSize, _numOps.One);
    }

    public WindowFunctionType GetWindowFunctionType() => WindowFunctionType.Rectangular;
}