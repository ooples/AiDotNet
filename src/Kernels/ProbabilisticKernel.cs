namespace AiDotNet.Kernels;

public class ProbabilisticKernel<T> : IKernelFunction<T>
{
    private readonly T _sigma;
    private readonly INumericOperations<T> _numOps;

    public ProbabilisticKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.One;
    }

    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T dotProduct = x1.DotProduct(x2);
        T x1Norm = x1.DotProduct(x1);
        T x2Norm = x2.DotProduct(x2);
        T exponent = _numOps.Divide(_numOps.Negate(_numOps.Square(_numOps.Subtract(x1Norm, x2Norm))), 
                                    _numOps.Multiply(_numOps.FromDouble(2), _numOps.Square(_sigma)));
        
        return _numOps.Multiply(_numOps.Exp(exponent), 
                                _numOps.Divide(dotProduct, _numOps.Sqrt(_numOps.Multiply(x1Norm, x2Norm))));
    }
}