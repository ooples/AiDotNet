namespace AiDotNet.RadialBasisFunctions;

public class CubicRBF<T> : IRadialBasisFunction<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly T _width;

    public CubicRBF(double width = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _width = _numOps.FromDouble(width);
    }

    public T Compute(T r)
    {
        T rOverWidth = _numOps.Divide(r, _width);
        return _numOps.Multiply(rOverWidth, _numOps.Multiply(rOverWidth, rOverWidth));
    }

    public T ComputeDerivative(T r)
    {
        T three = _numOps.FromDouble(3.0);
        T rSquared = _numOps.Multiply(r, r);
        T widthCubed = _numOps.Multiply(_width, _numOps.Multiply(_width, _width));

        return _numOps.Divide(_numOps.Multiply(three, rSquared), widthCubed);
    }

    public T ComputeWidthDerivative(T r)
    {
        // For φ(r) = (r/width)³, the width derivative is -3r³/width⁴
        T rCubed = _numOps.Multiply(r, _numOps.Multiply(r, r));
        T widthSquared = _numOps.Multiply(_width, _width);
        T widthFourth = _numOps.Multiply(widthSquared, widthSquared);
        T negThree = _numOps.FromDouble(-3.0);
        T numerator = _numOps.Multiply(negThree, rCubed);
        
        return _numOps.Divide(numerator, widthFourth);
    }
}