namespace AiDotNet.ActivationFunctions;

public class ThresholdedReLUActivation<T> : ActivationFunctionBase<T>
{
    private T _theta;

    public ThresholdedReLUActivation(double theta = 1.0)
    {
        _theta = NumOps.FromDouble(theta);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // ThresholdedReLU: x if x > theta, else 0
        return NumOps.GreaterThan(input, _theta) ? input : NumOps.Zero;
    }

    public override T Derivative(T input)
    {
        // Derivative of ThresholdedReLU:
        // 1 if x > theta
        // 0 otherwise
        return NumOps.GreaterThan(input, _theta) ? NumOps.One : NumOps.Zero;
    }

    public void UpdateTheta(T newTheta)
    {
        _theta = newTheta;
    }
}