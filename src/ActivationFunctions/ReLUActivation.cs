namespace AiDotNet.ActivationFunctions;

public class ReLUActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? input : NumOps.Zero;
    }

    public override T Derivative(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? NumOps.One : NumOps.Zero;
    }
}