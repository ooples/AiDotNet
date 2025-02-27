namespace AiDotNet.ActivationFunctions;

public class TanhActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        return MathHelper.Tanh(input);
    }

    public override T Derivative(T input)
    {
        T tanh = MathHelper.Tanh(input);
        return NumOps.Subtract(NumOps.One, NumOps.Multiply(tanh, tanh));
    }
}