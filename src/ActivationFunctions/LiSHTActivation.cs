namespace AiDotNet.ActivationFunctions;

public class LiSHTActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = x * tanh(x)
        T tanhInput = MathHelper.Tanh(input);
        return NumOps.Multiply(input, tanhInput);
    }

    public override T Derivative(T input)
    {
        // f'(x) = tanh(x) + x * (1 - tanh^2(x))
        T tanhInput = MathHelper.Tanh(input);
        T tanhSquared = NumOps.Multiply(tanhInput, tanhInput);
        T oneMinus = NumOps.Subtract(NumOps.One, tanhSquared);
        T secondTerm = NumOps.Multiply(input, oneMinus);

        return NumOps.Add(tanhInput, secondTerm);
    }
}