namespace AiDotNet.ActivationFunctions;

public class SoftPlusActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = ln(1 + e^x)
        T expInput = NumOps.Exp(input);
        T onePlusExp = NumOps.Add(NumOps.One, expInput);

        return NumOps.Log(onePlusExp);
    }

    public override T Derivative(T input)
    {
        // f'(x) = 1 / (1 + e^(-x))
        T negInput = NumOps.Negate(input);
        T expNegInput = NumOps.Exp(negInput);
        T denominator = NumOps.Add(NumOps.One, expNegInput);

        return NumOps.Divide(NumOps.One, denominator);
    }
}