namespace AiDotNet.ActivationFunctions;

public class SoftSignActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = x / (1 + |x|)
        T absInput = NumOps.Abs(input);
        T denominator = NumOps.Add(NumOps.One, absInput);

        return NumOps.Divide(input, denominator);
    }

    public override T Derivative(T input)
    {
        // f'(x) = 1 / (1 + |x|)^2
        T absInput = NumOps.Abs(input);
        T denominator = NumOps.Add(NumOps.One, absInput);
        T squaredDenominator = NumOps.Multiply(denominator, denominator);

        return NumOps.Divide(NumOps.One, squaredDenominator);
    }
}