namespace AiDotNet.ActivationFunctions;

public class ISRUActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _alpha;

    public ISRUActivation(double alpha = 1.0)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = x / sqrt(1 + αx^2)
        T squaredInput = NumOps.Multiply(input, input);
        T alphaSquaredInput = NumOps.Multiply(_alpha, squaredInput);
        T denominator = NumOps.Sqrt(NumOps.Add(NumOps.One, alphaSquaredInput));

        return NumOps.Divide(input, denominator);
    }

    public override T Derivative(T input)
    {
        // f'(x) = (1 + αx^2)^(-3/2)
        T squaredInput = NumOps.Multiply(input, input);
        T alphaSquaredInput = NumOps.Multiply(_alpha, squaredInput);
        T baseValue = NumOps.Add(NumOps.One, alphaSquaredInput);
        T exponent = NumOps.FromDouble(-1.5);

        return NumOps.Power(baseValue, exponent);
    }
}