namespace AiDotNet.ActivationFunctions;

public class GaussianActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = exp(-x^2)
        T negativeSquare = NumOps.Negate(NumOps.Multiply(input, input));
        return NumOps.Exp(negativeSquare);
    }

    public override T Derivative(T input)
    {
        // f'(x) = -2x * exp(-x^2)
        T activationValue = Activate(input);
        T negativeTwo = NumOps.FromDouble(-2);

        return NumOps.Multiply(NumOps.Multiply(negativeTwo, input), activationValue);
    }
}