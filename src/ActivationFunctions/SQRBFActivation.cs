namespace AiDotNet.ActivationFunctions;

public class SQRBFActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _beta;

    public SQRBFActivation(double beta = 1.0)
    {
        _beta = NumOps.FromDouble(beta);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = exp(-β * x^2)
        T square = NumOps.Multiply(input, input);
        T negBetaSquare = NumOps.Negate(NumOps.Multiply(_beta, square));

        return NumOps.Exp(negBetaSquare);
    }

    public override T Derivative(T input)
    {
        // f'(x) = -2βx * exp(-β * x^2)
        T activationValue = Activate(input);
        T negTwoBeta = NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(2), _beta));

        return NumOps.Multiply(NumOps.Multiply(negTwoBeta, input), activationValue);
    }
}