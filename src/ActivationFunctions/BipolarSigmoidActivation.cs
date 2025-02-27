namespace AiDotNet.ActivationFunctions;

public class BipolarSigmoidActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _beta;

    public BipolarSigmoidActivation(double beta = 1.0)
    {
        _beta = NumOps.FromDouble(beta);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = (1 - exp(-βx)) / (1 + exp(-βx))
        T negBetaX = NumOps.Negate(NumOps.Multiply(_beta, input));
        T expNegBetaX = NumOps.Exp(negBetaX);
        T numerator = NumOps.Subtract(NumOps.One, expNegBetaX);
        T denominator = NumOps.Add(NumOps.One, expNegBetaX);

        return NumOps.Divide(numerator, denominator);
    }

    public override T Derivative(T input)
    {
        // f'(x) = β * (1 - f(x)^2)
        T activationValue = Activate(input);
        T squaredActivation = NumOps.Multiply(activationValue, activationValue);
        T oneMinus = NumOps.Subtract(NumOps.One, squaredActivation);

        return NumOps.Multiply(_beta, oneMinus);
    }
}