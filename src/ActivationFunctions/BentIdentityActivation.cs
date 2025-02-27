namespace AiDotNet.ActivationFunctions;

public class BentIdentityActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // f(x) = (sqrt(x^2 + 1) - 1) / 2 + x
        T squarePlusOne = NumOps.Add(NumOps.Multiply(input, input), NumOps.One);
        T sqrtTerm = NumOps.Sqrt(squarePlusOne);
        T firstTerm = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Subtract(sqrtTerm, NumOps.One));

        return NumOps.Add(firstTerm, input);
    }

    public override T Derivative(T input)
    {
        // f'(x) = x / (2 * sqrt(x^2 + 1)) + 1
        T squarePlusOne = NumOps.Add(NumOps.Multiply(input, input), NumOps.One);
        T sqrtTerm = NumOps.Sqrt(squarePlusOne);
        T firstTerm = NumOps.Divide(input, NumOps.Multiply(NumOps.FromDouble(2), sqrtTerm));

        return NumOps.Add(firstTerm, NumOps.One);
    }
}