namespace AiDotNet.ActivationFunctions;

public class CELUActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _alpha;

    public CELUActivation(double alpha = 1.0)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // CELU: max(0, x) + min(0, α * (exp(x/α) - 1))
        T expTerm = NumOps.Subtract(NumOps.Exp(NumOps.Divide(input, _alpha)), NumOps.One);
        T negativepart = NumOps.Multiply(_alpha, expTerm);
        
        return NumOps.Add(
            MathHelper.Max(NumOps.Zero, input),
            MathHelper.Min(NumOps.Zero, negativepart)
        );
    }

    public override T Derivative(T input)
    {
        // Derivative of CELU:
        // 1 if x >= 0
        // exp(x/α) if x < 0
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return NumOps.One;
        }
        else
        {
            return NumOps.Exp(NumOps.Divide(input, _alpha));
        }
    }
}