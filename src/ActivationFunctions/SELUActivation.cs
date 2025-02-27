namespace AiDotNet.ActivationFunctions;

public class SELUActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _lambda;
    private readonly T _alpha;

    public SELUActivation()
    {
        _lambda = NumOps.FromDouble(1.0507009873554804934193349852946);
        _alpha = NumOps.FromDouble(1.6732632423543772848170429916717);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return NumOps.Multiply(_lambda, input);
        }
        else
        {
            T expTerm = NumOps.Subtract(NumOps.Exp(input), NumOps.One);
            return NumOps.Multiply(_lambda, NumOps.Multiply(_alpha, expTerm));
        }
    }

    public override T Derivative(T input)
    {
        if (NumOps.GreaterThanOrEquals(input, NumOps.Zero))
        {
            return _lambda;
        }
        else
        {
            return NumOps.Multiply(_lambda, NumOps.Multiply(_alpha, NumOps.Exp(input)));
        }
    }
}