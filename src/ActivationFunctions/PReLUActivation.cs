namespace AiDotNet.ActivationFunctions;

public class PReLUActivation<T> : ActivationFunctionBase<T>
{
    private T _alpha;

    public PReLUActivation(double alpha = 0.01)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // PReLU: max(0, x) + alpha * min(0, x)
        T positivepart = MathHelper.Max(NumOps.Zero, input);
        T negativepart = NumOps.Multiply(_alpha, MathHelper.Min(NumOps.Zero, input));

        return NumOps.Add(positivepart, negativepart);
    }

    public override T Derivative(T input)
    {
        // Derivative of PReLU:
        // 1 if x > 0
        // alpha if x <= 0
        if (NumOps.GreaterThan(input, NumOps.Zero))
        {
            return NumOps.One;
        }

        return _alpha;
    }

    public void UpdateAlpha(T newAlpha)
    {
        _alpha = newAlpha;
    }
}