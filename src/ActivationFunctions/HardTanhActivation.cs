namespace AiDotNet.ActivationFunctions;

public class HardTanhActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // Hard Tanh: max(-1, min(1, x))
        T minBound = NumOps.FromDouble(-1);
        T maxBound = NumOps.One;

        return MathHelper.Max(minBound, MathHelper.Min(maxBound, input));
    }

    public override T Derivative(T input)
    {
        // Derivative of Hard Tanh:
        // 1 if -1 < x < 1
        // 0 otherwise
        T minBound = NumOps.FromDouble(-1);
        T maxBound = NumOps.One;

        if (NumOps.GreaterThan(input, minBound) && NumOps.LessThan(input, maxBound))
        {
            return NumOps.One;
        }

        return NumOps.Zero;
    }
}