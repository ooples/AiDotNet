namespace AiDotNet.ActivationFunctions;

public class HardSigmoidActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        // Hard Sigmoid: max(0, min(1, (x + 1) / 2))
        T shifted = NumOps.Add(input, NumOps.One);
        T scaled = NumOps.Multiply(shifted, NumOps.FromDouble(0.5));
        T clamped = MathHelper.Max(NumOps.Zero, MathHelper.Min(NumOps.One, scaled));

        return clamped;
    }

    public override T Derivative(T input)
    {
        // Derivative of Hard Sigmoid:
        // 0.5 if -1 < x < 1
        // 0 otherwise
        T minBound = NumOps.FromDouble(-1);
        T maxBound = NumOps.One;

        if (NumOps.GreaterThan(input, minBound) && NumOps.LessThan(input, maxBound))
        {
            return NumOps.FromDouble(0.5);
        }

        return NumOps.Zero;
    }
}