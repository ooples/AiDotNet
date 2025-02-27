namespace AiDotNet.ActivationFunctions;

public class LinearActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        return input;
    }

    public override T Derivative(T input)
    {
        return NumOps.One;
    }

    public override Vector<T> Activate(Vector<T> input)
    {
        return input;
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        return Matrix<T>.CreateIdentity(input.Length);
    }
}