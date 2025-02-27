namespace AiDotNet.ActivationFunctions;

public class IdentityActivation<T> : ActivationFunctionBase<T>
{
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
        return Vector<T>.CreateDefault(input.Length, NumOps.One).ToDiagonalMatrix();
    }

    protected override bool SupportsScalarOperations() => true;
}