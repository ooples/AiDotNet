namespace AiDotNet.ActivationFunctions;

public class SigmoidActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(input))));
    }

    public override T Derivative(T input)
    {
        T sigmoid = Activate(input);
        return NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
    }

    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> sigmoid = Activate(input);
        return Matrix<T>.CreateDiagonal(sigmoid.Transform(s => NumOps.Multiply(s, NumOps.Subtract(NumOps.One, s))));
    }
}