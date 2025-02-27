namespace AiDotNet.ActivationFunctions;

public class ReLUActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? input : NumOps.Zero;
    }

    public override T Derivative(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? NumOps.One : NumOps.Zero;
    }

    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => MathHelper.Max(NumOps.Zero, x));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            jacobian[i, i] = NumOps.GreaterThan(input[i], NumOps.Zero) ? NumOps.One : NumOps.Zero;
        }

        return jacobian;
    }

    public override Tensor<T> Activate(Tensor<T> input)
    {
        return input.Transform((x, _) => MathHelper.Max(NumOps.Zero, x));
    }

    public override Tensor<T> Derivative(Tensor<T> input)
    {
        return input.Transform((x, _) => NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Zero);
    }
}