namespace AiDotNet.ActivationFunctions;

public abstract class ActivationFunctionBase<T> : IActivationFunction<T>, IVectorActivationFunction<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected abstract bool SupportsScalarOperations();

    public virtual T Activate(T input)
    {
        return input; // Default to identity function
    }

    public virtual T Derivative(T input)
    {
        return NumOps.One; // Default to constant derivative of 1
    }

    public virtual Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    public virtual Matrix<T> Derivative(Vector<T> input)
    {
        return Matrix<T>.CreateDiagonal(input.Transform(Derivative));
    }

    public virtual Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    public virtual Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Derivative(input[i]);
        }

        return output;
    }
}