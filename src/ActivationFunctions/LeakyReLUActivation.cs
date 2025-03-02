namespace AiDotNet.ActivationFunctions;

public class LeakyReLUActivation<T> : ActivationFunctionBase<T>
{
    private readonly T _alpha;

    public LeakyReLUActivation(double alpha = 0.01)
    {
        _alpha = NumOps.FromDouble(alpha);
    }

    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? input : NumOps.Multiply(_alpha, input);
    }

    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(x => Activate(x));
    }

    public override T Derivative(T input)
    {
        return NumOps.GreaterThan(input, NumOps.Zero) ? NumOps.One : _alpha;
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        int size = input.Length;
        Matrix<T> jacobian = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = Derivative(input[i]);
                }
                else
                {
                    jacobian[i, j] = NumOps.Zero;
                }
            }
        }

        return jacobian;
    }
}