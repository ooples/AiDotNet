namespace AiDotNet.ActivationFunctions;

public class TaylorSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    private readonly int _order;

    public TaylorSoftmaxActivation(int order = 2)
    {
        _order = order;
    }

    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> expValues = input.Transform(x => TaylorExp(x, _order));
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> output = Activate(input);
        int d = input.Length;
        Matrix<T> jacobian = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(output[i], NumOps.Subtract(NumOps.One, output[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Multiply(NumOps.Negate(output[i]), output[j]);
                }
            }
        }

        return jacobian;
    }

    private T TaylorExp(T x, int order)
    {
        T result = NumOps.One;
        T term = NumOps.One;

        for (int n = 1; n <= order; n++)
        {
            term = NumOps.Divide(NumOps.Multiply(term, x), NumOps.FromDouble(n));
            result = NumOps.Add(result, term);
        }

        return result;
    }
}