namespace AiDotNet.ActivationFunctions;

public class SparsemaxActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        int k = 1;
        int d = input.Length;
        var z = input.OrderByDescending(x => x).ToArray();
        T sum = NumOps.Zero;
        T threshold = NumOps.Zero;

        for (int i = 0; i < d; i++)
        {
            sum = NumOps.Add(sum, z[i]);
            T average = NumOps.Divide(sum, NumOps.FromDouble(i + 1));
            if (NumOps.GreaterThan(average, z[i]))
            {
                k = i;
                threshold = average;
                break;
            }
        }

        if (k == d)
        {
            threshold = NumOps.Divide(sum, NumOps.FromDouble(d));
        }

        return input.Transform(x => MathHelper.Max(NumOps.Zero, NumOps.Subtract(x, threshold)));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        var output = Activate(input);
        int d = input.Length;
        var jacobian = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (NumOps.GreaterThan(output[i], NumOps.Zero))
                {
                    if (i == j)
                    {
                        jacobian[i, j] = NumOps.One;
                    }
                    else
                    {
                        jacobian[i, j] = NumOps.Negate(NumOps.Divide(output[j], output[i]));
                    }
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