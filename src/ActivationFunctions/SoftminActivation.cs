namespace AiDotNet.ActivationFunctions;

public class SoftminActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> negInput = input.Transform(NumOps.Negate);
        Vector<T> expValues = negInput.Transform(NumOps.Exp);
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmin = Activate(input);
        int n = softmin.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(softmin[i], NumOps.Subtract(NumOps.One, softmin[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Negate(NumOps.Multiply(softmin[i], softmin[j]));
                }
            }
        }

        return jacobian;
    }
}