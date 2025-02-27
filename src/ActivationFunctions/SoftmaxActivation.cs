namespace AiDotNet.ActivationFunctions;

public class SoftmaxActivation<T> : ActivationFunctionBase<T>
{
    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> expValues = input.Transform(NumOps.Exp);
        T sum = expValues.Sum();

        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmaxOutput = Activate(input);
        int size = softmaxOutput.Length;
        Matrix<T> jacobian = new Matrix<T>(size, size);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Multiply(softmaxOutput[i], NumOps.Subtract(NumOps.One, softmaxOutput[i]));
                }
                else
                {
                    jacobian[i, j] = NumOps.Multiply(NumOps.Negate(softmaxOutput[i]), softmaxOutput[j]);
                }
            }
        }

        return jacobian;
    }

    protected override bool SupportsScalarOperations() => false;
}