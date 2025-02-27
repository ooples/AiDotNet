namespace AiDotNet.ActivationFunctions;

public class LogSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        T maxInput = input.Max();
        Vector<T> shiftedExp = input.Transform(x => NumOps.Exp(NumOps.Subtract(x, maxInput)));
        T sumExp = shiftedExp.Sum();
        T logSumExp = NumOps.Add(NumOps.Log(sumExp), maxInput);

        return input.Transform(x => NumOps.Subtract(x, logSumExp));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmax = input.Transform(NumOps.Exp);
        T sum = softmax.Sum();
        softmax = softmax.Transform(x => NumOps.Divide(x, sum));

        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Subtract(NumOps.One, softmax[i]);
                }
                else
                {
                    jacobian[i, j] = NumOps.Negate(softmax[j]);
                }
            }
        }

        return jacobian;
    }
}