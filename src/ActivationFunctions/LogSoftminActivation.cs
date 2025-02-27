namespace AiDotNet.ActivationFunctions;

public class LogSoftminActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        T minInput = input.Min();
        Vector<T> shiftedExp = input.Transform(x => NumOps.Exp(NumOps.Subtract(minInput, x)));
        T sumExp = shiftedExp.Sum();
        T logSumExp = NumOps.Add(NumOps.Log(sumExp), NumOps.Negate(minInput));

        return input.Transform(x => NumOps.Subtract(NumOps.Negate(x), logSumExp));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> softmin = Activate(input).Transform(NumOps.Exp);
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    jacobian[i, j] = NumOps.Subtract(softmin[i], NumOps.One);
                }
                else
                {
                    jacobian[i, j] = softmin[j];
                }
            }
        }

        return jacobian;
    }
}