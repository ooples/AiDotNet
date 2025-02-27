namespace AiDotNet.ActivationFunctions;

public class SphericalSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        // Compute the L2 norm of the input vector
        T norm = NumOps.Sqrt(input.Transform(x => NumOps.Multiply(x, x)).Sum());

        // Normalize the input vector
        Vector<T> normalizedInput = input.Transform(x => NumOps.Divide(x, norm));

        // Apply exponential function to each element
        Vector<T> expValues = normalizedInput.Transform(NumOps.Exp);

        // Compute the sum of exponential values
        T sum = expValues.Sum();

        // Divide each exponential value by the sum
        return expValues.Transform(x => NumOps.Divide(x, sum));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Vector<T> output = Activate(input);
        int d = input.Length;
        Matrix<T> jacobian = new Matrix<T>(d, d);

        T norm = NumOps.Sqrt(input.Transform(x => NumOps.Multiply(x, x)).Sum());

        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                if (i == j)
                {
                    T term1 = NumOps.Multiply(output[i], NumOps.Subtract(NumOps.One, output[i]));
                    T term2 = NumOps.Divide(NumOps.Multiply(input[i], input[i]), NumOps.Multiply(norm, norm));
                    jacobian[i, j] = NumOps.Divide(NumOps.Subtract(term1, term2), norm);
                }
                else
                {
                    T term1 = NumOps.Multiply(NumOps.Negate(output[i]), output[j]);
                    T term2 = NumOps.Divide(NumOps.Multiply(input[i], input[j]), NumOps.Multiply(norm, norm));
                    jacobian[i, j] = NumOps.Divide(NumOps.Subtract(term1, term2), norm);
                }
            }
        }

        return jacobian;
    }
}