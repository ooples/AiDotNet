namespace AiDotNet.ActivationFunctions;

public class SquashActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => false;

    public override T Activate(T input)
    {
        throw new NotSupportedException("SquashActivation does not support scalar operations.");
    }

    public override T Derivative(T input)
    {
        throw new NotSupportedException("SquashActivation does not support scalar operations.");
    }

    public override Vector<T> Activate(Vector<T> input)
    {
        T normSquared = input.DotProduct(input);
        T norm = NumOps.Sqrt(normSquared);
        T scale = NumOps.Divide(normSquared, NumOps.Add(NumOps.One, normSquared));

        return input.Multiply(NumOps.Divide(scale, norm));
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        T normSquared = input.DotProduct(input);
        T norm = NumOps.Sqrt(normSquared);
        T scale = NumOps.Divide(normSquared, NumOps.Add(NumOps.One, normSquared));

        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    T term1 = NumOps.Divide(scale, norm);
                    T term2 = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(input[i], input[i]));
                    T term3 = NumOps.Multiply(NumOps.Add(NumOps.One, normSquared), norm);
                    term2 = NumOps.Divide(term2, term3);
                    jacobian[i, j] = NumOps.Subtract(term1, term2);
                }
                else
                {
                    T term = NumOps.Multiply(input[i], input[j]);
                    term = NumOps.Multiply(NumOps.FromDouble(2), term);
                    term = NumOps.Divide(term, NumOps.Multiply(NumOps.Add(NumOps.One, normSquared), norm));
                    term = NumOps.Multiply(scale, term);
                    jacobian[i, j] = term;
                }
            }
        }

        return jacobian;
    }

    public override Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        int batchSize = input.Shape[0];
        int vectorLength = input.Shape[1];

        for (int i = 0; i < batchSize; i++)
        {
            Vector<T> vector = new Vector<T>(vectorLength);
            for (int j = 0; j < vectorLength; j++)
            {
                vector[j] = input[i, j];
            }

            Vector<T> activatedVector = Activate(vector);

            for (int j = 0; j < vectorLength; j++)
            {
                output[i, j] = activatedVector[j];
            }
        }

        return output;
    }

    public override Tensor<T> Derivative(Tensor<T> input)
    {
        Tensor<T> output = new([.. input.Shape, input.Shape.Last()]);
        int batchSize = input.Shape[0];
        int vectorLength = input.Shape[1];

        for (int i = 0; i < batchSize; i++)
        {
            Vector<T> vector = new Vector<T>(vectorLength);
            for (int j = 0; j < vectorLength; j++)
            {
                vector[j] = input[i, j];
            }

            Matrix<T> jacobian = Derivative(vector);

            for (int j = 0; j < vectorLength; j++)
            {
                for (int k = 0; k < vectorLength; k++)
                {
                    output[i, j, k] = jacobian[j, k];
                }
            }
        }

        return output;
    }
}