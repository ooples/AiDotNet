namespace AiDotNet.ActivationFunctions;

public class SignActivation<T> : ActivationFunctionBase<T>
{
    protected override bool SupportsScalarOperations() => true;

    public override T Activate(T input)
    {
        if (NumOps.LessThan(input, NumOps.Zero))
            return NumOps.FromDouble(-1);
        else if (NumOps.GreaterThan(input, NumOps.Zero))
            return NumOps.One;
        else
            return NumOps.Zero;
    }

    public override T Derivative(T input)
    {
        // The derivative of the sign function is 0 everywhere except at 0,
        // where it's undefined. We'll return 0 for all inputs.
        return NumOps.Zero;
    }

    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> output = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);
        // The Jacobian matrix will be all zeros
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                jacobian[i, j] = NumOps.Zero;
            }
        }

        return jacobian;
    }

    public override Tensor<T> Activate(Tensor<T> input)
    {
        Tensor<T> output = new Tensor<T>(input.Shape);
        int totalElements = input.Shape.Aggregate(1, (a, b) => a * b);

        for (int i = 0; i < totalElements; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    public override Tensor<T> Derivative(Tensor<T> input)
    {
        int[] outputShape = new int[input.Shape.Length + 1];
        Array.Copy(input.Shape, outputShape, input.Shape.Length);
        outputShape[outputShape.Length - 1] = input.Shape[input.Shape.Length - 1];

        Tensor<T> output = new Tensor<T>(outputShape);
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