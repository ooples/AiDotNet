namespace AiDotNet.NeuralNetworks.Layers;

public abstract class LayerBase<T> : ILayer<T>
{
    protected IActivationFunction<T>? ScalarActivation { get; }
    protected IVectorActivationFunction<T>? VectorActivation { get; }
    protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
    protected Random Random => new();

    protected int[] InputShape { get; }
    protected int[] OutputShape { get; }

    protected LayerBase(int[] inputShape, int[] outputShape)
    {
        InputShape = inputShape;
        OutputShape = outputShape;
    }

    protected LayerBase(int[] inputShape, int[] outputShape, IActivationFunction<T> scalarActivation)
        : this(inputShape, outputShape)
    {
        ScalarActivation = scalarActivation;
    }

    protected LayerBase(int[] inputShape, int[] outputShape, IVectorActivationFunction<T> vectorActivation)
        : this(inputShape, outputShape)
    {
        VectorActivation = vectorActivation;
    }

    public abstract Tensor<T> Forward(Tensor<T> input);
    public abstract Tensor<T> Backward(Tensor<T> outputGradient);
    public abstract void UpdateParameters(T learningRate);

    protected static int[] CalculateInputShape(int inputDepth, int height, int width)
    {
        return [1, inputDepth, height, width];
    }

    protected static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [1, outputDepth, outputHeight, outputWidth];
    }

    protected Tensor<T> ApplyActivation(Tensor<T> input)
    {
        if (input.Rank != 1)
            throw new ArgumentException("Input tensor must be rank-1 (vector).");

        Vector<T> inputVector = input.ToVector();
        Vector<T> outputVector = ApplyActivationToVector(inputVector);

        return Tensor<T>.FromVector(outputVector);
    }

    private Vector<T> ApplyActivationToVector(Vector<T> input)
    {
        if (VectorActivation != null)
        {
            return VectorActivation.Activate(input);
        }
        else if (ScalarActivation != null)
        {
            return input.Transform(ScalarActivation.Activate);
        }
        else
        {
            return input; // Identity activation
        }
    }

    protected Tensor<T> ApplyActivationDerivative(Tensor<T> input, Tensor<T> outputGradient)
    {
        if (input.Rank != outputGradient.Rank)
            throw new ArgumentException("Input and output gradient tensors must have the same rank.");

        if (VectorActivation != null)
        {
            // Use the vector activation function's derivative method
            return VectorActivation.Derivative(input).Multiply(outputGradient);
        }
        else if (ScalarActivation != null)
        {
            // Element-wise application of scalar activation derivative
            return input.Transform(ScalarActivation.Derivative).ElementwiseMultiply(outputGradient);
        }
        else
        {
            // Identity activation: derivative is just the output gradient
            return outputGradient;
        }
    }

    protected Matrix<T> ComputeActivationJacobian(Vector<T> input)
    {
        if (VectorActivation != null)
        {
            return VectorActivation.Derivative(input);
        }
        else if (ScalarActivation != null)
        {
            // Create a diagonal matrix with the derivatives
            Vector<T> derivatives = input.Transform(ScalarActivation.Derivative);
            return Matrix<T>.CreateDiagonal(derivatives);
        }
        else
        {
            // Identity function: Jacobian is the identity matrix
            return Matrix<T>.CreateIdentity(input.Length);
        }
    }

    protected Vector<T> ApplyActivationDerivative(Vector<T> input, Vector<T> outputGradient)
    {
        Matrix<T> jacobian = ComputeActivationJacobian(input);
        return jacobian.Multiply(outputGradient);
    }
}