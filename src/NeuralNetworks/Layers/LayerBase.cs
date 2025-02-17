namespace AiDotNet.NeuralNetworks.Layers;

public abstract class LayerBase<T> : ILayer<T>
{
    protected IActivationFunction<T>? ScalarActivation { get; private set; }
    protected IVectorActivationFunction<T>? VectorActivation { get; private set; }
    protected bool UsingVectorActivation { get; }
    protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();
    protected Random Random => new();
    protected Vector<T> Parameters;

    protected int[] InputShape { get; private set; }
    protected int[][] InputShapes { get; private set; }
    protected int[] OutputShape { get; private set; }

    protected LayerBase(int[] inputShape, int[] outputShape)
    {
        InputShape = inputShape;
        InputShapes = [inputShape];
        OutputShape = outputShape;
        Parameters = Vector<T>.Empty();
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
        UsingVectorActivation = true;
    }

    protected LayerBase(int[][] inputShapes, int[] outputShape)
    {
        InputShapes = inputShapes;
        InputShape = inputShapes.Length == 1 ? inputShapes[0] : [];
        OutputShape = outputShape;
        Parameters = Vector<T>.Empty();
    }

    protected LayerBase(int[][] inputShapes, int[] outputShape, IActivationFunction<T> scalarActivation)
        : this(inputShapes, outputShape)
    {
        ScalarActivation = scalarActivation;
    }

    protected LayerBase(int[][] inputShapes, int[] outputShape, IVectorActivationFunction<T> vectorActivation)
        : this(inputShapes, outputShape)
    {
        VectorActivation = vectorActivation;
        UsingVectorActivation = true;
    }

    public virtual int[] GetInputShape() => InputShape ?? InputShapes[0];
    public virtual int[][] GetInputShapes() => InputShapes;
    public int[] GetOutputShape() => OutputShape;

    public abstract Tensor<T> Forward(Tensor<T> input);
    public abstract Tensor<T> Backward(Tensor<T> outputGradient);
    public abstract void UpdateParameters(T learningRate);

    public virtual Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
        {
            throw new ArgumentException("At least one input tensor is required.");
        }

        if (inputs.Length == 1)
        {
            // If there's only one input, use the standard Forward method
            return Forward(inputs[0]);
        }

        // Default behavior: concatenate along the channel dimension (assuming NCHW format)
        int channelDimension = 1;

        // Ensure all input tensors have the same shape except for the channel dimension
        for (int i = 1; i < inputs.Length; i++)
        {
            if (inputs[i].Rank != inputs[0].Rank)
            {
                throw new ArgumentException($"All input tensors must have the same rank. Tensor at index {i} has a different rank.");
            }

            for (int dim = 0; dim < inputs[i].Rank; dim++)
            {
                if (dim != channelDimension && inputs[i].Shape[dim] != inputs[0].Shape[dim])
                {
                    throw new ArgumentException($"Input tensors must have the same dimensions except for the channel dimension. Mismatch at dimension {dim} for tensor at index {i}.");
                }
            }
        }

        // Calculate the total number of channels
        int totalChannels = inputs.Sum(t => t.Shape[channelDimension]);

        // Create the output shape
        int[] outputShape = new int[inputs[0].Rank];
        Array.Copy(inputs[0].Shape, outputShape, inputs[0].Rank);
        outputShape[channelDimension] = totalChannels;

        // Create the output tensor
        Tensor<T> output = new Tensor<T>(outputShape);

        // Copy data from input tensors to the output tensor
        int channelOffset = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            int channels = inputs[i].Shape[channelDimension];
            for (int c = 0; c < channels; c++)
            {
                var slice = inputs[i].Slice(channelDimension, c, c + 1);
                output.SetSlice(channelDimension, channelOffset + c, slice);
            }
            channelOffset += channels;
        }

        return output;
    }

    protected static int[] CalculateInputShape(int inputDepth, int height, int width)
    {
        return [1, inputDepth, height, width];
    }

    protected static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [1, outputDepth, outputHeight, outputWidth];
    }

    public virtual LayerBase<T> Copy()
    {
        var copy = (LayerBase<T>)this.MemberwiseClone();
            
        // Deep copy any reference type members
        copy.InputShape = (int[])InputShape.Clone();
        copy.OutputShape = (int[])OutputShape.Clone();

        // Copy activation functions
        if (ScalarActivation != null)
        {
            copy.ScalarActivation = (IActivationFunction<T>)((ICloneable)ScalarActivation).Clone();
        }
        if (VectorActivation != null)
        {
            copy.VectorActivation = (IVectorActivationFunction<T>)((ICloneable)VectorActivation).Clone();
        }

        return copy;
    }

    protected Tensor<T> ApplyActivation(Tensor<T> input)
    {
        if (input.Rank != 1)
            throw new ArgumentException("Input tensor must be rank-1 (vector).");

        Vector<T> inputVector = input.ToVector();
        Vector<T> outputVector = ApplyActivationToVector(inputVector);

        return Tensor<T>.FromVector(outputVector);
    }

    protected Vector<T> ApplyActivationToVector(Vector<T> input)
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

    protected T ApplyActivationDerivative(T input, T outputGradient)
    {
        if (ScalarActivation != null)
        {
            return NumOps.Multiply(ScalarActivation.Derivative(input), outputGradient);
        }
        else
        {
            // Identity activation: derivative is just the output gradient
            return outputGradient;
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
            return input.Transform((x, _) => ScalarActivation.Derivative(x)).ElementwiseMultiply(outputGradient);
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

    public virtual void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        Parameters = parameters;
    }

    public virtual int ParameterCount => Parameters.Length;

    public virtual void Serialize(BinaryWriter writer)
    {
        writer.Write(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            writer.Write(Convert.ToDouble(Parameters[i]));
        }
    }

    public virtual void Deserialize(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        Parameters = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            Parameters[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}