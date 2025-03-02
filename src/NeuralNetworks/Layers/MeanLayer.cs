namespace AiDotNet.NeuralNetworks.Layers;

public class MeanLayer<T> : LayerBase<T>
{
    public int Axis { get; private set; }

    public override bool SupportsTraining => false;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    public MeanLayer(int[] inputShape, int axis)
        : base(inputShape, CalculateOutputShape(inputShape, axis))
    {
        Axis = axis;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int axis)
    {
        var outputShape = new int[inputShape.Length - 1];
        int outputIndex = 0;
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (i != axis)
            {
                outputShape[outputIndex++] = inputShape[i];
            }
        }

        return outputShape;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var output = new Tensor<T>(OutputShape);

        int axisSize = input.Shape[Axis];
        T axisScale = NumOps.FromDouble(1.0 / axisSize);

        // Iterate over all dimensions except the mean axis
        var indices = new int[input.Shape.Length];
        IterateOverDimensions(input, output, indices, 0, Axis, (input, output, indices) =>
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                sum = NumOps.Add(sum, input[indices]);
            }
            output[indices] = NumOps.Multiply(sum, axisScale);
        });

        _lastOutput = output;
        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        int axisSize = _lastInput.Shape[Axis];
        T axisScale = NumOps.FromDouble(1.0 / axisSize);

        // Iterate over all dimensions except the mean axis
        var indices = new int[_lastInput.Shape.Length];
        IterateOverDimensions(_lastInput, outputGradient, indices, 0, Axis, (_, outputGradient, indices) =>
        {
            for (int i = 0; i < axisSize; i++)
            {
                indices[Axis] = i;
                inputGradient[indices] = NumOps.Multiply(outputGradient[indices], axisScale);
            }
        });

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // MeanLayer has no learnable parameters, so this method is empty
    }

    private void IterateOverDimensions(Tensor<T> input, Tensor<T> output, int[] indices, int currentDim, int skipDim, Action<Tensor<T>, Tensor<T>, int[]> action)
    {
        if (currentDim == input.Shape.Length)
        {
            action(input, output, indices);
            return;
        }

        if (currentDim == skipDim)
        {
            IterateOverDimensions(input, output, indices, currentDim + 1, skipDim, action);
        }
        else
        {
            for (int i = 0; i < input.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                IterateOverDimensions(input, output, indices, currentDim + 1, skipDim, action);
            }
        }
    }

    public override Vector<T> GetParameters()
    {
        // MeanLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastOutput = null;
    }
}