namespace AiDotNet.NeuralNetworks.Layers;

public class ConcatenateLayer<T> : LayerBase<T>
{
    private readonly int _axis;
    private Tensor<T>[]? _lastInputs;
    private Tensor<T>? _lastOutput;

    public ConcatenateLayer(int[][] inputShapes, int axis, IActivationFunction<T>? activationFunction = null)
        : base(inputShapes, CalculateOutputShape(inputShapes, axis), activationFunction ?? new IdentityActivation<T>())
    {
        _axis = axis;
        ValidateInputShapes(inputShapes);
    }

    public ConcatenateLayer(int[][] inputShapes, int axis, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShapes, CalculateOutputShape(inputShapes, axis), vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _axis = axis;
        ValidateInputShapes(inputShapes);
    }

    private static void ValidateInputShapes(int[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            throw new ArgumentException("At least two input shapes are required for concatenation.");
        }

        int rank = inputShapes[0].Length;
        if (inputShapes.Any(shape => shape.Length != rank))
        {
            throw new ArgumentException("All input shapes must have the same rank.");
        }
    }

    private static int[] CalculateOutputShape(int[][] inputShapes, int axis)
    {
        int[] outputShape = new int[inputShapes[0].Length];
        Array.Copy(inputShapes[0], outputShape, inputShapes[0].Length);

        for (int i = 1; i < inputShapes.Length; i++)
        {
            outputShape[axis] += inputShapes[i][axis];
        }

        return outputShape;
    }

    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("At least two input tensors are required for concatenation.");
        }

        _lastInputs = inputs;
        _lastOutput = Tensor<T>.Concatenate(inputs, _axis);

        if (ScalarActivation != null)
        {
            _lastOutput = _lastOutput.Transform((x, _) => ScalarActivation.Activate(x));
        }
        else if (VectorActivation != null)
        {
            _lastOutput = VectorActivation.Activate(_lastOutput);
        }

        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInputs == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        if (ScalarActivation != null)
        {
            outputGradient = outputGradient.ElementwiseMultiply(_lastOutput.Transform((x, _) => ScalarActivation.Derivative(x)));
        }
        else if (VectorActivation != null)
        {
            outputGradient = outputGradient.ElementwiseMultiply(VectorActivation.Derivative(_lastOutput));
        }

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        int startIndex = 0;

        for (int i = 0; i < _lastInputs.Length; i++)
        {
            int length = _lastInputs[i].Shape[_axis];
            inputGradients[i] = outputGradient.Slice(_axis, startIndex, length);
            startIndex += length;
        }

        return Tensor<T>.Stack(inputGradients);
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a concatenate layer
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException("ConcatenateLayer requires multiple inputs. Use Forward(params Tensor<T>[] inputs) instead.");
    }
}