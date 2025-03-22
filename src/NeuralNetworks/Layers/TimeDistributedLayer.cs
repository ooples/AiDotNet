namespace AiDotNet.NeuralNetworks.Layers;

public class TimeDistributedLayer<T> : LayerBase<T>
{
    private readonly LayerBase<T> _innerLayer;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    public override bool SupportsTraining => _innerLayer.SupportsTraining;

    public TimeDistributedLayer(LayerBase<T> innerLayer, IActivationFunction<T>? activationFunction = null, int[]? inputShape = null)
        : base(CalculateInputShape(innerLayer, inputShape), CalculateOutputShape(innerLayer, inputShape), activationFunction ?? new ReLUActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    public TimeDistributedLayer(LayerBase<T> innerLayer, IVectorActivationFunction<T>? vectorActivationFunction = null, int[]? inputShape = null)
        : base(CalculateInputShape(innerLayer, inputShape), CalculateOutputShape(innerLayer, inputShape), vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    private static int[] CalculateInputShape(LayerBase<T> innerLayer, int[]? inputShape)
    {
        int[] result;
        if (inputShape != null && inputShape.Length >= 2)
        {
            result = new int[inputShape.Length];
            result[0] = -1;
            Array.Copy(inputShape, 1, result, 1, inputShape.Length - 1);

            return result;
        }

        int[] innerShape = innerLayer.GetInputShape();
        result = new int[innerShape.Length + 1];
        result[0] = -1;
        Array.Copy(innerShape, 0, result, 1, innerShape.Length);

        return result;
    }

    private static int[] CalculateOutputShape(LayerBase<T> innerLayer, int[]? inputShape)
    {
        int[] result;
        if (inputShape != null && inputShape.Length >= 2)
        {
            int[] innerOutputShape = innerLayer.GetOutputShape();
            result = new int[innerOutputShape.Length + 1];
            result[0] = -1;
            result[1] = inputShape[1];
            Array.Copy(innerOutputShape, 1, result, 2, innerOutputShape.Length - 1);

            return result;
        }

        int[] innerShape = innerLayer.GetOutputShape();
        result = new int[innerShape.Length + 1];
        result[0] = -1;
        Array.Copy(innerShape, 0, result, 1, innerShape.Length);

        return result;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int timeSteps = input.Shape[0];
        int batchSize = input.Shape[1];

        var outputShape = new[] { timeSteps, batchSize }.Concat(_innerLayer.GetOutputShape()).ToArray();
        var output = new Tensor<T>(outputShape);

        for (int t = 0; t < timeSteps; t++)
        {
            var stepInput = input.Slice(0, t, 1);
            var stepOutput = _innerLayer.Forward(stepInput);
            output.SetSlice(0, t, stepOutput);
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int timeSteps = _lastInput.Shape[0];
        int batchSize = _lastInput.Shape[1];

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        if (ScalarActivation != null)
        {
            outputGradient = outputGradient.ElementwiseMultiply(_lastOutput.Transform((x, _) => ScalarActivation.Derivative(x)));
        }
        else if (VectorActivation != null)
        {
            outputGradient = outputGradient.ElementwiseMultiply(VectorActivation.Derivative(_lastOutput));
        }

        for (int t = 0; t < timeSteps; t++)
        {
            var stepOutputGradient = outputGradient.Slice(0, t, 1);
            var stepInputGradient = _innerLayer.Backward(stepOutputGradient);
            inputGradient.SetSlice(0, t, stepInputGradient);
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        _innerLayer.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        // Return the parameters of the inner layer
        return _innerLayer.GetParameters();
    }

    public override void ResetState()
    {
        // Reset the inner layer's state
        _innerLayer.ResetState();
    
        // Clear cached values
        _lastInput = null;
        _lastOutput = null;
    }
}