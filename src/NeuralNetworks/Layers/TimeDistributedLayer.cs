namespace AiDotNet.NeuralNetworks.Layers;

public class TimeDistributedLayer<T> : LayerBase<T>
{
    private readonly LayerBase<T> _innerLayer;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    public TimeDistributedLayer(LayerBase<T> innerLayer, IActivationFunction<T>? activationFunction = null)
        : base(CalculateInputShape(innerLayer), CalculateOutputShape(innerLayer), activationFunction ?? new ReLUActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    public TimeDistributedLayer(LayerBase<T> innerLayer, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(CalculateInputShape(innerLayer), CalculateOutputShape(innerLayer), vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    private static int[] CalculateInputShape(LayerBase<T> innerLayer)
    {
        return [-1, .. innerLayer.GetInputShape()];
    }

    private static int[] CalculateOutputShape(LayerBase<T> innerLayer)
    {
        return [-1, .. innerLayer.GetOutputShape()];
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
}