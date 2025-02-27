namespace AiDotNet.NeuralNetworks.Layers;

public class ResidualLayer<T> : LayerBase<T>
{
    private readonly ILayer<T> _innerLayer;
    private Tensor<T>? _lastInput;

    public ResidualLayer(ILayer<T> innerLayer)
        : base(innerLayer.GetInputShape(), innerLayer.GetOutputShape())
    {
        if (!Enumerable.SequenceEqual(innerLayer.GetInputShape(), innerLayer.GetOutputShape()))
        {
            throw new ArgumentException("Inner layer must have the same input and output shape for residual connections.");
        }

        _innerLayer = innerLayer;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var innerOutput = _innerLayer.Forward(input);
        return input.Add(innerOutput);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var innerGradient = _innerLayer.Backward(outputGradient);
        return outputGradient.Add(innerGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        _innerLayer.UpdateParameters(learningRate);
    }
}