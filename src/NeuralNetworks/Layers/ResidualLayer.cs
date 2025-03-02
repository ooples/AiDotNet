namespace AiDotNet.NeuralNetworks.Layers;

public class ResidualLayer<T> : LayerBase<T>
{
    private readonly ILayer<T> _innerLayer;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => _innerLayer.SupportsTraining;

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

    public override Vector<T> GetParameters()
    {
        // Return the parameters of the inner layer
        return _innerLayer.GetParameters();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    
        // Reset the inner layer's state
        _innerLayer.ResetState();
    }
}