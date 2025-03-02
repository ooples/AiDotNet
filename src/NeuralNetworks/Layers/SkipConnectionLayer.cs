namespace AiDotNet.NeuralNetworks.Layers;

public class SkipConnectionLayer<T> : LayerBase<T>
{
    private readonly ILayer<T>? _innerLayer;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => _innerLayer?.SupportsTraining ?? true;

    public SkipConnectionLayer(ILayer<T>? innerLayer = null, IVectorActivationFunction<T>? vectorActivation = null)
        : base(innerLayer?.GetInputShape() ?? [], innerLayer?.GetOutputShape() ?? [], vectorActivation ?? new LinearActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    public SkipConnectionLayer(ILayer<T>? innerLayer = null, IActivationFunction<T>? scalarActivation = null)
        : base(innerLayer?.GetInputShape() ?? [], innerLayer?.GetOutputShape() ?? [], scalarActivation ?? new LinearActivation<T>())
    {
        _innerLayer = innerLayer;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        if (_innerLayer != null)
        {
            var innerOutput = _innerLayer.Forward(input);
            return input.Add(innerOutput);
        }

        return input;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (_innerLayer != null)
        {
            var innerGradient = _innerLayer.Backward(outputGradient);
            return outputGradient.Add(innerGradient);
        }

        return outputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        _innerLayer?.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        // If there's no inner layer, return an empty vector
        if (_innerLayer == null)
        {
            return Vector<T>.Empty();
        }
    
        // Otherwise, return the parameters of the inner layer
        return _innerLayer.GetParameters();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    
        // Reset the inner layer's state if it exists
        _innerLayer?.ResetState();
    }
}