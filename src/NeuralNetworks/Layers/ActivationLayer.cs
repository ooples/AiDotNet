namespace AiDotNet.NeuralNetworks.Layers;

public class ActivationLayer<T> : LayerBase<T>
{
    private Tensor<T>? _lastInput;
    private readonly bool _useVectorActivation;

    public override bool SupportsTraining => false;

    public ActivationLayer(int[] inputShape, IActivationFunction<T> activationFunction)
        : base(inputShape, inputShape, activationFunction)
    {
        _useVectorActivation = false;
    }

    public ActivationLayer(int[] inputShape, IVectorActivationFunction<T> vectorActivationFunction)
        : base(inputShape, inputShape, vectorActivationFunction)
    {
        _useVectorActivation = true;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        return _useVectorActivation ? ApplyVectorActivation(input) : ApplyScalarActivation(input);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (_lastInput.Shape != outputGradient.Shape)
            throw new ArgumentException("Input and output gradient tensors must have the same shape.");

        return _useVectorActivation 
            ? BackwardVectorActivation(outputGradient) 
            : BackwardScalarActivation(outputGradient);
    }

    private Tensor<T> ApplyScalarActivation(Tensor<T> input)
    {
        return input.Transform((x, _) => ScalarActivation!.Activate(x));
    }

    private Tensor<T> ApplyVectorActivation(Tensor<T> input)
    {
        return VectorActivation!.Activate(input);
    }

    private Tensor<T> BackwardScalarActivation(Tensor<T> outputGradient)
    {
        return _lastInput!.Transform((x, indices) => 
            NumOps.Multiply(ScalarActivation!.Derivative(x), outputGradient[indices]));
    }

    private Tensor<T> BackwardVectorActivation(Tensor<T> outputGradient)
    {
        return VectorActivation!.Derivative(_lastInput!) * outputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Activation layer has no parameters to update
    }

    public override Vector<T> GetParameters()
    {
        // Activation layers don't have parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        _lastInput = null;
    }
}