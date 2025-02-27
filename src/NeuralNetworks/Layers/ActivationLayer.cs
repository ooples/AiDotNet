namespace AiDotNet.NeuralNetworks.Layers;

public class ActivationLayer<T> : LayerBase<T>
{
    private Tensor<T>? _lastInput;

    public ActivationLayer(int[] inputShape, IActivationFunction<T> activationFunction)
        : base(inputShape, inputShape, activationFunction)
    {
    }

    public ActivationLayer(int[] inputShape, IVectorActivationFunction<T> vectorActivationFunction)
        : base(inputShape, inputShape, vectorActivationFunction)
    {
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        return ApplyActivation(input);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (_lastInput.Shape != outputGradient.Shape)
            throw new ArgumentException("Input and output gradient tensors must have the same shape.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        if (UsingVectorActivation)
        {
            // Handle vector activation
            for (int i = 0; i < _lastInput.Shape[0]; i++)
            {
                var inputVector = _lastInput.GetRow(i);
                var gradientVector = outputGradient.GetRow(i);
                var derivativeVector = ApplyActivationDerivative(inputVector, gradientVector);
                inputGradient.SetRow(i, derivativeVector);
            }
        }
        else
        {
            // Handle scalar activation
            for (int i = 0; i < _lastInput.Shape[0]; i++)
            {
                for (int j = 0; j < _lastInput.Shape[1]; j++)
                {
                    inputGradient[i, j] = ApplyActivationDerivative(_lastInput[i, j], outputGradient[i, j]);
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Activation layer has no parameters to update
    }
}