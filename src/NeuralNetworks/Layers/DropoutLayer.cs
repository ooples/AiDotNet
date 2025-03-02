namespace AiDotNet.NeuralNetworks.Layers;

public class DropoutLayer<T> : LayerBase<T>
{
    private readonly T _dropoutRate;
    private readonly T _scale;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _dropoutMask;

    public override bool SupportsTraining => true;

    public DropoutLayer(double dropoutRate = 0.5)
        : base(Array.Empty<int>(), []) // Dropout layer doesn't change the shape of the input
    {
        if (dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentException("Dropout rate must be between 0 and 1", nameof(dropoutRate));

        _dropoutRate = NumOps.FromDouble(dropoutRate);
        _scale = NumOps.FromDouble(1.0 / (1.0 - dropoutRate));
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        if (!IsTrainingMode)
            return input;

        _dropoutMask = new Tensor<T>(input.Shape);
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            if (Random.NextDouble() > Convert.ToDouble(_dropoutRate))
            {
                _dropoutMask[i] = _scale;
                output[i] = NumOps.Multiply(input[i], _scale);
            }
            else
            {
                _dropoutMask[i] = NumOps.Zero;
                output[i] = NumOps.Zero;
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _dropoutMask == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (!IsTrainingMode)
            return outputGradient;

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < outputGradient.Length; i++)
        {
            inputGradient[i] = NumOps.Multiply(outputGradient[i], _dropoutMask[i]);
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Dropout layer has no parameters to update
    }

    public override Vector<T> GetParameters()
    {
        // Dropout layer has no trainable parameters
        return new Vector<T>(0);
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _dropoutMask = null;
    }
}