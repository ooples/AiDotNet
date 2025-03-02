namespace AiDotNet.NeuralNetworks.Layers;

public class PaddingLayer<T> : LayerBase<T>
{
    private readonly int[] _padding;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    public PaddingLayer(int[] inputShape, int[] padding, IActivationFunction<T>? activationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, padding), activationFunction ?? new IdentityActivation<T>())
    {
        _padding = padding;
    }

    public PaddingLayer(int[] inputShape, int[] padding, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, padding), vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _padding = padding;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int[] padding)
    {
        int[] outputShape = new int[inputShape.Length];
        for (int i = 0; i < inputShape.Length; i++)
        {
            outputShape[i] = inputShape[i] + 2 * padding[i];
        }

        return outputShape;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var paddedOutput = new Tensor<T>(OutputShape);

        int batchSize = input.Shape[0];
        int channels = input.Shape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < input.Shape[1]; i++)
                {
                    for (int j = 0; j < input.Shape[2]; j++)
                    {
                        paddedOutput[b, i + _padding[0], j + _padding[1], c] = input[b, i, j, c];
                    }
                }
            }
        }

        return ApplyActivation(paddedOutput);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        int batchSize = _lastInput.Shape[0];
        int channels = _lastInput.Shape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int i = 0; i < _lastInput.Shape[1]; i++)
                {
                    for (int j = 0; j < _lastInput.Shape[2]; j++)
                    {
                        inputGradient[b, i, j, c] = outputGradient[b, i + _padding[0], j + _padding[1], c];
                    }
                }
            }
        }

        return ApplyActivationDerivative(_lastInput, inputGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a padding layer
    }

    public override Vector<T> GetParameters()
    {
        // PaddingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    }
}