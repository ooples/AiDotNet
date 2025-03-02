namespace AiDotNet.NeuralNetworks.Layers;

public class SplitLayer<T> : LayerBase<T>
{
    private readonly int _numSplits;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    public SplitLayer(int[] inputShape, int numSplits)
        : base(inputShape, CalculateOutputShape(inputShape, numSplits))
    {
        _numSplits = numSplits;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int numSplits)
    {
        if (inputShape[0] % numSplits != 0)
        {
            throw new ArgumentException("Input size must be divisible by the number of splits");
        }

        return [inputShape[0] / numSplits];
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputSize = input.Shape[1];
        int splitSize = inputSize / _numSplits;

        var output = new Tensor<T>([batchSize, _numSplits, splitSize]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _numSplits; j++)
            {
                for (int k = 0; k < splitSize; k++)
                {
                    output[i, j, k] = input[i, j * splitSize + k];
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape[1];
        int splitSize = inputSize / _numSplits;

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _numSplits; j++)
            {
                for (int k = 0; k < splitSize; k++)
                {
                    inputGradient[i, j * splitSize + k] = outputGradient[i, j, k];
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    public override Vector<T> GetParameters()
    {
        // SplitLayer has no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    }
}