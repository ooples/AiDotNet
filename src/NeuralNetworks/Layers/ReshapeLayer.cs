namespace AiDotNet.NeuralNetworks.Layers;

public class ReshapeLayer<T> : LayerBase<T>
{
    private int[] _inputShape;
    private int[] _outputShape;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    public ReshapeLayer(int[] inputShape, int[] outputShape)
        : base(inputShape, outputShape)
    {
        _inputShape = inputShape;
        _outputShape = outputShape;

        if (inputShape.Aggregate(1, (a, b) => a * b) != outputShape.Aggregate(1, (a, b) => a * b))
        {
            throw new ArgumentException("Input and output shapes must have the same total number of elements.");
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var output = new Tensor<T>([batchSize, .. _outputShape]);

        for (int i = 0; i < batchSize; i++)
        {
            ReshapeForward(input, i, new int[_inputShape.Length], output, i, new int[_outputShape.Length]);
        }

        return output;
    }

    private void ReshapeForward(Tensor<T> input, int inputBatchIndex, int[] inputIndices,
                                Tensor<T> output, int outputBatchIndex, int[] outputIndices)
    {
        if (inputIndices.Length == _inputShape.Length)
        {
            output[[outputBatchIndex, .. outputIndices]] =
                input[[inputBatchIndex, .. inputIndices]];
            return;
        }

        for (int i = 0; i < _inputShape[inputIndices.Length]; i++)
        {
            inputIndices[inputIndices.Length - 1] = i;
            outputIndices[outputIndices.Length - 1] = i % _outputShape[outputIndices.Length - 1];
            if (i > 0 && i % _outputShape[outputIndices.Length - 1] == 0)
            {
                IncrementIndices(outputIndices);
            }

            ReshapeForward(input, inputBatchIndex, inputIndices, output, outputBatchIndex, outputIndices);
        }
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        int batchSize = outputGradient.Shape[0];

        for (int i = 0; i < batchSize; i++)
        {
            ReshapeBackward(outputGradient, i, new int[_outputShape.Length], inputGradient, i, new int[_inputShape.Length]);
        }

        return inputGradient;
    }

    private void ReshapeBackward(Tensor<T> outputGradient, int outputBatchIndex, int[] outputIndices,
                                 Tensor<T> inputGradient, int inputBatchIndex, int[] inputIndices)
    {
        if (outputIndices.Length == _outputShape.Length)
        {
            inputGradient[[inputBatchIndex, .. inputIndices]] =
                outputGradient[[outputBatchIndex, .. outputIndices]];
            return;
        }

        for (int i = 0; i < _outputShape[outputIndices.Length]; i++)
        {
            outputIndices[outputIndices.Length - 1] = i;
            inputIndices[inputIndices.Length - 1] = i % _inputShape[inputIndices.Length - 1];
            if (i > 0 && i % _inputShape[inputIndices.Length - 1] == 0)
            {
                IncrementIndices(inputIndices);
            }

            ReshapeBackward(outputGradient, outputBatchIndex, outputIndices, inputGradient, inputBatchIndex, inputIndices);
        }
    }

    private void IncrementIndices(int[] indices)
    {
        for (int i = indices.Length - 2; i >= 0; i--)
        {
            indices[i]++;
            if (indices[i] < _outputShape[i])
            {
                break;
            }

            indices[i] = 0;
        }
    }

    public override void UpdateParameters(T learningRate)
    {
        // ReshapeLayer has no parameters to update
    }

    public override Vector<T> GetParameters()
    {
        // ReshapeLayer has no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    }
}