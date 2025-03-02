namespace AiDotNet.NeuralNetworks.Layers;

public class FlattenLayer<T> : LayerBase<T>
{
    private int[] _inputShape;
    private int _outputSize;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => false;

    public FlattenLayer(int[] inputShape)
        : base(inputShape, [inputShape.Aggregate(1, (a, b) => a * b)])
    {
        _inputShape = inputShape;
        _outputSize = inputShape.Aggregate(1, (a, b) => a * b);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var output = new Tensor<T>([batchSize, _outputSize]);

        for (int i = 0; i < batchSize; i++)
        {
            int flatIndex = 0;
            FlattenRecursive(input, i, new int[_inputShape.Length], ref flatIndex, output);
        }

        return output;
    }

    private void FlattenRecursive(Tensor<T> input, int batchIndex, int[] indices, ref int flatIndex, Tensor<T> output)
    {
        if (indices.Length == _inputShape.Length)
        {
            output[batchIndex, flatIndex++] = input[new int[] { batchIndex }.Concat(indices).ToArray()];
            return;
        }

        for (int i = 0; i < _inputShape[indices.Length]; i++)
        {
            indices[indices.Length - 1] = i;
            FlattenRecursive(input, batchIndex, indices, ref flatIndex, output);
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
            int flatIndex = 0;
            UnflattenRecursive(outputGradient, i, new int[_inputShape.Length], ref flatIndex, inputGradient);
        }

        return inputGradient;
    }

    private void UnflattenRecursive(Tensor<T> outputGradient, int batchIndex, int[] indices, ref int flatIndex, Tensor<T> inputGradient)
    {
        if (indices.Length == _inputShape.Length)
        {
            inputGradient[new int[] { batchIndex }.Concat(indices).ToArray()] = outputGradient[batchIndex, flatIndex++];
            return;
        }

        for (int i = 0; i < _inputShape[indices.Length]; i++)
        {
            indices[indices.Length - 1] = i;
            UnflattenRecursive(outputGradient, batchIndex, indices, ref flatIndex, inputGradient);
        }
    }

    public override void UpdateParameters(T learningRate)
    {
        // FlattenLayer has no parameters to update
    }

    public override Vector<T> GetParameters()
    {
        // FlattenLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    }
}