namespace AiDotNet.NeuralNetworks.Layers;

public class MaskingLayer<T> : LayerBase<T>
{
    private readonly T _maskValue;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMask;

    public override bool SupportsTraining => false;

    public MaskingLayer(int[] inputShape, double maskValue = 0) : base(inputShape, inputShape)
    {
        _maskValue = NumOps.FromDouble(maskValue);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _lastMask = CreateMask(input);

        return ApplyMask(input, _lastMask);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastMask == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        return ApplyMask(outputGradient, _lastMask);
    }

    private Tensor<T> CreateMask(Tensor<T> input)
    {
        var mask = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Shape[0]; i++)
        {
            for (int j = 0; j < input.Shape[1]; j++)
            {
                for (int k = 0; k < input.Shape[2]; k++)
                {
                    mask[i, j, k] = !NumOps.Equals(input[i, j, k], _maskValue) ? NumOps.One : NumOps.Zero;
                }
            }
        }

        return mask;
    }

    private Tensor<T> ApplyMask(Tensor<T> input, Tensor<T> mask)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Shape[0]; i++)
        {
            for (int j = 0; j < input.Shape[1]; j++)
            {
                for (int k = 0; k < input.Shape[2]; k++)
                {
                    output[i, j, k] = NumOps.Multiply(input[i, j, k], mask[i, j, k]);
                }
            }
        }

        return output;
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    public override Vector<T> GetParameters()
    {
        // MaskingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastMask = null;
    }
}