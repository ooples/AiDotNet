namespace AiDotNet.NeuralNetworks.Layers;

public class CroppingLayer<T> : LayerBase<T>
{
    private readonly int[] _cropTop;
    private readonly int[] _cropBottom;
    private readonly int[] _cropLeft;
    private readonly int[] _cropRight;

    public CroppingLayer(
        int[] inputShape,
        int[] cropTop,
        int[] cropBottom,
        int[] cropLeft,
        int[] cropRight,
        IActivationFunction<T>? scalarActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, cropTop, cropBottom, cropLeft, cropRight), scalarActivation ?? new IdentityActivation<T>())
    {
        _cropTop = cropTop;
        _cropBottom = cropBottom;
        _cropLeft = cropLeft;
        _cropRight = cropRight;
    }

    public CroppingLayer(
        int[] inputShape,
        int[] cropTop,
        int[] cropBottom,
        int[] cropLeft,
        int[] cropRight,
        IVectorActivationFunction<T>? vectorActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, cropTop, cropBottom, cropLeft, cropRight), vectorActivation ?? new IdentityActivation<T>())
    {
        _cropTop = cropTop;
        _cropBottom = cropBottom;
        _cropLeft = cropLeft;
        _cropRight = cropRight;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int[] cropTop, int[] cropBottom, int[] cropLeft, int[] cropRight)
    {
        int[] outputShape = new int[inputShape.Length];
        for (int i = 0; i < inputShape.Length; i++)
        {
            outputShape[i] = inputShape[i] - cropTop[i] - cropBottom[i] - cropLeft[i] - cropRight[i];
        }

        return outputShape;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        int[] inputShape = input.Shape;
        int[] outputShape = GetOutputShape();
        Tensor<T> output = new Tensor<T>(outputShape);

        int batchSize = inputShape[0];
        int channels = inputShape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int y = 0; y < outputShape[1]; y++)
            {
                for (int x = 0; x < outputShape[2]; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        output[b, y, x, c] = input[b, y + _cropTop[1], x + _cropLeft[2], c];
                    }
                }
            }
        }

        return ApplyActivation(output);
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        int[] inputShape = GetInputShape();
        Tensor<T> inputGradient = new Tensor<T>(inputShape);

        int batchSize = inputShape[0];
        int channels = inputShape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int y = 0; y < outputGradient.Shape[1]; y++)
            {
                for (int x = 0; x < outputGradient.Shape[2]; x++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        inputGradient[b, y + _cropTop[1], x + _cropLeft[2], c] = outputGradient[b, y, x, c];
                    }
                }
            }
        }

        return ApplyActivationDerivative(inputGradient, outputGradient);
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a cropping layer
    }
}