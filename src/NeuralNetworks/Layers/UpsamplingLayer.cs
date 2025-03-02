namespace AiDotNet.NeuralNetworks.Layers;

public class UpsamplingLayer<T> : LayerBase<T>
{
    private readonly int _scaleFactor;
    private Tensor<T>? _lastInput;

    public override bool SupportsTraining => true;

    public UpsamplingLayer(int[] inputShape, int scaleFactor)
        : base(inputShape, CalculateOutputShape(inputShape, scaleFactor))
    {
        _scaleFactor = scaleFactor;
    }

    private static int[] CalculateOutputShape(int[] inputShape, int scaleFactor)
    {
        return
        [
            inputShape[0],
            inputShape[1] * scaleFactor,
            inputShape[2] * scaleFactor
        ];
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];
        int outputHeight = inputHeight * _scaleFactor;
        int outputWidth = inputWidth * _scaleFactor;

        var output = new Tensor<T>([batchSize, channels, outputHeight, outputWidth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        int sourceH = h / _scaleFactor;
                        int sourceW = w / _scaleFactor;
                        output[b, c, h, w] = input[b, c, sourceH, sourceW];
                    }
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
        int channels = _lastInput.Shape[1];
        int inputHeight = _lastInput.Shape[2];
        int inputWidth = _lastInput.Shape[3];

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < inputHeight; h++)
                {
                    for (int w = 0; w < inputWidth; w++)
                    {
                        T sum = NumOps.Zero;
                        for (int i = 0; i < _scaleFactor; i++)
                        {
                            for (int j = 0; j < _scaleFactor; j++)
                            {
                                int outputH = h * _scaleFactor + i;
                                int outputW = w * _scaleFactor + j;
                                sum = NumOps.Add(sum, outputGradient[b, c, outputH, outputW]);
                            }
                        }

                        inputGradient[b, c, h, w] = sum;
                    }
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
        // This layer doesn't have any trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear the cached input
        _lastInput = null;
    }
}