namespace AiDotNet.NeuralNetworks.Layers;

public class PoolingLayer<T> : LayerBase<T>
{
    public int PoolSize { get; }
    public int Stride { get; }
    public PoolingType Type { get; }

    public override bool SupportsTraining => true;

    private Tensor<T>? _lastInput;
    private Tensor<int>? _maxIndices;

    public PoolingLayer(int inputDepth, int inputHeight, int inputWidth, int poolSize, int stride, PoolingType type = PoolingType.Max)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(inputDepth, CalculateOutputDimension(inputHeight, poolSize, stride), CalculateOutputDimension(inputWidth, poolSize, stride)))
    {
        PoolSize = poolSize;
        Stride = stride;
        Type = type;
    }

    private static int CalculateOutputDimension(int inputDim, int poolSize, int stride)
    {
        return (inputDim - poolSize) / stride + 1;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];

        int outputHeight = (inputHeight - PoolSize) / Stride + 1;
        int outputWidth = (inputWidth - PoolSize) / Stride + 1;

        var output = new Tensor<T>([batchSize, channels, outputHeight, outputWidth]);
        _maxIndices = new Tensor<int>(output.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outputHeight; h++)
                {
                    for (int w = 0; w < outputWidth; w++)
                    {
                        int hStart = h * Stride;
                        int wStart = w * Stride;
                        var poolRegion = input.GetSubTensor(b, c, hStart, wStart, PoolSize, PoolSize);

                        if (Type == PoolingType.Max)
                        {
                            (T maxVal, int maxIndex) = poolRegion.Max();
                            output[b, c, h, w] = maxVal;
                            _maxIndices[b, c, h, w] = maxIndex;
                        }
                        else if (Type == PoolingType.Average)
                        {
                            output[b, c, h, w] = poolRegion.Mean();
                        }
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
                for (int h = 0; h < outputGradient.Shape[2]; h++)
                {
                    for (int w = 0; w < outputGradient.Shape[3]; w++)
                    {
                        int hStart = h * Stride;
                        int wStart = w * Stride;

                        if (Type == PoolingType.Max)
                        {
                            if (_maxIndices != null)
                            {
                                int maxIndex = _maxIndices[b, c, h, w];
                                int maxH = hStart + maxIndex / PoolSize;
                                int maxW = wStart + maxIndex % PoolSize;
                                inputGradient[b, c, maxH, maxW] = NumOps.Add(inputGradient[b, c, maxH, maxW], outputGradient[b, c, h, w]);
                            }
                            else
                            {
                                throw new InvalidOperationException("_maxIndices is null. This should not happen.");
                            }
                        }
                        else if (Type == PoolingType.Average)
                        {
                            T gradValue = outputGradient[b, c, h, w];
                            gradValue = NumOps.Divide(gradValue, NumOps.FromDouble(PoolSize * PoolSize));
                            for (int ph = 0; ph < PoolSize; ph++)
                            {
                                for (int pw = 0; pw < PoolSize; pw++)
                                {
                                    inputGradient[b, c, hStart + ph, wStart + pw] = NumOps.Add(inputGradient[b, c, hStart + ph, wStart + pw], gradValue);
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // Pooling layers don't have trainable parameters, so this method does nothing.
    }

    public override Vector<T> GetParameters()
    {
        // PoolingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _maxIndices = null;
    }
}