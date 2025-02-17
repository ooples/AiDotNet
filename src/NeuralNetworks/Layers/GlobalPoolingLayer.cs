namespace AiDotNet.NeuralNetworks.Layers;

public class GlobalPoolingLayer<T> : LayerBase<T>
{
    private readonly PoolingType _poolingType;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;

    public GlobalPoolingLayer(int[] inputShape, PoolingType poolingType, IActivationFunction<T>? activationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape), activationFunction ?? new LinearActivation<T>())
    {
        _poolingType = poolingType;
    }

    public GlobalPoolingLayer(int[] inputShape, PoolingType poolingType, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape), vectorActivationFunction ?? new LinearActivation<T>())
    {
        _poolingType = poolingType;
    }

    private static int[] CalculateOutputShape(int[] inputShape)
    {
        // Global pooling reduces spatial dimensions to 1x1
        return [inputShape[0], 1, 1, inputShape[3]];
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int channels = input.Shape[3];
        int height = input.Shape[1];
        int width = input.Shape[2];

        var output = new Tensor<T>([batchSize, 1, 1, channels]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T pooledValue = _poolingType == PoolingType.Average ? NumOps.Zero : NumOps.MinValue;

                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        T value = input[b, h, w, c];
                        if (_poolingType == PoolingType.Average)
                        {
                            pooledValue = NumOps.Add(pooledValue, value);
                        }
                        else // Max pooling
                        {
                            pooledValue = MathHelper.Max(pooledValue, value);
                        }
                    }
                }

                if (_poolingType == PoolingType.Average)
                {
                    pooledValue = NumOps.Divide(pooledValue, NumOps.FromDouble(height * width));
                }

                output[b, 0, 0, c] = pooledValue;
            }
        }

        _lastOutput = ApplyActivation(output);
        return _lastOutput;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _lastInput.Shape[0];
        int height = _lastInput.Shape[1];
        int width = _lastInput.Shape[2];
        int channels = _lastInput.Shape[3];

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Apply activation derivative
        outputGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T gradientValue = outputGradient[b, 0, 0, c];

                if (_poolingType == PoolingType.Average)
                {
                    T averageGradient = NumOps.Divide(gradientValue, NumOps.FromDouble(height * width));
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            inputGradient[b, h, w, c] = averageGradient;
                        }
                    }
                }
                else // Max pooling
                {
                    T maxValue = NumOps.MinValue;
                    int maxH = 0, maxW = 0;

                    // Find the position of the maximum value
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            T value = _lastInput[b, h, w, c];
                            if (NumOps.GreaterThan(value, maxValue))
                            {
                                maxValue = value;
                                maxH = h;
                                maxW = w;
                            }
                        }
                    }

                    // Set the gradient only for the maximum value position
                    inputGradient[b, maxH, maxW, c] = gradientValue;
                }
            }
        }

        return inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a pooling layer
    }
}