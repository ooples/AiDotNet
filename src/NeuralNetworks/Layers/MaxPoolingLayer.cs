namespace AiDotNet.NeuralNetworks.Layers;

public class MaxPoolingLayer<T> : LayerBase<T>
{
    public int PoolSize { get; private set; }
    public int Strides { get; private set; }

    public override bool SupportsTraining => true;

    private Tensor<int> _maxIndices;

    public MaxPoolingLayer(int[] inputShape, int poolSize, int strides) 
        : base(inputShape, CalculateOutputShape(inputShape, poolSize, strides))
    {
        PoolSize = poolSize;
        Strides = strides;
        _maxIndices = new Tensor<int>(OutputShape);
    }

    private static int[] CalculateOutputShape(int[] inputShape, int poolSize, int strides)
    {
        int outputHeight = (inputShape[1] - poolSize) / strides + 1;
        int outputWidth = (inputShape[2] - poolSize) / strides + 1;

        return [inputShape[0], outputHeight, outputWidth];
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape.Length != 3)
            throw new ArgumentException("Input tensor must have 3 dimensions (channels, height, width)");

        int channels = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int outputHeight = OutputShape[1];
        int outputWidth = OutputShape[2];

        var output = new Tensor<T>(OutputShape);
        _maxIndices = new Tensor<int>(OutputShape);

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    T maxVal = NumOps.Zero;
                    int maxIdx = -1;

                    for (int ph = 0; ph < PoolSize; ph++)
                    {
                        for (int pw = 0; pw < PoolSize; pw++)
                        {
                            int ih = h * Strides + ph;
                            int iw = w * Strides + pw;

                            if (ih < inputHeight && iw < inputWidth)
                            {
                                T val = input[c, ih, iw];
                                if (maxIdx == -1 || NumOps.GreaterThan(maxVal, NumOps.Zero))
                                {
                                    maxVal = val;
                                    maxIdx = ph * PoolSize + pw;
                                }
                            }
                        }
                    }

                    output[c, h, w] = maxVal;
                    _maxIndices[c, h, w] = maxIdx;
                }
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (outputGradient.Shape.Length != 3)
            throw new ArgumentException("Output gradient tensor must have 3 dimensions (channels, height, width)");

        int channels = InputShape[0];
        int inputHeight = InputShape[1];
        int inputWidth = InputShape[2];

        var inputGradient = new Tensor<T>(InputShape);

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < outputGradient.Shape[1]; h++)
            {
                for (int w = 0; w < outputGradient.Shape[2]; w++)
                {
                    int maxIdx = _maxIndices[c, h, w];
                    int ph = maxIdx / PoolSize;
                    int pw = maxIdx % PoolSize;

                    int ih = h * Strides + ph;
                    int iw = w * Strides + pw;

                    if (ih < inputHeight && iw < inputWidth)
                    {
                        inputGradient[c, ih, iw] = outputGradient[c, h, w];
                    }
                }
            }
        }

        return inputGradient;
    }

    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(PoolSize);
        writer.Write(Strides);
    }

    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        PoolSize = reader.ReadInt32();
        Strides = reader.ReadInt32();
    }

    public override IEnumerable<ActivationFunction> GetActivationTypes()
    {
        // Max pooling doesn't have an activation function
        return [];
    }

    public override void UpdateParameters(T learningRate)
    {
        // Max pooling layer doesn't have trainable parameters
    }

    public override Vector<T> GetParameters()
    {
        // MaxPoolingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _maxIndices = new Tensor<int>(OutputShape);
    }
}