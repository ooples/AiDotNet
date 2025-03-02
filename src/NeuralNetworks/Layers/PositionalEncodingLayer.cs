namespace AiDotNet.NeuralNetworks.Layers;

public class PositionalEncodingLayer<T> : LayerBase<T>
{
    private readonly int maxSequenceLength;
    private readonly int embeddingSize;
    private Tensor<T> encodings;

    public override bool SupportsTraining => true;

    public PositionalEncodingLayer(int maxSequenceLength, int embeddingSize)
        : base([maxSequenceLength, embeddingSize], [maxSequenceLength, embeddingSize])
    {
        this.maxSequenceLength = maxSequenceLength;
        this.embeddingSize = embeddingSize;
        encodings = new Tensor<T>([maxSequenceLength, embeddingSize]);

        InitializeEncodings();
    }

    private void InitializeEncodings()
    {
        for (int pos = 0; pos < maxSequenceLength; pos++)
        {
            for (int i = 0; i < embeddingSize; i++)
            {
                double angle = pos / Math.Pow(10000, (2 * (i / 2)) / (double)embeddingSize);

                if (i % 2 == 0)
                {
                    encodings[pos, i] = NumOps.FromDouble(Math.Sin(angle));
                }
                else
                {
                    encodings[pos, i] = NumOps.FromDouble(Math.Cos(angle));
                }
            }
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Shape[0] > maxSequenceLength)
        {
            throw new ArgumentException($"Input sequence length {input.Shape[0]} exceeds maximum sequence length {maxSequenceLength}");
        }

        var slicedEncodings = encodings.Slice(0, 0, input.Shape[0], embeddingSize);
        return input + slicedEncodings;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // The gradient flows through unchanged
        return outputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    public override Vector<T> GetParameters()
    {
        // PositionalEncodingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // No state to reset in this layer
        // The encodings are fixed and don't change during training
    }
}