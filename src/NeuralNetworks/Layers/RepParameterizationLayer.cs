namespace AiDotNet.NeuralNetworks.Layers;

public class RepParameterizationLayer<T> : LayerBase<T>
{
    private Tensor<T>? _lastMean;
    private Tensor<T>? _lastLogVar;
    private Tensor<T>? _lastEpsilon;

    public override bool SupportsTraining => true;

    public RepParameterizationLayer(int[] inputShape)
        : base(inputShape, inputShape)
    {
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int latentSize = input.Shape[1] / 2;

        _lastMean = new Tensor<T>([batchSize, latentSize]);
        _lastLogVar = new Tensor<T>([batchSize, latentSize]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < latentSize; j++)
            {
                _lastMean[i, j] = input[i, j];
                _lastLogVar[i, j] = input[i, j + latentSize];
            }
        }

        _lastEpsilon = new Tensor<T>([batchSize, latentSize]);
        var output = new Tensor<T>([batchSize, latentSize]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < latentSize; j++)
            {
                _lastEpsilon[i, j] = NumOps.FromDouble(Random.NextDouble());
                T stdDev = NumOps.Exp(NumOps.Multiply(_lastLogVar[i, j], NumOps.FromDouble(0.5)));
                output[i, j] = NumOps.Add(_lastMean[i, j], NumOps.Multiply(stdDev, _lastEpsilon[i, j]));
            }
        }

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastMean == null || _lastLogVar == null || _lastEpsilon == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = outputGradient.Shape[0];
        int latentSize = outputGradient.Shape[1];

        var inputGradient = new Tensor<T>([batchSize, latentSize * 2]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < latentSize; j++)
            {
                T stdDev = NumOps.Exp(NumOps.Multiply(_lastLogVar[i, j], NumOps.FromDouble(0.5)));
                
                // Gradient for mean
                inputGradient[i, j] = outputGradient[i, j];

                // Gradient for log variance
                T gradLogVar = NumOps.Multiply(
                    NumOps.Multiply(outputGradient[i, j], _lastEpsilon[i, j]),
                    NumOps.Multiply(stdDev, NumOps.FromDouble(0.5))
                );
                inputGradient[i, j + latentSize] = gradLogVar;
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
        // This layer has no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastMean = null;
        _lastLogVar = null;
        _lastEpsilon = null;
    }
}