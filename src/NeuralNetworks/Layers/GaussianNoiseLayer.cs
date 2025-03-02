namespace AiDotNet.NeuralNetworks.Layers;

public class GaussianNoiseLayer<T> : LayerBase<T>
{
    private readonly T _mean;
    private readonly T _standardDeviation;
    private Tensor<T>? _lastNoise;
    private readonly Random _random;
    private bool _isTraining;

    public override bool SupportsTraining => false;

    public GaussianNoiseLayer(
        int[] inputShape, 
        double standardDeviation = 0.1, 
        double mean = 0)
        : base(inputShape, inputShape)
    {
        _mean = NumOps.FromDouble(mean);
        _standardDeviation = NumOps.FromDouble(standardDeviation);
        _random = new Random();
        _isTraining = true;
    }

    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    public bool IsTraining()
    {
        return _isTraining;
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (IsTrainingMode)
        {
            _lastNoise = GenerateNoise(input.Shape);
            return input.Add(_lastNoise);
        }

        return input; // During inference, no noise is added
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // The gradient flows through unchanged
        return outputGradient;
    }

    private Tensor<T> GenerateNoise(int[] shape)
    {
        var noise = new Tensor<T>(shape);
        for (int i = 0; i < noise.Length; i++)
        {
            T u1 = NumOps.FromDouble(Random.NextDouble());
            T u2 = NumOps.FromDouble(Random.NextDouble());
            T z = NumOps.Multiply(
                NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(-2.0), NumOps.Log(u1))),
                MathHelper.Cos(NumOps.Multiply(NumOps.FromDouble(2.0 * Math.PI), u2))
            );
            noise[i] = NumOps.Add(_mean, NumOps.Multiply(_standardDeviation, z));
        }

        return noise;
    }

    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update for this layer
    }

    public override Vector<T> GetParameters()
    {
        // GaussianNoiseLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastNoise = null;
    }
}