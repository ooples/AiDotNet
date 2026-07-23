namespace AiDotNet.Training;

internal sealed class StreamingAdamW8Bit<T> : StreamingAdam8Bit<T>
{
    public StreamingAdamW8Bit(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay)
        : base(learningRate, beta1, beta2, epsilon, weightDecay)
    {
    }
}

