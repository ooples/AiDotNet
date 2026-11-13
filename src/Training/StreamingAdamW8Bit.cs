namespace AiDotNet.Training;

internal sealed class StreamingAdamW8Bit<T> : StreamingAdam8Bit<T>
{
    public StreamingAdamW8Bit(
        double learningRate,
        double beta1,
        double beta2,
        double epsilon,
        double weightDecay,
        int blockSize = 2048,
        bool useBf16Moments = false,
        int min8BitSize = 0)
        : base(
            learningRate,
            beta1,
            beta2,
            epsilon,
            weightDecay,
            blockSize: blockSize,
            decoupledWeightDecay: true,
            useBf16Moments: useBf16Moments,
            min8BitSize: min8BitSize)
    {
    }
}
