namespace AiDotNet.Training;

internal sealed class StreamingSgd8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    public StreamingSgd8Bit(double learningRate)
        : base(nameof(StreamingSgd8Bit<T>), learningRate, Array.Empty<bool>())
    {
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
        => parameter - ClampUpdate(LearningRate * gradient);
}

