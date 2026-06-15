namespace AiDotNet.Training;

internal sealed class StreamingAdagrad8Bit<T> : BlockQuantizedStreamingOptimizer<T>
{
    private readonly double _epsilon;

    public StreamingAdagrad8Bit(double learningRate, double epsilon)
        : base(nameof(StreamingAdagrad8Bit<T>), learningRate, new[] { false })
    {
        _epsilon = epsilon;
    }

    protected override double UpdateElement(double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index)
    {
        double acc = moments[0] + gradient * gradient;
        nextMoments[0] = acc;
        double update = LearningRate * gradient / (Math.Sqrt(acc) + _epsilon);
        return parameter - ClampUpdate(update);
    }
}

